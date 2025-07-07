import asyncio
import os
import io
import re
import tempfile
import logging
from typing import List, Dict, Any, Optional, Union, Tuple, Set
from dotenv import load_dotenv
import uuid
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.embeddings import Embeddings
from pydantic import BaseModel, Field as PydanticField
import json
import functools # <--- เพิ่มบรรทัดนี้
from datetime import datetime, timezone
from functools import partial # For unstructured_partition callable
# from unstructured.partition.auto import partition as unstructured_partition # Keep this for analyze_pdf_impact
import traceback
from sklearn.metrics.pairwise import cosine_similarity # For similarity checks
import numpy as np
from sklearn.cluster import HDBSCAN # หรือ KMeans if you prefer
# Make sure HDBSCAN is installed: pip install hdbscan scikit-learn
# For KMeans: from sklearn.cluster import KMeans
import igraph as ig
from app.models.esg_question_model import ESGQuestion, SubQuestionDetail
from app.services.neo4j_service import Neo4jService # Neo4jService is imported
from app.models.esg_question_model import RelatedSETQuestion
from app.services.persistence.mongodb import MongoDBSaver
from app.services.rate_limit import RateLimiter
from llama_index.core.llms import LLM
from llama_index.core.settings import Settings
from app.data.set_benchmarks import benchmark_questions as all_set_benchmarks # <-- Import directly

load_dotenv()

TARGET_GRI_STANDARDS = [
    'GRI-302', 'GRI-303', 'GRI-306', 'GRI-401', 'GRI-403', 
    'GRI-404', 'GRI-408', 'GRI-205'
]
FINAL_VALIDATION_SIMILARITY_THRESHOLD = 0.90 
SIMILARITY_THRESHOLD_KG_THEME_UPDATE = 0.85
DEFAULT_NODE_TYPE = "UnknownEntityType"

# PROCESS_SCOPE constants remain the same
PROCESS_SCOPE_PDF_SPECIFIC_IMPACT = "pdf_specific_impact" # Not directly used by new theme logic, but analyze_pdf_impact might still be
PROCESS_SCOPE_KG_INITIAL_FULL_SCAN_FROM_CONTENT = "KG_INITIAL_FULL_SCAN_FROM_CONTENT"
PROCESS_SCOPE_KG_UPDATED_FULL_SCAN_FROM_CONTENT = "KG_UPDATED_FULL_SCAN_FROM_CONTENT"
PROCESS_SCOPE_KG_FULL_SCAN_DB_EMPTY_FROM_CONTENT = "KG_FULL_SCAN_DB_EMPTY_FROM_CONTENT"
PROCESS_SCOPE_KG_SCHEDULED_REFRESH_FROM_CONTENT = "KG_SCHEDULED_REFRESH_FROM_CONTENT"
PROCESS_SCOPE_NO_ACTION_DEFAULT = "NO_ACTION_DEFAULT"
PROCESS_SCOPE_NO_ACTION_UNEXPECTED_STATE = "NO_ACTION_UNEXPECTED_STATE"
PROCESS_SCOPE_NO_ACTION_KG_AND_DB_EMPTY_ON_SCHEDULED_RUN = "NO_ACTION_KG_AND_DB_EMPTY_ON_SCHEDULED_RUN"

DEFAULT_NODE_TYPE = "UnknownEntityType"

# GeneratedQuestion Pydantic model (ใช้ภายใน service) อาจจะยังคงเดิม หรือเพิ่ม model_extra
class GeneratedQuestion(BaseModel):
    question_text_en: str = PydanticField(..., description="The text of the generated ESG question.")
    question_text_th: Optional[str] = PydanticField(None, description="The Thai text of the generated ESG question.")
    category: str = PydanticField(..., description="The primary ESG dimension (E, S, or G) for this question's theme/category.")
    theme: str = PydanticField(..., description="The Main ESG Category name this question belongs to.")
    sub_theme_name: Optional[str] = PydanticField(None, description="The specific Consolidated Sub-Theme name, if this is a sub-question.")
    is_main_question: bool = PydanticField(False, description="True if this is a Main Question for the 'theme' (Main Category).")
    additional_info: Optional[Dict[str, Any]] = PydanticField(None, description="Additional non-standard information, e.g., detailed_source_info_for_subquestions.")

class GeneratedQuestionSet(BaseModel):
    # Main Question part
    main_question_text_en: str
    # Sub-questions part (rolled-up)
    rolled_up_sub_questions_text_en: str

    # Common metadata for the Main Category
    main_category_name: str
    main_category_dimension: str # E, S, G  <--- THIS IS THE REQUIRED FIELD
    main_category_keywords: Optional[str] = None
    main_category_description: Optional[str] = None
    main_category_constituent_entities: Optional[List[str]] = None
    main_category_source_docs: Optional[List[str]] = None

    # Source info specifically for the sub-questions set
    detailed_source_info_for_subquestions: Optional[str] = None

class QuestionGenerationService:
    SIMILARITY_THRESHOLD_KG_THEME_UPDATE = 0.85 # Threshold for considering a new KG theme an update to an old one

    def __init__(self,
                 llm: ChatGoogleGenerativeAI, 
                 neo4j_service: Neo4jService,
                 mongodb_service: MongoDBSaver,
                 similarity_embedding_model: Embeddings,
                 rate_limiter: Optional[RateLimiter] = None,
                ):
        self.llm = llm
        self.qg_llm = llm 
        self.neo4j_service = neo4j_service
        self.similarity_llm_embedding = similarity_embedding_model # Used for are_questions_substantially_similar
        self.mongodb_service = mongodb_service # <--- เพิ่ม
        
        self.qg_llm = llm or ChatGoogleGenerativeAI( # LLM for theme naming, question generation
            model=os.getenv("QUESTION_GENERATION_MODEL", "gemini-2.5-flash-preview-05-20"), # Updated model name
            temperature=0.4, 
            top_p=0.9,
            top_k=40,
            max_retries=3,
            rate_limiter=rate_limiter or RateLimiter(requests_per_minute=int(os.getenv("REQUESTS_PER_MINUTE", "60")))

        )
        self.translation_llm = ChatGoogleGenerativeAI(
            model=os.getenv("TRANSLATION_MODEL", "gemini-2.5-flash-preview-05-20"), # Updated model name
            temperature=0.8,
            max_retries=3,
        )
        self.rate_limiter = rate_limiter or RateLimiter(requests_per_minute=int(os.getenv("REQUESTS_PER_MINUTE", "60"))) # <--- เพิ่ม
        logging.basicConfig(level=logging.INFO) # <--- เพิ่ม
        self.logger = logging.getLogger(__name__) # <--- เพิ่ม
        # self._load_graph_schema() # Remove this call or its usage for themes

        if self.similarity_llm_embedding:
            print("[QG_SERVICE LOG] Similarity embedding model received and configured.")
        else:
            print("[QG_SERVICE WARNING] Similarity embedding model was NOT provided. Similarity checks might fallback or be less accurate.")

    # _load_graph_schema method can be removed if no longer used elsewhere.

    async def _llm_generate_search_keywords(self, set_question_topic: str) -> List[str]:
        """
        Uses an LLM to generate alternative search keywords for a given ESG topic.
        """
        self.logger.info(f"Generating expanded search keywords for topic: {set_question_topic}")
        prompt = f"""
        You are a Thai ESG expert. For the SET topic "{set_question_topic}", generate 5-10 related search keywords and phrases that a company might use in its sustainability report.
        For example, if the topic is 'Energy Management', keywords could be 'reducing electricity consumption', 'energy efficiency', 'solar rooftop'.
        Return the result as a JSON list of strings only.

        Example output:
        ["keyword1", "phrase for keyword 2", "keyword3"]
        """
        try:
            response = await self.llm.acomplete(prompt)
            keywords = json.loads(response.text)
            if isinstance(keywords, list):
                self.logger.info(f"Generated keywords: {keywords}")
                return keywords
        except Exception as e:
            self.logger.error(f"Failed to generate or parse keywords for '{set_question_topic}': {e}")
        return []

    async def translate_text_to_thai(
        self,
        text_to_translate: str,
        category_name: Optional[str] = None, # Context: Main Category Name or Theme Name
        keywords: Optional[str] = None       # Context: Keywords for the category/theme
    ) -> Optional[str]:
        if not text_to_translate:
            return None

        context_info = ""
        if category_name:
            context_info += f"The text relates to the ESG category or theme: '{category_name}'. "
        if keywords:
            context_info += f"Relevant keywords include: '{keywords}'. "

        prompt_template_str = (
            "You are an expert translator specializing in ESG (Environmental, Social, and Governance) terminology "
            "for Thai corporate reporting, particularly for the industrial sector.\n"
            "Translate the following English ESG text (which could be a question, theme name, description, or keywords) "
            "to clear, natural-sounding, and accurate Thai, suitable for an ESG questionnaire or report.\n"
            "Ensure the translation is professional and uses appropriate ESG vocabulary.\n"
        )
        if context_info:
            prompt_template_str += f"Contextual Information: {context_info.strip()}\n\n"
        else:
            prompt_template_str += "\n"

        prompt_template_str += "English Text: \"{english_text}\"\n\nThai Translation (Provide only the Thai translation):"
        
        prompt_template = PromptTemplate.from_template(prompt_template_str)
        formatted_prompt = prompt_template.format(english_text=text_to_translate)
        
        try:
            # print(f"[QG_SERVICE DEBUG /translate] Prompt for translation:\n{formatted_prompt}") # For debugging
            response = await self.translation_llm.ainvoke(formatted_prompt)
            translated_text = response.content.strip()
            # Basic check to ensure it's likely Thai (can be improved)
            if translated_text and not re.match(r"^[a-zA-Z0-9\s\W]*$", translated_text): # If not purely English/numbers/symbols
                return translated_text
            else:
                print(f"[QG_SERVICE WARNING /translate] Translation for '{text_to_translate[:30]}...' might not be Thai or is empty. LLM output: '{translated_text}'")
                return text_to_translate # Fallback to original if translation looks suspicious or is empty
        except Exception as e:
            print(f"[QG_SERVICE ERROR /translate] Error during translation to Thai for text '{text_to_translate[:30]}...': {e}")
            return None # Or return original text_to_translate as fallback

    async def _get_entity_graph_data(self,
                                     exclude_entity_ids: Optional[Set[str]] = None,
                                     include_only_entity_ids: Optional[List[str]] = None,
                                     doc_ids: Optional[List[str]] = None) -> Optional[Dict[str, Any]]:
        self.logger.info("Fetching __Entity__ graph data from Neo4j...")
        params = {}
        node_match_clause = ""

        # --- FIX START: แก้ไขการสร้าง Cypher Query ---
        if doc_ids:
            self.logger.info(f"Filtering graph data by doc_ids: {doc_ids}")
            # กำหนดตัวแปร sc ให้กับ StandardChunk node ตรงนี้
            node_match_clause = "MATCH (e:__Entity__)<--(sc:StandardChunk) WHERE sc.doc_id IN $doc_ids"
            params['doc_ids'] = doc_ids
        elif include_only_entity_ids:
            self.logger.info(f"Filtering graph data to include only {len(include_only_entity_ids)} entities.")
            node_match_clause = "MATCH (e:__Entity__) WHERE e.id IN $include_ids"
            params['include_ids'] = include_only_entity_ids
        else:
            node_match_clause = "MATCH (e:__Entity__)"
            if exclude_entity_ids:
                self.logger.info(f"Filtering graph data to exclude {len(exclude_entity_ids)} entities.")
                # ส่วนนี้ไม่จำเป็นต้องมี sc เพราะไม่ได้กรองด้วย doc_id
                node_match_clause += " WHERE NOT e.id IN $exclude_ids"
                params['exclude_ids'] = list(exclude_entity_ids)
        # --- FIX END ---

        # Query ส่วนที่เหลือยังคงเดิม และจะทำงานถูกต้องกับ node_match_clause ที่แก้ไขแล้ว
        query_nodes = f"""
        {node_match_clause}
        WITH DISTINCT e
        OPTIONAL MATCH (sc:StandardChunk)-[:CONTAINS_ENTITY]->(e)
        RETURN
            e.id AS id,
            COALESCE(e.description, '') AS description,
            [lbl IN labels(e) WHERE lbl <> '__Entity__'] AS specific_labels,
            COLLECT(DISTINCT {{doc_id: sc.doc_id, standard_code: sc.standard_code}}) as sources
        """

        query_edges = f"""
        {node_match_clause}
        WITH COLLECT(DISTINCT e.id) as valid_ids
        MATCH (e1:__Entity__)-[r]-(e2:__Entity__)
        WHERE e1.id IN valid_ids AND e2.id IN valid_ids AND e1.id <> e2.id
        RETURN DISTINCT e1.id AS source, e2.id AS target
        """

        try:
            nodes_result = await self.run_in_executor(self.neo4j_service.graph.query, query_nodes, params)
            edges_result = await self.run_in_executor(self.neo4j_service.graph.query, query_edges, params)

            nodes_map = self._process_node_results(nodes_result)
            edges = self._process_edge_results(edges_result, nodes_map)

            self.logger.info(f"Fetched {len(nodes_map)} nodes and {len(edges)} edges based on filters.")
            if not nodes_map:
                self.logger.warning("No nodes found with the current filter criteria.")
                return None
            return {"nodes_map": nodes_map, "edges": edges}

        except Exception as e:
            self.logger.error(f"Error fetching filtered __Entity__ graph data: {e}", exc_info=True)
            return None

    async def _detect_first_order_communities(self, entity_graph_data: Dict[str, Any], min_community_size: int = 2) -> Dict[int, List[str]]:
        # ... (Method from your latest code) ...
        if not entity_graph_data or not entity_graph_data.get("nodes_map"): return {}
        nodes_map = entity_graph_data["nodes_map"]; edges_info = entity_graph_data.get("edges", [])
        if not nodes_map: return {}
        entity_ids_ordered = list(nodes_map.keys()); entity_id_to_idx = {id_val: i for i, id_val in enumerate(entity_ids_ordered)}
        igraph_edges = [(entity_id_to_idx[s], entity_id_to_idx[t]) for s, t in edges_info if s in entity_id_to_idx and t in entity_id_to_idx]
        num_vertices = len(entity_ids_ordered)
        if num_vertices == 0: return {}
        g = ig.Graph(n=num_vertices, edges=igraph_edges, directed=False)
        print(f"[QG_SERVICE INFO] Running igraph Louvain on __Entity__ graph ({g.vcount()} V, {g.ecount()} E) for 1st order communities.")
        communities_result: Dict[int, List[str]] = {}
        if g.vcount() == 0: return communities_result
        try:
            vc = g.community_multilevel(weights=None, return_levels=False)
            for c_id, member_indices in enumerate(vc):
                if len(member_indices) >= min_community_size:
                    community_entity_ids = [entity_ids_ordered[idx] for idx in member_indices if idx < len(entity_ids_ordered)]
                    if community_entity_ids: communities_result[c_id] = community_entity_ids
            print(f"[QG_SERVICE INFO] igraph Louvain (1st order) found {len(communities_result)} communities (>= {min_community_size} members).")
        except Exception as e:
            print(f"[QG_SERVICE ERROR] igraph Louvain (1st order) error: {e}"); traceback.print_exc()
            if 1 < g.vcount() <= 30: return {i: [entity_ids_ordered[i]] for i in range(num_vertices) if i < len(entity_ids_ordered)}
        return communities_result

    async def _create_community_meta_graph(self, first_order_communities_map: Dict[int, List[str]], entity_graph_data: Dict[str, Any], min_inter_community_edges_for_link: int = 1) -> Optional[ig.Graph]:
        # ... (Method from your latest code) ...
        if not first_order_communities_map: return None
        print(f"[QG_SERVICE LOG] Creating meta-graph from {len(first_order_communities_map)} first-order communities.")
        fo_community_ids_ordered = sorted(list(first_order_communities_map.keys())); fo_community_id_to_meta_idx = {c_id: i for i, c_id in enumerate(fo_community_ids_ordered)}
        meta_graph_edges: List[Tuple[int, int]] = []
        original_entity_edges = entity_graph_data.get("edges", [])
        entity_to_fo_community_id_map: Dict[str, int] = {eid: fo_cid for fo_cid, eids in first_order_communities_map.items() for eid in eids}
        inter_community_links_counts: Dict[Tuple[int, int], int] = {}
        for src_e, tgt_e in original_entity_edges:
            c1, c2 = entity_to_fo_community_id_map.get(src_e), entity_to_fo_community_id_map.get(tgt_e)
            if c1 is not None and c2 is not None and c1 != c2: inter_community_links_counts[tuple(sorted((c1, c2)))] = inter_community_links_counts.get(tuple(sorted((c1, c2))), 0) + 1
        for (c1_orig, c2_orig), count in inter_community_links_counts.items():
            if count >= min_inter_community_edges_for_link and c1_orig in fo_community_id_to_meta_idx and c2_orig in fo_community_id_to_meta_idx:
                meta_graph_edges.append((fo_community_id_to_meta_idx[c1_orig], fo_community_id_to_meta_idx[c2_orig]))
        if not fo_community_ids_ordered: return None
        meta_g = ig.Graph(n=len(fo_community_ids_ordered), edges=meta_graph_edges, directed=False)
        meta_g.vs["original_fo_community_id"] = fo_community_ids_ordered
        print(f"[QG_SERVICE INFO] Meta-graph created with {meta_g.vcount()} meta-nodes and {meta_g.ecount()} meta-edges.")
        return meta_g

    async def _detect_main_categories_from_meta_graph(self, meta_graph: Optional[ig.Graph], min_main_category_fo_community_count: int = 1) -> Dict[int, List[int]]:
        # ... (Method from your latest code, ensure parameter name matches) ...
        if not meta_graph or meta_graph.vcount() == 0:
            if meta_graph and meta_graph.vcount() > 0: return {i: [meta_graph.vs[i]["original_fo_community_id"]] for i in range(meta_graph.vcount())}
            return {}
        print(f"[QG_SERVICE INFO] Running Louvain on meta-graph ({meta_graph.vcount()} V, {meta_graph.ecount()} E) for main categories.")
        main_categories_result: Dict[int, List[int]] = {}; 
        if meta_graph.vcount() == 0: return main_categories_result
        try:
            vc_meta = meta_graph.community_multilevel(weights=None, return_levels=False)
            for mc_raw_id, meta_member_indices in enumerate(vc_meta):
                if len(meta_member_indices) >= min_main_category_fo_community_count: # Use correct param name
                    main_categories_result[mc_raw_id] = [meta_graph.vs[meta_idx]["original_fo_community_id"] for meta_idx in meta_member_indices]
            print(f"[QG_SERVICE INFO] Louvain (2nd order) found {len(main_categories_result)} main categories (>= {min_main_category_fo_community_count} FO communities).")
        except Exception as e:
            print(f"[QG_SERVICE ERROR] Louvain (2nd order) error: {e}"); traceback.print_exc()
            if meta_graph.vcount() > 0: return {i: [meta_graph.vs[i]["original_fo_community_id"]] for i in range(meta_graph.vcount())}
        return main_categories_result

    async def _generate_main_category_details_with_llm(
        self, main_category_raw_id: int, consolidated_themes_for_context: List[Dict[str, Any]], 
        all_source_docs_for_mc: Set[str], all_constituent_entity_ids_for_mc: Set[str]
    ) -> Optional[Dict[str, Any]]:
        # ... (Method from your latest code - ensure it returns "main_category_name_en", "consolidated_themes" list etc.)
        print(f"[QG_SERVICE LOG] LLM generating details for Main Category (raw_id: {main_category_raw_id}). Using {len(consolidated_themes_for_context)} sub-themes for context.")
        context_sub_themes_for_llm: List[str] = []
        char_count_mc, max_chars_mc_context, MAX_SUBTHEMES_TO_SAMPLE_FOR_MC_PROMPT = 0, 15000, 7 # Adjusted max_chars
        for sub_theme_data in consolidated_themes_for_context[:MAX_SUBTHEMES_TO_SAMPLE_FOR_MC_PROMPT]:
            name, desc = sub_theme_data.get("theme_name_en", "N/A"), sub_theme_data.get("description_en", "N/A")
            keywords, dim = sub_theme_data.get("keywords_en", "N/A"), sub_theme_data.get("dimension", "N/A")
            theme_entry = f"Sub-Theme Name: {name}\n  Description: {desc}\n  Keywords: {keywords}\n  Dimension: {dim}\n"
            if char_count_mc + len(theme_entry) <= max_chars_mc_context:
                context_sub_themes_for_llm.append(theme_entry); char_count_mc += len(theme_entry)
            else: break
        final_context_for_llm = "\n---\n".join(context_sub_themes_for_llm)
        if not final_context_for_llm.strip():
            print(f"[QG_SERVICE WARNING] No sub-theme context for Main Category (raw_id: {main_category_raw_id})."); return None
        prompt_template_str = """
        You are an expert ESG analyst tasked with defining overarching Main ESG Categories.
        You are given a group of related, detailed ESG themes that have been automatically clustered. This group represents a potential Main ESG Category.
        Group of Detailed ESG Themes:
        --- DETAILED THEMES START ---
        {grouped_detailed_themes_content}
        --- DETAILED THEMES END ---
        These themes primarily originate from or relate to standards document(s) such as: {source_standards_list_str}
        Based on ALL the provided information for this group:
        1.  Propose a concise and descriptive **Main Category Name** (in English) for this entire group. This name should be broad, suitable for structuring an ESG report for the **industrial packaging sector**.
        2.  Provide a brief **Main Category Description** (in English, 1-2 sentences) that clearly explains its overall scope.
        3.  Determine the most relevant primary **ESG Dimension** ('E', 'S', or 'G') for this Main Category.
        4.  Suggest 3-5 broad yet relevant **Main Category Keywords** (in English, comma-separated).
        Output ONLY a single, valid JSON object with the keys: "main_category_name_en", "main_category_description_en", "dimension", "main_category_keywords_en".
        """
        prompt = PromptTemplate.from_template(prompt_template_str)
        source_docs_str = ", ".join(sorted(list(all_source_docs_for_mc))[:3]) if all_source_docs_for_mc else "various ESG standards"
        formatted_prompt = prompt.format(grouped_detailed_themes_content=final_context_for_llm, source_standards_list_str=source_docs_str)
        try:
            response = await self.qg_llm.ainvoke(formatted_prompt)
            llm_output = response.content.strip(); json_str = ""
            match = re.search(r"```json\s*(\{[\s\S]*?\})\s*```", llm_output, re.DOTALL | re.MULTILINE)
            if match: json_str = match.group(1)
            else:
                fb, lb = llm_output.find('{'), llm_output.rfind('}')
                if fb != -1 and lb != -1 and lb > fb: json_str = llm_output[fb : lb + 1]
                else: print(f"[QG_SERVICE ERROR / MainCat {main_category_raw_id}] LLM output not valid JSON: {llm_output}"); return None
            main_data_llm = json.loads(json_str)
            req_keys = ["main_category_name_en", "main_category_description_en", "dimension", "main_category_keywords_en"]
            if not all(k in main_data_llm for k in req_keys):
                print(f"[QG_SERVICE WARNING / MainCat {main_category_raw_id}] LLM output missing keys. Got: {main_data_llm.keys()}"); return None
            main_cat_name = main_data_llm["main_category_name_en"].strip()
            if not main_cat_name: print(f"[QG_SERVICE WARNING / MainCat {main_category_raw_id}] LLM empty 'main_category_name_en'. Skip."); return None
            main_cat_output = {
                "main_category_name_en": main_cat_name,
                "main_category_description_en": main_data_llm["main_category_description_en"],
                "dimension": str(main_data_llm["dimension"]).upper(),
                "main_category_keywords_en": main_data_llm["main_category_keywords_en"],
                "consolidated_themes": consolidated_themes_for_context, 
                "_main_category_raw_id": main_category_raw_id,
                "_constituent_entity_ids_in_mc": sorted(list(all_constituent_entity_ids_for_mc)),
                "_source_document_ids_for_mc": sorted(list(all_source_docs_for_mc)),
                "generation_method": "kg_hierarchical_main_category"
            }
            if main_cat_output["dimension"] not in ['E','S','G']: main_cat_output["dimension"] = "G"
            print(f"[QG_SERVICE INFO] LLM defined Main Category '{main_cat_output['main_category_name_en']}' from MC raw_id {main_category_raw_id}")
            return main_cat_output
        except Exception as e: print(f"[QG_SERVICE ERROR / MainCat {main_category_raw_id}] LLM call/parse error: {e}"); 
        traceback.print_exc(); 
        return None
    
    async def identify_hierarchical_themes_from_kg(
        self,
        min_first_order_community_size: int,
        min_main_category_fo_community_count: int,
        entity_graph_data: Dict[str, Any]  # <-- เพิ่ม parameter นี้
    ) -> List[Dict[str, Any]]:
        """
        Orchestrates the hierarchical theme identification process using pre-fetched graph data.
        """
        self.logger.info(f"Identifying themes with settings: min_fo_comm_size={min_first_order_community_size}, min_mc_fo_comm_count={min_main_category_fo_community_count}")
        
        # entity_graph_data = await self._get_entity_graph_data() # <--- ลบบรรทัดนี้ทิ้ง

        if not entity_graph_data or not entity_graph_data.get("nodes_map"):
            self.logger.warning("Could not use provided entity graph data. Aborting theme identification.")
            return []

        # Step 1: Detect first-order communities (sub-themes)
        first_order_communities_map = await self._detect_first_order_communities(entity_graph_data, min_community_size=min_first_order_community_size)
        if not first_order_communities_map:
            self.logger.warning("No first-order communities found with current settings. Aborting.")
            return []

        # (The rest of the function logic remains the same)
        all_consolidated_themes, all_entity_data_map = await self._generate_all_consolidated_themes(first_order_communities_map, entity_graph_data)
        if not all_consolidated_themes:
            self.logger.warning("No consolidated themes (sub-themes) generated from first-order communities.")
            return []

        meta_graph = await self._create_community_meta_graph(first_order_communities_map, entity_graph_data)
        main_category_groupings = await self._detect_main_categories_from_meta_graph(
            meta_graph, 
            min_main_category_fo_community_count=min_main_category_fo_community_count
        )

        if not main_category_groupings:
            self.logger.warning("No main category groupings from meta-graph. Structuring flat themes as main categories.")
            return [self._structure_flat_theme_as_main_category(ct, idx) for idx, ct in enumerate(all_consolidated_themes)]

        main_categories_final = await self._assemble_main_categories(main_category_groupings, all_consolidated_themes)
        
        self.logger.info(f"Identified {len(main_categories_final)} hierarchical main categories.")
        return main_categories_final

    async def _generate_all_consolidated_themes(self, first_order_communities_map, entity_graph_data):
        """Helper to generate details for all first-order communities."""
        all_consolidated_themes = []
        all_entity_data_map = entity_graph_data["nodes_map"]
        tasks = []
        for fo_comm_id, entity_ids in first_order_communities_map.items():
            context, constituent_ids, source_docs = self._prepare_context_for_consolidated_theme(fo_comm_id, entity_ids, all_entity_data_map)
            if not context.strip(): continue
            
            prompt_str = """
            You are an expert ESG analyst. A group of semantically related entities has been identified.
            Your task is to define a consolidated ESG reporting theme for this group for the industrial packaging sector.
            --- REPRESENTATIVE ENTITIES/CONCEPTS START ---
            {representative_entity_content}
            --- REPRESENTATIVE ENTITIES/CONCEPTS END ---
            Based on this information:
            1. Propose a concise **Theme Name** (in English).
            2. Provide a brief **Theme Description** (in English, 1-2 sentences).
            3. Determine the most relevant primary **ESG Dimension** ('E', 'S', or 'G').
            4. Suggest 3-5 relevant **Keywords** (in English, comma-separated).
            Output ONLY a single, valid JSON object with keys: "theme_name_en", "description_en", "dimension", "keywords_en".
            """
            prompt = PromptTemplate.from_template(prompt_str)
            formatted_prompt = prompt.format(representative_entity_content=context)
            tasks.append(self._process_first_order_community_for_consolidated_theme(formatted_prompt, constituent_ids, sorted(list(source_docs)), fo_comm_id))
            
        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for res in results:
                if isinstance(res, dict): all_consolidated_themes.append(res)
        return all_consolidated_themes, all_entity_data_map

    # --- ฟังก์ชัน Helper ใหม่ (ไม่ได้แก้ไข แต่เพิ่มเข้ามา) ---
    async def _assemble_main_categories(self, main_category_groupings, all_consolidated_themes):
        """Helper to assemble final main category structures."""
        main_categories_final = []
        tasks = []
        original_fo_comm_id_to_theme = {ct.get("_first_order_community_id_source"): ct for ct in all_consolidated_themes if ct.get("_first_order_community_id_source") is not None}

        for mc_raw_id, fo_comm_ids in main_category_groupings.items():
            consolidated_themes_for_mc = [original_fo_comm_id_to_theme.get(fo_id) for fo_id in fo_comm_ids if original_fo_comm_id_to_theme.get(fo_id)]
            if not consolidated_themes_for_mc: continue
            
            all_entities = set(eid for theme in consolidated_themes_for_mc for eid in theme.get("constituent_entity_ids", []))
            all_docs = set(doc for theme in consolidated_themes_for_mc for doc in theme.get("source_standard_document_ids", []))
            
            tasks.append(self._generate_main_category_details_with_llm(mc_raw_id, consolidated_themes_for_mc, all_docs, all_entities))
        
        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for res in results:
                if isinstance(res, dict): main_categories_final.append(res)
        
        return main_categories_final

    def _prepare_context_for_consolidated_theme(self, fo_comm_id: int, entity_ids_in_fo_comm: List[str], all_entity_data_map: Dict[str, Any]):
        # ... (Method from your latest code) ...
        context_for_llm_ct = ""
        char_count_ct = 0
        max_chars_ct_context = 1000000
        num_nodes_ct_context = 0
        MAX_NODES_CT_SAMPLE = 10 
        temp_context_parts_ct = []
        constituent_entity_ids_for_ct = []
        source_doc_names_in_ct = set()
        for entity_id in entity_ids_in_fo_comm:
            entity_data = all_entity_data_map.get(entity_id)
            if entity_data:
                constituent_entity_ids_for_ct.append(entity_id)
                node_type_display_list = entity_data.get('labels', [DEFAULT_NODE_TYPE])
                node_type_display = ", ".join(node_type_display_list) if node_type_display_list else DEFAULT_NODE_TYPE
                node_desc = entity_data.get('description')
                if not node_desc or not node_desc.strip():
                    node_desc = "No substantive description."
                doc_id = entity_data.get('source_document_doc_id')
                if doc_id and isinstance(doc_id, str) and doc_id.strip():
                    source_doc_names_in_ct.add(doc_id)
                if num_nodes_ct_context < MAX_NODES_CT_SAMPLE:
                    node_info_sample = f"ENTITY_ID: {entity_id}\nENTITY_TYPE(S): {node_type_display}\nENTITY_DESCRIPTION: {node_desc}\n\n"
                    if char_count_ct + len(node_info_sample) <= max_chars_ct_context:
                        temp_context_parts_ct.append(node_info_sample)
                        char_count_ct += len(node_info_sample)
                        num_nodes_ct_context +=1
                    else: break
                else: break
        context_for_llm_ct = "".join(temp_context_parts_ct)
        if not context_for_llm_ct.strip() and entity_ids_in_fo_comm:
            first_entity_data_ct = all_entity_data_map.get(entity_ids_in_fo_comm[0])
            if first_entity_data_ct:
                context_for_llm_ct = f"Primary Entity: {entity_ids_in_fo_comm[0]} - Description: {first_entity_data_ct.get('description', 'N/A')}"[:max_chars_ct_context]
        return context_for_llm_ct, constituent_entity_ids_for_ct, source_doc_names_in_ct
    
    def _structure_flat_theme_as_main_category(self, consolidated_theme_dict: Dict[str, Any], index: int) -> Dict[str, Any]:
        # ... (เหมือนเดิม)
        main_cat_name = consolidated_theme_dict.get("theme_name_en", f"Uncategorized Main Theme {index}")
        return {
            "main_category_name_en": main_cat_name,
            "main_category_description_en": consolidated_theme_dict.get("description_en", "N/A"),
            "dimension": consolidated_theme_dict.get("dimension", "G"),
            "main_category_keywords_en": consolidated_theme_dict.get("keywords_en", ""),
            "consolidated_themes": [consolidated_theme_dict], 
            "_main_category_raw_id": f"fallback_flat_{index}",
            "generation_method": "kg_fallback_flat_as_main",
            "_constituent_entity_ids_in_mc": consolidated_theme_dict.get("constituent_entity_ids", []),
            "_source_document_ids_for_mc": consolidated_theme_dict.get("source_standard_document_ids", [])
        }

    async def _process_llm_for_main_category_definition(
        self,
        formatted_prompt: str,
        main_category_raw_id: int,
        consolidated_themes_in_this_main_category: List[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """
        Helper method to call LLM for defining a single main category and attaching its consolidated themes.
        """
        print(f"[QG_SERVICE INFO] Sending prompt to LLM for Main Category definition (raw_id: {main_category_raw_id})")
        try:
            response = await self.qg_llm.ainvoke(formatted_prompt)
            llm_output = response.content.strip()
            json_str = ""
            match = re.search(r"```json\s*(\{[\s\S]*?\})\s*```", llm_output, re.DOTALL | re.MULTILINE)
            if match: json_str = match.group(1)
            else:
                first_brace = llm_output.find('{'); last_brace = llm_output.rfind('}')
                if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
                    json_str = llm_output[first_brace : last_brace + 1]
                else:
                    print(f"[QG_SERVICE ERROR / MainCatDef {main_category_raw_id}] LLM output not valid JSON: {llm_output}")
                    return None
            
            main_category_llm_data = json.loads(json_str)
            required_keys = ["main_category_name_en", "main_category_description_en", "dimension", "main_category_keywords_en"]
            if not all(k in main_category_llm_data for k in required_keys):
                print(f"[QG_SERVICE WARNING / MainCatDef {main_category_raw_id}] LLM output missing required keys. Got: {main_category_llm_data.keys()}")
                return None

            # สร้าง dict สำหรับ Main Category นี้
            main_category_output = {
                "main_category_name_en": main_category_llm_data["main_category_name_en"],
                "main_category_description_en": main_category_llm_data["main_category_description_en"],
                "dimension": main_category_llm_data["dimension"],
                "main_category_keywords_en": main_category_llm_data["main_category_keywords_en"],
                "consolidated_themes": consolidated_themes_in_this_main_category, # list of consolidated theme dicts
                "_main_category_raw_id": main_category_raw_id,
                "generation_method": "kg_hierarchical_community_detection"
            }
            print(f"[QG_SERVICE INFO] Successfully defined Main Category: {main_category_output['main_category_name_en']}")
            return main_category_output

        except json.JSONDecodeError as e_json:
            print(f"[QG_SERVICE ERROR / MainCatDef {main_category_raw_id}] Failed to parse JSON from LLM. Cleaned: '{json_str}'. Original: '{llm_output}'. Error: {e_json}")
            return None
        except Exception as e_llm_mc_def:
            print(f"[QG_SERVICE ERROR / MainCatDef {main_category_raw_id}] LLM call or processing error: {e_llm_mc_def}")
            traceback.print_exc()
            return None

    async def _process_first_order_community_for_consolidated_theme( # Renamed
        self, formatted_prompt: str, constituent_entity_ids: List[str], 
        source_doc_names: List[str], first_order_community_id: int
    ) -> Optional[Dict[str, Any]]:
        # ... (Method from your latest code - ensure it returns the "theme_name" for sub-theme)
        print(f"[QG_SERVICE INFO] LLM processing for Consolidated Sub-Theme from first-order community ID: {first_order_community_id}")
        try:
            response = await self.qg_llm.ainvoke(formatted_prompt)
            llm_output = response.content.strip(); json_str = ""
            match = re.search(r"```json\s*(\{[\s\S]*?\})\s*```", llm_output, re.DOTALL | re.MULTILINE)
            if match: json_str = match.group(1)
            else:
                fb, lb = llm_output.find('{'), llm_output.rfind('}')
                if fb != -1 and lb != -1 and lb > fb: json_str = llm_output[fb : lb + 1]
                else: print(f"[QG_SERVICE ERROR / FO-Com {first_order_community_id}] LLM output not valid JSON: {llm_output}"); return None
            llm_theme_data = json.loads(json_str)
            theme_name_en = llm_theme_data.get("theme_name_en")
            if not theme_name_en or not isinstance(theme_name_en, str) or not theme_name_en.strip(): 
                print(f"[QG_SERVICE WARNING / FO-Com {first_order_community_id}] LLM no 'theme_name_en'. Skip."); return None
            keywords_val = llm_theme_data.get("keywords_en", "")
            return {
                "theme_name": theme_name_en.strip(), "theme_name_en": theme_name_en.strip(),
                "theme_name_th": await self.translate_text_to_thai(theme_name_en, category_name=theme_name_en, keywords=llm_theme_data.get("keywords_en", "")),
                "description_en": llm_theme_data.get("description_en", ""),
                "dimension": str(llm_theme_data.get("dimension", "G")).upper() if str(llm_theme_data.get("dimension", "G")).upper() in ['E','S','G'] else "G",
                "keywords_en": ", ".join(str(k).strip() for k in keywords_val.split(',')) if isinstance(keywords_val, str) else (", ".join(str(k).strip() for k in keywords_val) if isinstance(keywords_val, list) else ""),
                "constituent_entity_ids": constituent_entity_ids, 
                "source_standard_document_ids": list(set(s for s in source_doc_names if s)),
                "generation_method": "kg_first_order_community_igraph",
                "_first_order_community_id_source": first_order_community_id 
            }
        except Exception as e: print(f"[QG_SERVICE ERROR / FO-Com {first_order_community_id}] LLM call/parse error: {e}"); traceback.print_exc(); return None
        
    # generate_question_for_theme is renamed to generate_question_for_consolidated_theme
    async def generate_question_for_theme_level(
        self,
        theme_info: Dict[str, Any], # This is a Main Category Info dictionary
    ) -> Optional[GeneratedQuestionSet]:

        main_category_name = theme_info.get("main_category_name_en")
        main_category_description = theme_info.get("main_category_description_en", "")
        main_category_keywords = theme_info.get("main_category_keywords_en", "")
        main_category_dimension = theme_info.get("dimension", "G") # From _generate_main_category_details_with_llm

        # Entities and source documents aggregated at the Main Category level
        constituent_entity_ids_for_mc = theme_info.get("_constituent_entity_ids_in_mc", [])
        source_documents_for_mc = theme_info.get("_source_document_ids_for_mc", []) # List of doc_ids

        if not main_category_name:
            print(f"[QG_SERVICE ERROR /gen_q_set] Critical: Main Category name is missing. Info: {str(theme_info)[:200]}")
            return None

        # --- 1A. Prepare Concise Context for Main Question (Optional Enhancement) ---
        concise_graph_context_summary = "No specific graph hints available for this category."
        concise_chunk_context_summary = "No specific document excerpts were summarized for this category." # Default


        if self.neo4j_service:
            # Concise Graph Context (existing logic from previous version - seems okay)
            if main_category_keywords or main_category_name:
                try:
                    graph_res_summary = await self.neo4j_service.get_graph_context_for_theme_chunks_v2(
                        theme_name=main_category_name, theme_keywords=main_category_keywords,
                        max_central_entities=2, max_hops_for_central_entity=2,
                        max_relations_to_collect_per_central_entity=2,
                        max_total_context_items_str_len=1000000
                    )
                    if graph_res_summary and \
                    "No central entities identified" not in graph_res_summary and \
                    "Neo4j graph not available" not in graph_res_summary and \
                    "No detailed graph context elements could be formatted" not in graph_res_summary:
                        concise_graph_context_summary = graph_res_summary
                except Exception as e_mc_graph_summary:
                    print(f"[QG_SERVICE WARNING /gen_q_set] Error getting concise graph context for MC '{main_category_name}': {e_mc_graph_summary}")

            # <<< NEW: Concise Chunk Context Summary >>>
            temp_concise_chunk_list = []
            if constituent_entity_ids_for_mc and self.neo4j_service.graph:
                concise_chunk_query = """
                UNWIND $entity_ids AS target_entity_id
                MATCH (e:__Entity__ {id: target_entity_id})
                MATCH (doc_node:Document)-[:MENTIONS]->(e) // <<< ใช้ Label Document และ Relationship MENTIONS
                WHERE doc_node.text IS NOT NULL AND trim(doc_node.text) <> "" // ตรวจสอบว่ามี text
                WITH DISTINCT doc_node // เอา node ที่ไม่ซ้ำกัน
                RETURN doc_node.text AS text,
                    doc_node.doc_id AS source_doc, // ใช้ doc_node.doc_id
                    doc_node.chunk_id AS chunk_id   // ใช้ doc_node.chunk_id
                ORDER BY size(doc_node.text) ASC
                LIMIT 20
                """ # Fetches 1 shortest chunk linked to any of the constituent entities
                try:
                    loop = asyncio.get_running_loop()
                    results_concise_chunk = await loop.run_in_executor(None,
                                                                    self.neo4j_service.graph.query,
                                                                    concise_chunk_query,
                                                                    {'entity_ids': constituent_entity_ids_for_mc[:5]}) # Use first 5 entities to find a relevant chunk
                    if results_concise_chunk and results_concise_chunk[0]:
                        record = results_concise_chunk[0]
                        text = record.get('text', '')[:1000000] # Get first 300 chars as preview
                        doc_id = record.get('source_doc', 'Unknown Document')
                        temp_concise_chunk_list.append(f"Key excerpt from '{doc_id}': \"{text}...\"")
                except Exception as e_concise_direct:
                    print(f"[QG_SERVICE WARNING /gen_q_set] Error fetching concise direct chunk for MC '{main_category_name}': {e_concise_direct}")
            
            # Fallback to similarity if no direct chunk or if desired
            if not temp_concise_chunk_list and self.neo4j_service.store and (main_category_name or main_category_keywords):
                try:
                    sim_query = f"{main_category_name} {main_category_keywords}"
                    similar_docs = await self.neo4j_service.get_relate(sim_query, k=1)
                    if similar_docs and similar_docs[0]:
                        text = similar_docs[0].page_content[:1000000]
                        doc_id = similar_docs[0].metadata.get('doc_id', 'Unknown Document')
                        temp_concise_chunk_list.append(f"Relevant excerpt (similar) from '{doc_id}': \"{text}...\"")
                except Exception as e_concise_sim:
                    print(f"[QG_SERVICE WARNING /gen_q_set] Error fetching concise similarity chunk for MC '{main_category_name}': {e_concise_sim}")

            if temp_concise_chunk_list:
                concise_chunk_context_summary = " ".join(temp_concise_chunk_list)

            main_q_prompt_template_str = """
            You are a meticulous ESG strategist reviewing a factory in the **industrial packaging sector**.
            For the Main ESG Category named "{main_category_name}" (Description: {main_category_description}):

            For your awareness, here are some related concepts found in the factory's data:
            Knowledge Graph Hints: {concise_graph_context_summary}
            Relevant Document Excerpt Hint: {concise_chunk_context_summary}

            Based on the Main Category's identity, formulate ONE single, high-level, strategic question.
            This question must:
            - Act as a true umbrella for the entire category.
            - Assess the company's overall strategy, policy, or governance structure for this topic.
            - AVOID asking for specific data, numbers, or lists. It should probe for strategic commitment.
            - Example of a GOOD strategic question: "What is the board's oversight mechanism for managing climate-related risks and opportunities?"
            - Example of a BAD, non-strategic question: "What were the total GHG emissions last year?"

            Output ONLY the single Main Question text in English.
            """
        main_q_prompt = PromptTemplate.from_template(main_q_prompt_template_str)
        formatted_main_q_prompt = main_q_prompt.format(
            main_category_name=main_category_name,
            main_category_description=main_category_description if main_category_description else "Not specified.",
            main_category_keywords=main_category_keywords if main_category_keywords else "Not specified.",
            concise_graph_context_summary=concise_graph_context_summary,
            concise_chunk_context_summary=concise_chunk_context_summary # Now populated
        )
        main_question_text_en = f"What is the company's overall strategic approach and commitment to {main_category_name}?" # Default fallback
        try:
            # print(f"[QG_SERVICE DEBUG /gen_q_set] MainQ Prompt for {main_category_name}:\n{formatted_main_q_prompt}")
            response_main_q = await self.qg_llm.ainvoke(formatted_main_q_prompt)
            generated_main_q_text = response_main_q.content.strip()
            if generated_main_q_text:
                main_question_text_en = generated_main_q_text
            else:
                print(f"[QG_SERVICE WARNING /gen_q_set] LLM did not generate Main Question text for '{main_category_name}'. Using fallback.")
        except Exception as e_main_q:
            print(f"[QG_SERVICE ERROR /gen_q_set] Main Question LLM call failed for '{main_category_name}': {e_main_q}. Using fallback.")

        # --- 2. Prepare Super Context from Consolidated Sub-Themes (for sub-questions) ---
        super_context_from_sub_themes = ""
        consolidated_sub_themes_in_mc = theme_info.get("consolidated_themes", [])
        if consolidated_sub_themes_in_mc:
            temp_super_context_parts = []
            char_count_super_ctx = 0
            MAX_CHARS_SUPER_CTX = 10000 # Limit for sub-theme details in sub-question prompt
            for sub_theme_data_for_ctx in consolidated_sub_themes_in_mc[:7]: # Sample up to 7
                s_name = sub_theme_data_for_ctx.get("theme_name_en", "N/A")
                s_desc = sub_theme_data_for_ctx.get("description_en", "N/A")
                s_keywords = sub_theme_data_for_ctx.get("keywords_en", "N/A")
                s_dim = sub_theme_data_for_ctx.get("dimension", "N/A")
                entry = f"Sub-Theme: {s_name}\n  Description: {s_desc}\n  Keywords: {s_keywords}\n  Dimension: {s_dim}\n"
                if char_count_super_ctx + len(entry) <= MAX_CHARS_SUPER_CTX:
                    temp_super_context_parts.append(entry)
                    char_count_super_ctx += len(entry)
                else:
                    break
            super_context_from_sub_themes = "\n---\n".join(temp_super_context_parts)

        if not super_context_from_sub_themes.strip() and consolidated_sub_themes_in_mc:
            super_context_from_sub_themes = f"This main category generally covers topics such as: {', '.join([st.get('theme_name_en', 'Unnamed Sub-Theme') for st in consolidated_sub_themes_in_mc[:3]])}."
        elif not consolidated_sub_themes_in_mc:
            super_context_from_sub_themes = "No specific sub-themes were detailed for this main category. Focus on its general description and keywords."

        # --- 3. Fetch Detailed Chunk & Graph Context for Sub-Questions ---
        final_chunk_context_str = "No relevant standard document excerpts could be retrieved for detailed context."
        valuable_chunks_count = 0
        MAX_CHUNKS_FOR_SUBQ_PROMPT = 3
        MAX_CHUNK_CHARS_TOTAL_SUBQ = 80000
        MIN_VALUABLE_CHUNK_LENGTH_SUBQ = 50
        current_chunk_context_chars = 0 # Initialize character count for chunks
        chunk_texts_list_for_prompt = [] # Initialize list for storing formatted chunk strings

        # 3A. Fetch Chunks via Direct Entity Query
        if constituent_entity_ids_for_mc and self.neo4j_service and self.neo4j_service.graph:
            chunk_query_cypher = """
            UNWIND $entity_ids AS target_entity_id
            MATCH (e:__Entity__ {id: target_entity_id})
            MATCH (doc_node:Document)-[:MENTIONS]->(e) // <<< ใช้ Label Document และ Relationship MENTIONS
            WHERE doc_node.text IS NOT NULL AND trim(doc_node.text) <> ""
            WITH DISTINCT doc_node
            RETURN doc_node.text AS text,
                doc_node.doc_id AS source_doc,
                doc_node.chunk_id AS chunk_id,
                doc_node.page_number AS page_number, // ถ้า doc_node มี page_number
                // ปรับส่วน ORDER BY ถ้า doc_node มี property สำหรับเรียงลำดับ
                // เช่น COALESCE(doc_node.sequence_in_document, doc_node.page_number * 10000, 999999) AS order_prop
                // หรือเอา ORDER BY ที่ซับซ้อนออกไปก่อนถ้าไม่แน่ใจ
                doc_node.page_number AS order_prop // สมมติว่าใช้ page_number เรียง
            ORDER BY order_prop ASC
            LIMIT 20
            """
            try:
                loop = asyncio.get_running_loop()
                chunk_results = await loop.run_in_executor(None,
                                                        self.neo4j_service.graph.query,
                                                        chunk_query_cypher,
                                                        {'entity_ids': constituent_entity_ids_for_mc})
                if chunk_results:
                    print(f"[QG_SERVICE INFO /gen_q_set] Fetched {len(chunk_results)} raw chunks via direct entity query for MC '{main_category_name}'.")
                    for res in chunk_results:
                        text_val = res.get('text', '').strip()
                        if text_val and len(text_val) >= MIN_VALUABLE_CHUNK_LENGTH_SUBQ:
                            source_doc_val = res.get('source_doc', 'Unknown Document')
                            chunk_id_val = res.get('chunk_id', 'Unknown Chunk')
                            page_num_val = res.get('page_number', 'N/A')
                            chunk_entry_str = f"Excerpt from Document '{source_doc_val}' (Page: {page_num_val}, Chunk: {chunk_id_val}):\n{text_val}\n---\n"
                            if current_chunk_context_chars + len(chunk_entry_str) <= MAX_CHUNK_CHARS_TOTAL_SUBQ and \
                            valuable_chunks_count < MAX_CHUNKS_FOR_SUBQ_PROMPT:
                                chunk_texts_list_for_prompt.append(chunk_entry_str)
                                current_chunk_context_chars += len(chunk_entry_str)
                                valuable_chunks_count += 1
                            else: break
                if not chunk_texts_list_for_prompt:
                    print(f"[QG_SERVICE INFO /gen_q_set] No valuable chunks selected from direct query for MC '{main_category_name}'.")
            except Exception as e_direct_chunk:
                print(f"[QG_SERVICE ERROR /gen_q_set] Error fetching/processing direct chunks for MC '{main_category_name}': {e_direct_chunk}")

        # 3B. Fetch Chunks via Similarity Search (Fallback/Enhancement)
        if valuable_chunks_count < MAX_CHUNKS_FOR_SUBQ_PROMPT and self.neo4j_service and self.neo4j_service.store:
            search_keywords_for_similarity = f"{main_category_name} {main_category_keywords}"
            needed_chunks_similarity = MAX_CHUNKS_FOR_SUBQ_PROMPT - valuable_chunks_count
            if needed_chunks_similarity > 0:
                try:
                    print(f"[QG_SERVICE INFO /gen_q_set] Attempting similarity search for {needed_chunks_similarity} chunks for MC '{main_category_name}'.")
                    similar_chunks_docs = await self.neo4j_service.get_relate(search_keywords_for_similarity, k=needed_chunks_similarity)
                    if similar_chunks_docs:
                        for doc_sim in similar_chunks_docs:
                            text_val_sim = doc_sim.page_content.strip()
                            if text_val_sim and len(text_val_sim) >= MIN_VALUABLE_CHUNK_LENGTH_SUBQ:
                                source_doc_val_sim = doc_sim.metadata.get('doc_id', doc_sim.metadata.get('source_doc', 'Unknown'))
                                chunk_id_val_sim = doc_sim.metadata.get('chunk_id', 'Unknown')
                                page_num_val_sim = doc_sim.metadata.get('page_number', 'N/A')
                                chunk_entry_sim_str = f"Excerpt (Similar) from Document '{source_doc_val_sim}' (Page: {page_num_val_sim}, Chunk: {chunk_id_val_sim}):\n{text_val_sim}\n---\n"
                                if current_chunk_context_chars + len(chunk_entry_sim_str) <= MAX_CHUNK_CHARS_TOTAL_SUBQ:
                                    chunk_texts_list_for_prompt.append(chunk_entry_sim_str) # Append to the same list
                                    current_chunk_context_chars += len(chunk_entry_sim_str)
                                    valuable_chunks_count +=1
                                else: break
                                if valuable_chunks_count >= MAX_CHUNKS_FOR_SUBQ_PROMPT: break
                        print(f"[QG_SERVICE INFO /gen_q_set] Added up to {len(similar_chunks_docs)} chunks via similarity for MC '{main_category_name}'. Total valuable: {valuable_chunks_count}")
                except Exception as e_sim_chunk_mc:
                    print(f"[QG_SERVICE ERROR /gen_q_set] Error fetching/processing similarity chunks for MC '{main_category_name}': {e_sim_chunk_mc}")

        if chunk_texts_list_for_prompt:
            final_chunk_context_str = "".join(chunk_texts_list_for_prompt) # Join directly
            if len(final_chunk_context_str) > MAX_CHUNK_CHARS_TOTAL_SUBQ: # Final truncation if somehow still over
                final_chunk_context_str = final_chunk_context_str[:MAX_CHUNK_CHARS_TOTAL_SUBQ] + "\n... [CHUNK CONTEXT TRUNCATED DUE TO OVERALL LENGTH]"
        else: # If list is still empty after all attempts
            final_chunk_context_str = "No specific content chunks could be retrieved for this Main Category. Base sub-questions on the main category's general information."

        # 3C. Fetch Graph Context
        final_graph_context_str = "No specific knowledge graph context available for this Main Category."
        if self.neo4j_service and (main_category_keywords or main_category_name):
            try:
                graph_res = await self.neo4j_service.get_graph_context_for_theme_chunks_v2(
                    theme_name=main_category_name, theme_keywords=main_category_keywords,
                    max_central_entities=3, max_hops_for_central_entity=3,
                    max_relations_to_collect_per_central_entity=5,
                    max_total_context_items_str_len=900000
                )
                if graph_res and \
                "No central entities identified" not in graph_res and \
                "Neo4j graph not available" not in graph_res and \
                "No detailed graph context elements could be formatted" not in graph_res:
                    final_graph_context_str = graph_res
            except Exception as e_mc_graph:
                print(f"[QG_SERVICE WARNING /gen_q_set] Error getting graph context for MC '{main_category_name}': {e_mc_graph}")

        # --- 4. Generate Rolled-up Sub-Questions for the Main Category ---
        sub_q_prompt_template_str = """
        You are an expert ESG consultant for an **industrial packaging factory**.
        The Main ESG Category is: "{main_category_name}"
        The Main Question for this category is: "{main_question_text}"

        This Main Category encompasses the following specific aspects or sub-themes:
        --- OVERVIEW OF SUB-THEME DETAILS START ---
        {super_context_from_sub_themes}
        --- OVERVIEW OF SUB-THEME DETAILS END ---

        --- Supporting Context from Knowledge Graph for "{main_category_name}": START---
        {graph_context}
        --- Supporting Context from Knowledge Graph END ---

        --- Supporting Context from Standard Documents for "{main_category_name}": ---
        {chunk_context}
        --- Supporting Context from Standard Documents END ---

        Task: Based on the Main Question and ALL provided context (sub-theme overview, knowledge graph, and standard documents), formulate a set of 1-3 concise and highly critical Sub-Questions.
        These Sub-Questions should:
        1. Directly help answer or provide detailed supporting information for the Main Question.
        2. Explore the MOST CRITICAL and REPRESENTATIVE aspects covered by the "Overview of Sub-Theme Details" and other contexts. Do NOT try to create questions for every single detail if it makes the list too long or redundant.
        3. Elicit a mix of:
            a. **Policies & Commitments:** (e.g., "What is the company's formal policy on...?", "Are there publicly stated commitments regarding...?")
            b. **Strategies & Processes:** (e.g., "Describe the strategy/process for managing...", "What are the key steps in the company's approach to...?")
            c. **Performance & Metrics:** (e.g., "What are the key performance indicators (KPIs) used to track progress on...?", "What was the [specific metric] for the last reporting period?", "What are the company's targets for [metric]?")
            d. **Governance & Oversight:** (e.g., "Who is responsible for overseeing...?", "How frequently is performance on this topic reviewed by management/board?")
        4. Be relevant to an industrial packaging factory.
        5. If evident from context (especially Standard Documents or specific KG entities), specify source (Source: Document Name - Section/Page OR Source: KG - Entity Name).
        6.  **VERY IMPORTANT: Each sub-question MUST be a single, distinct question. DO NOT combine multiple questions into one sentence using "and" or by listing different topics.**

        Output ONLY a single, valid JSON object with these exact keys:
        - "rolled_up_sub_questions_text_en": A string containing ONLY 1-3 sub-questions, each numbered and on a new line.
        - "detailed_source_info_for_subquestions": A brief textual summary of how the contexts were used to formulate the sub-questions, or specific sources if attributable.
        """
        sub_q_prompt = PromptTemplate.from_template(sub_q_prompt_template_str)
        formatted_sub_q_prompt = sub_q_prompt.format(
            main_category_name=main_category_name,
            main_question_text=main_question_text_en,
            super_context_from_sub_themes=super_context_from_sub_themes,
            graph_context=final_graph_context_str,
            chunk_context=final_chunk_context_str
        )

        rolled_up_sub_questions_text_en = "No specific sub-questions were generated for this main category."
        detailed_source_info_sub_q = "Sub-questions are based on the general understanding of the main category; specific source attribution from context was not directly feasible for all generated sub-questions."

        try:
            # print(f"[QG_SERVICE DEBUG /gen_q_set] SubQ_Rollup Prompt for {main_category_name} (first 3000 chars):\n{formatted_sub_q_prompt[:3000]}...")
            response_sub_q = await self.qg_llm.ainvoke(formatted_sub_q_prompt)
            llm_sub_q_output = response_sub_q.content.strip()
            json_sub_q_str = ""
            match_sub_q = re.search(r"```json\s*(\{[\s\S]*?\})\s*```", llm_sub_q_output, re.DOTALL | re.MULTILINE)
            if match_sub_q:
                json_sub_q_str = match_sub_q.group(1)
            else:
                fb_sq, lb_sq = llm_sub_q_output.find('{'), llm_sub_q_output.rfind('}')
                if fb_sq != -1 and lb_sq != -1 and lb_sq > fb_sq:
                    json_sub_q_str = llm_sub_q_output[fb_sq : lb_sq + 1]
                else:
                    print(f"[QG_SERVICE ERROR /gen_q_set] LLM output for SubQ of '{main_category_name}' was not valid JSON: {llm_sub_q_output}")

            if json_sub_q_str:
                sub_q_llm_data = json.loads(json_sub_q_str)
                generated_sq_text = sub_q_llm_data.get("rolled_up_sub_questions_text_en")
                if generated_sq_text and generated_sq_text.strip() and "No specific sub-questions were generated" not in generated_sq_text : # Check if not default
                    rolled_up_sub_questions_text_en = generated_sq_text
                else:
                    print(f"[QG_SERVICE WARNING /gen_q_set] LLM did not provide meaningful sub-questions for '{main_category_name}'. Using default.")


                generated_source_info = sub_q_llm_data.get("detailed_source_info_for_subquestions")
                if generated_source_info and generated_source_info.strip():
                    detailed_source_info_sub_q = generated_source_info
            else:
                print(f"[QG_SERVICE WARNING /gen_q_set] No JSON object could be extracted from LLM SubQ output for '{main_category_name}'. Using defaults for sub-questions.")

        except json.JSONDecodeError as e_json_sub:
            print(f"[QG_SERVICE ERROR /gen_q_set] JSONDecodeError for SubQ of '{main_category_name}': {e_json_sub}. JSON string was: '{json_sub_q_str}'")
        except Exception as e_sub_q:
            print(f"[QG_SERVICE ERROR /gen_q_set] Sub-Question LLM call/parse error for '{main_category_name}': {e_sub_q}")

        return GeneratedQuestionSet(
            main_question_text_en=main_question_text_en,
            rolled_up_sub_questions_text_en=rolled_up_sub_questions_text_en,
            main_category_name=main_category_name,
            main_category_dimension=main_category_dimension,
            main_category_keywords=main_category_keywords,
            main_category_description=main_category_description,
            main_category_constituent_entities=constituent_entity_ids_for_mc, # These are __Entity__ IDs
            main_category_source_docs=source_documents_for_mc, # These are doc_ids like "files_0_GRI..."
            detailed_source_info_for_subquestions=detailed_source_info_sub_q
        )

    async def are_questions_substantially_similar(self, text1: str, text2: str, threshold: float = 0.90) -> bool: # Increased threshold slightly
        # ... (existing implementation is good, ensure self.similarity_llm_embedding is used) ...
        if not text1 or not text2: return False
        text1_lower = text1.strip().lower()
        text2_lower = text2.strip().lower()
        if text1_lower == text2_lower: return True
        
        if not self.similarity_llm_embedding:
            print("[QG_SERVICE WARNING] Similarity model N/A. Basic string compare for similarity.")
            return text1_lower == text2_lower
        
        try:
            loop = asyncio.get_running_loop()
            # embed_documents is synchronous
            embeddings = await loop.run_in_executor(None, self.similarity_llm_embedding.embed_documents, [text1_lower, text2_lower])
            if len(embeddings) < 2 or not embeddings[0] or not embeddings[1]:
                print("[QG_SERVICE ERROR] Failed to generate embeddings for similarity check.")
                return False # Or raise error
            
            embedding1 = np.array(embeddings[0]).reshape(1, -1)
            embedding2 = np.array(embeddings[1]).reshape(1, -1)
            sim_score = cosine_similarity(embedding1, embedding2)[0][0]
            is_similar = sim_score >= threshold
            # print(f"[QG_SERVICE SIMILARITY DEBUG] Score for '{text1[:30]}...' vs '{text2[:30]}...': {sim_score:.4f} (Th: {threshold}, Similar: {is_similar})")
            return is_similar
        except Exception as e:
            print(f"[QG_SERVICE ERROR] Similarity check error: {e}. Defaulting to False (not similar).")
            traceback.print_exc()
            return False
    
            # --- ฟังก์ชันนี้คือเวอร์ชันสมบูรณ์ที่เติมโค้ดส่วนที่ขาดไปแล้ว ---
    async def evolve_and_store_questions(self, document_ids: List[str], is_baseline_upload: bool):
        """
        Main orchestrator that runs the pipeline and returns a JSON-serializable comparison result.
        """
        self.logger.info(f"Evolve and store called with baseline_upload = {is_baseline_upload}")

        # Helper function to make data JSON serializable
        def serialize_docs(docs_raw: List[Dict]) -> List[Dict]:
            serialized = []
            for doc in docs_raw:
                # Convert ObjectId to string
                if '_id' in doc:
                    doc['_id'] = str(doc['_id'])
                # Convert datetime to string (ISO format)
                for key, value in doc.items():
                    if isinstance(value, datetime):
                        doc[key] = value.isoformat()
                serialized.append(doc)
            return serialized

        # 1. Get state BEFORE processing
        before_state_raw = await self.mongodb_service.get_all_active_questions_raw()
        before_state_serializable = serialize_docs(before_state_raw)
        before_map = {q["theme"]: q for q in before_state_serializable}

        # 2. Decide which pipeline to run
        if is_baseline_upload:
            self.logger.info("--- Baseline mode selected by user. ---")
            await self.mongodb_service.clear_all_questions()
            await self._run_baseline_generation()
        else:
            # If not baseline, ensure we're not running on empty docs
            if not document_ids:
                self.logger.warning("Update mode called with no document IDs. Aborting evolution.")
                # Return the "before" state as "unchanged"
                return [{"question": q, "status": "unchanged"} for q in before_state_serializable]
                
            self.logger.info(f"--- Update mode selected. Running for docs: {document_ids} ---")
            await self._run_update_generation(document_ids)

        # 3. Get state AFTER processing
        all_q_raw = await self.mongodb_service.get_all_questions_raw()
        after_state_serializable = serialize_docs(all_q_raw)
        after_state_serializable.sort(key=lambda x: (x.get("theme", ""), x.get("version", 0)))

        # 4. Compare and create the result payload
        display_list = []
        processed_themes = set()

        # Iterate through the "after" state to find new and updated items
        for q_after in after_state_serializable:
            q_theme = q_after.get("theme")
            processed_themes.add(q_theme)
            
            status = "unchanged"
            q_before = before_map.get(q_theme)
            
            if not q_before:
                status = "new"
            elif q_after.get("version", 1) > q_before.get("version", 1):
                status = "updated"
            
            if not q_after.get("is_active", False):
                # This logic is complex, let's simplify: only show if it was active before
                if q_before and q_before.get("is_active", False):
                    status = "deactivated"
                else:
                    continue # Skip if it was already inactive or never existed

            display_list.append({"question": q_after, "status": status})
            
        # Iterate through the "before" state to find items that are now missing (if any)
        # This logic is implicitly handled by the "deactivated" status above.
        
        # If no changes were detected, just return the list of unchanged items
        if not any(item['status'] != 'unchanged' for item in display_list):
            return [{"question": q, "status": "unchanged"} for q in after_state_serializable]

        return display_list

    # ===================================================================
    # MODE 1: BASELINE GENERATION LOGIC (เหมือนเดิม แต่ปรับการบันทึกเล็กน้อย)
    # ===================================================================
    async def _run_baseline_generation(self):
        """
        Performs the full 4-phase question generation for the initial setup.
        """
        self.logger.info("Starting full 4-Phase question generation for baseline.")
        
        all_set_benchmarks_from_db = self.mongodb_service.get_set_benchmark_questions()
        if not all_set_benchmarks_from_db:
            self.logger.error("BASELINE RUN: Could not load SET benchmark questions. Aborting.")
            return

        used_entity_ids, used_chunk_ids = set(), set()

        # Phase 1: SET-Driven
        set_driven_questions, used_entity_ids_p1, used_chunk_ids_p1 = await self._run_set_driven_phase(all_set_benchmarks_from_db)
        used_entity_ids.update(used_entity_ids_p1)
        used_chunk_ids.update(used_chunk_ids_p1)

        # Phase 2: Targeted GRI
        targeted_questions, used_entity_ids_p2, used_chunk_ids_p2 = await self._run_targeted_gri_phase(TARGET_GRI_STANDARDS, used_entity_ids, used_chunk_ids)
        used_entity_ids.update(used_entity_ids_p2)
        used_chunk_ids.update(used_chunk_ids_p2)
        
        # Phase 3: Organic Discovery
        organic_questions = await self._run_organic_discovery_phase(used_entity_ids, used_chunk_ids)

        # Phase 4A: Final Validation
        candidates_for_validation = targeted_questions + organic_questions
        final_approved_questions = await self._run_final_validation_phase(candidates_for_validation, set_driven_questions)

        # Phase 4B: Integration (Baseline Mode)
        all_questions_to_integrate = set_driven_questions + final_approved_questions
        self.logger.info(f"BASELINE (Integration): Saving {len(all_questions_to_integrate)} question sets.")
        
        for candidate_dict in all_questions_to_integrate:
            candidate_dict['version'] = 1
            await self.mongodb_service.store_esg_question(ESGQuestion(**candidate_dict))
        
        self.logger.info("Baseline generation process completed successfully.")

    async def _run_organic_discovery_phase(self, used_entity_ids: Set[str], used_chunk_ids: Set[str]) -> List[Dict]:
        self.logger.info(f"PHASE 3: Fetching residual graph, excluding {len(used_entity_ids)} entities.")
        
        residual_graph_data = await self._get_entity_graph_data(exclude_entity_ids=used_entity_ids)
        if not residual_graph_data or not residual_graph_data.get("nodes_map"):
            self.logger.warning("PHASE 3: Residual graph is empty. No organic themes to discover.")
            return []

        theme_structures = await self._run_adaptive_theme_identification(entity_graph_data=residual_graph_data)
        if not theme_structures: return []
            
        q_sets, info_map = await self._generate_qsets_from_theme_structures(theme_structures)
        
        organic_questions = []
        for q_set in q_sets:
            q_dict = await self._convert_gqs_to_final_dict(q_set, info_map[q_set.main_category_name])
            organic_questions.append(q_dict)
            
        return organic_questions

    async def _run_final_validation_phase(self, candidate_questions: List[Dict], set_driven_questions: List[Dict]) -> List[Dict]:
        if not candidate_questions: return []
        
        self.logger.info(f"PHASE 4 (Validation): Checking {len(candidate_questions)} candidates for redundancy against SET-driven questions.")
        approved_questions = []
        for candidate in candidate_questions:
            is_redundant = await self._is_theme_redundant(candidate, set_driven_questions)
            if not is_redundant:
                approved_questions.append(candidate)
            else:
                self.logger.warning(f"Candidate '{candidate.get('theme')}' REJECTED as redundant to a SET-driven question.")
        return approved_questions

    def _process_node_results(self, nodes_result) -> Dict:
        """Processes the raw node query result into a nodes_map."""
        nodes_map = {}
        if not nodes_result:
            return nodes_map
        for record in nodes_result:
            node_id = record.get('id')
            if node_id:
                nodes_map[node_id] = {
                    'id': node_id,
                    'description': record.get('description', ''),
                    'labels': record.get('specific_labels', [DEFAULT_NODE_TYPE]),
                    'sources': record.get('sources', []) # List of {doc_id, standard_code} dicts
                }
        return nodes_map

    def _process_edge_results(self, edges_result, nodes_map: Dict) -> List:
        """Processes the raw edge query result into a list of tuples."""
        edges = []
        if not edges_result:
            return edges
        for record in edges_result:
            source, target = record.get('source'), record.get('target')
            # Ensure edges only connect nodes that are actually in our map
            if source and target and source in nodes_map and target in nodes_map:
                edges.append(tuple(sorted((source, target))))
        return list(set(edges))

    async def embed_texts(self, texts: List[str]) -> List[Optional[np.ndarray]]:
        """A helper to embed a list of texts and return a list of 1D numpy arrays."""
        if not self.similarity_llm_embedding or not texts:
            return [None] * len(texts)
        try:
            embeddings = await self.run_in_executor(self.similarity_llm_embedding.embed_documents, texts)
            # คืนค่าเป็น list ของ 1D arrays โดยตรง
            return [np.array(emb) for emb in embeddings]
        except Exception as e:
            self.logger.error(f"Failed to generate embeddings: {e}", exc_info=True)
            return [None] * len(texts)

    async def run_in_executor(self, func, *args):
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, func, *args)

    async def _run_set_driven_phase(self, all_set_benchmarks: List[Dict]) -> Tuple[List[Dict], Set[str], Set[str]]:
        self.logger.info("--- Starting Phase 1: SET-Driven Generation (2-Round Strategy) ---")
        final_questions, used_entities, used_chunks = [], set(), set()
        
        self.logger.info("PHASE 1 (Round 1): Starting high-precision search.")
        tasks_r1 = [self._find_evidence_and_generate_for_set_q(set_q) for set_q in all_set_benchmarks]
        results_r1 = await asyncio.gather(*tasks_r1)

        covered_set_ids = set()
        for res in results_r1:
            if res and res.get("question"):
                final_questions.append(res["question"])
                used_entities.update(res.get("used_entity_ids", set()))
                used_chunks.update(res.get("used_chunk_ids", set()))
                if res["question"].get("related_set_questions"):
                    covered_set_ids.add(res["question"]["related_set_questions"][0].set_id)
        
        self.logger.info(f"PHASE 1 (Round 1) Complete: Generated {len(final_questions)} questions, covering {len(covered_set_ids)} SET IDs.")

        uncovered_set_qs = [q for q in all_set_benchmarks if q['id'] not in covered_set_ids]
        if not uncovered_set_qs:
            self.logger.info("PHASE 1 (Round 2): All SET questions covered in Round 1. Skipping.")
            return final_questions, used_entities, used_chunks

        self.logger.info(f"PHASE 1 (Round 2): Starting broader search for {len(uncovered_set_qs)} uncovered SET questions.")
        tasks_r2 = []
        for set_q in uncovered_set_qs:
            broad_query = set_q.get('question_text_en', '')
            tasks_r2.append(self._find_evidence_and_generate_for_set_q(set_q, custom_query=broad_query))

        results_r2 = await asyncio.gather(*tasks_r2)
        
        round2_question_count = 0
        for res in results_r2:
            if res and res.get("question"):
                set_id = res["question"]["related_set_questions"][0].set_id
                if set_id not in covered_set_ids:
                    final_questions.append(res["question"])
                    used_entities.update(res.get("used_entity_ids", set()))
                    used_chunks.update(res.get("used_chunk_ids", set()))
                    covered_set_ids.add(set_id)
                    round2_question_count += 1

        self.logger.info(f"PHASE 1 (Round 2) Complete: Generated {round2_question_count} additional questions.")
        return final_questions, used_entities, used_chunks

    async def _run_update_generation(self, document_ids: List[str]):
        """
        Generates questions only from new documents and compares them to existing ones.
        """
        self.logger.info("UPDATE RUN (Step 1): Generating candidate questions from new documents.")
        
        subgraph_data = await self._get_entity_graph_data(doc_ids=document_ids)
        
        if not subgraph_data:
            self.logger.warning(f"UPDATE RUN: No graph data found for new documents: {document_ids}. Aborting.")
            return

        candidate_themes = await self.identify_hierarchical_themes_from_kg(entity_graph_data=subgraph_data, min_first_order_community_size=3, min_main_category_fo_community_count=2)
        if not candidate_themes:
            self.logger.warning(f"UPDATE RUN: Could not generate any candidate themes from new documents. Aborting.")
            return

        candidate_qsets, info_map = await self._generate_qsets_from_theme_structures(candidate_themes)
        self.logger.info(f"UPDATE RUN (Step 1) Complete: Generated {len(candidate_qsets)} candidate questions.")

        self.logger.info("UPDATE RUN (Step 2): Evaluating candidates against existing questions in DB.")
        active_db_questions = await self.mongodb_service.get_all_active_questions()
        
        for candidate_qset in candidate_qsets:
            await self._evaluate_and_integrate_update(candidate_qset, info_map.get(candidate_qset.main_category_name, {}), active_db_questions)
            
        self.logger.info("Update generation process completed successfully.")

    async def _run_baseline_generation(self):
        """
        Performs the full 4-phase question generation for the initial setup.
        """
        self.logger.info("Starting full 4-Phase question generation for baseline.")
        
        all_set_benchmarks_from_db = self.mongodb_service.get_set_benchmark_questions()
        if not all_set_benchmarks_from_db:
            self.logger.error("BASELINE RUN: Could not load SET benchmark questions. Aborting.")
            return

        used_entity_ids, used_chunk_ids = set(), set()

        # Phase 1: SET-Driven
        set_driven_questions, used_entity_ids_p1, used_chunk_ids_p1 = await self._run_set_driven_phase(all_set_benchmarks_from_db)
        used_entity_ids.update(used_entity_ids_p1)
        used_chunk_ids.update(used_chunk_ids_p1)
        self.logger.info(f"BASELINE (Phase 1) Complete: Generated {len(set_driven_questions)} SET-driven questions.")

        # Phase 2: Targeted GRI
        targeted_questions, used_entity_ids_p2, used_chunk_ids_p2 = await self._run_targeted_gri_phase(TARGET_GRI_STANDARDS, used_entity_ids, used_chunk_ids)
        used_entity_ids.update(used_entity_ids_p2)
        used_chunk_ids.update(used_chunk_ids_p2)
        self.logger.info(f"BASELINE (Phase 2) Complete: Generated {len(targeted_questions)} targeted GRI questions.")
        
        # Phase 3: Organic Discovery
        organic_questions = await self._run_organic_discovery_phase(used_entity_ids, used_chunk_ids)
        self.logger.info(f"BASELINE (Phase 3) Complete: Generated {len(organic_questions)} organic questions.")

        # Phase 4A: Final Validation
        candidates_for_validation = targeted_questions + organic_questions
        final_approved_questions = await self._run_final_validation_phase(candidates_for_validation, all_set_benchmarks_from_db)
        self.logger.info(f"BASELINE (Phase 4A) Complete: {len(final_approved_questions)} questions passed final validation.")

        # Phase 4B: Integration (Baseline Mode)
        all_questions_to_integrate = set_driven_questions + final_approved_questions
        self.logger.info(f"BASELINE (Phase 4B): Integrating a total of {len(all_questions_to_integrate)} question sets.")
        
        for candidate_dict in all_questions_to_integrate:
            # For baseline, only add if the theme does not exist at all.
            theme_name = candidate_dict.get('theme')
            existing = await ESGQuestion.find_one({"theme": theme_name})
            if not existing:
                self.logger.info(f"[BASELINE] Adding new question for theme: {theme_name}")
                # Ensure related_set_questions are converted to dicts if they are Pydantic models
                if 'related_set_questions' in candidate_dict:
                    candidate_dict['related_set_questions'] = [q.model_dump() for q in candidate_dict['related_set_questions']]
                if 'sub_questions_sets' in candidate_dict:
                    candidate_dict['sub_questions_sets'] = [sq.model_dump() for sq in candidate_dict['sub_questions_sets']]
                await ESGQuestion(**candidate_dict).insert()
            else:
                self.logger.info(f"[BASELINE] Theme '{theme_name}' already exists. Skipping.")
        
        self.logger.info("Baseline generation process completed successfully.")

    async def _evaluate_and_integrate_update(self, 
                                           candidate_qset: GeneratedQuestionSet,
                                           candidate_theme_info: Dict, 
                                           active_db_questions: List[ESGQuestion]):
        """
        Compares a single candidate against all active DB questions and decides
        whether to ADD, REPLACE, or DISCARD.
        """
        # Find the most similar existing question in the database
        most_similar_db_q = await self._find_most_similar_db_question(candidate_qset, active_db_questions)
        
        # Case 1: The candidate is completely new
        if not most_similar_db_q:
            self.logger.info(f"UPDATE: Candidate '{candidate_qset.main_category_name}' is novel. Adding as new question.")
            new_question_dict = await self._convert_gqs_to_final_dict(candidate_qset, candidate_theme_info)
            await self.mongodb_service.store_esg_question(ESGQuestion(**new_question_dict))
            return

        # Case 2: The candidate is similar to an existing question; an LLM must decide which is better.
        self.logger.info(f"UPDATE: Candidate '{candidate_qset.main_category_name}' is similar to existing question '{most_similar_db_q.theme}'. Evaluating quality...")
        is_better, is_same_topic = await self._is_new_theme_better(most_similar_db_q, candidate_qset)

        if not is_same_topic:
            self.logger.info(f"UPDATE: LLM judged topics are different. Adding '{candidate_qset.main_category_name}' as new question.")
            new_question_dict = await self._convert_gqs_to_final_dict(candidate_qset, candidate_theme_info)
            await self.mongodb_service.store_esg_question(ESGQuestion(**new_question_dict))
        elif is_better:
            self.logger.warning(f"UPDATE: New candidate '{candidate_qset.main_category_name}' is BETTER. Replacing old question.")
            # Deactivate the old version
            await self.mongodb_service.deactivate_question_set_in_db(str(most_similar_db_q.id))
            # Prepare and store the new version
            new_question_dict = await self._convert_gqs_to_final_dict(candidate_qset, candidate_theme_info)
            new_question_dict['version'] = most_similar_db_q.version + 1
            await self.mongodb_service.store_esg_question(ESGQuestion(**new_question_dict))
        else:
            self.logger.info(f"UPDATE: Existing question '{most_similar_db_q.theme}' is better. Discarding new candidate.")

    async def _find_most_similar_db_question(self, 
                                           candidate_qset: GeneratedQuestionSet, 
                                           active_db_questions: List[ESGQuestion],
                                           threshold=SIMILARITY_THRESHOLD_KG_THEME_UPDATE) -> Optional[ESGQuestion]:
        # ... (Implementation is sound, no changes needed) ...
        if not active_db_questions: return None

        candidate_text = f"{candidate_qset.main_category_name} {candidate_qset.main_category_description}"
        db_q_texts = [f"{q.theme} {q.theme_description_en}" for q in active_db_questions]

        candidate_embedding = (await self.embed_texts([candidate_text]))[0]
        db_embeddings = await self.embed_texts(db_q_texts)
        
        valid_db_embeddings_with_indices = [(i, emb) for i, emb in enumerate(db_embeddings) if emb is not None]
        if not valid_db_embeddings_with_indices or candidate_embedding is None:
            return None

        indices, valid_embs = zip(*valid_db_embeddings_with_indices)
        db_embeddings_matrix = np.vstack(valid_embs)
        
        similarities = cosine_similarity(candidate_embedding.reshape(1, -1), db_embeddings_matrix)
        
        highest_score_idx = np.argmax(similarities)
        highest_score = similarities[0, highest_score_idx]

        if highest_score >= threshold:
            original_db_index = indices[highest_score_idx]
            return active_db_questions[original_db_index]
        
        return None

    async def _find_evidence_and_generate_for_set_q(self, set_question: Dict, custom_query: Optional[str] = None) -> Optional[Dict]:
        search_query = custom_query or f"{set_question.get('theme_set', '')}: {set_question.get('question_text_en', '')}"
        
        evidence_chunks = await self.neo4j_service.find_semantically_similar_chunks(query_text=search_query, top_k=3)
        if not evidence_chunks: return None
        
        validated_evidence = [chunk for chunk in evidence_chunks if chunk.get('score', 0.0) >= 0.67]
        if not validated_evidence: return None

        question_dict = await self._generate_final_dict_for_set_q(set_question, validated_evidence)
        if not question_dict: return None

        used_chunk_ids = {chunk['chunk_id'] for chunk in validated_evidence}
        used_entity_ids = {eid for chunk in validated_evidence for eid in chunk.get('entity_ids', [])}
        
        return {
            "question": question_dict,
            "used_entity_ids": used_entity_ids,
            "used_chunk_ids": used_chunk_ids
        }

    async def _generate_final_dict_for_set_q(
        self,
        set_question: Dict[str, Any],
        validated_evidence: List[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """
        Takes a SET benchmark question and validated evidence to generate sub-questions
        and format the final dictionary for a SET-driven question.
        """
        if not validated_evidence:
            # This check is redundant if the calling function already checks, but good for safety
            return None

        main_topic_str = f"{set_question.get('theme_set', '')}: {set_question.get('question_text_en', '')}"
        self.logger.info(f"Attempting to generate sub-questions for SET ID: {set_question.get('id')}")
        
        context_str = "\n\n---\n\n".join([chunk['text'] for chunk in validated_evidence])
        
        # --- FIX: ปรับปรุง Prompt ให้ยืดหยุ่นมากขึ้น ---
        prompt = f"""
        You are an expert ESG analyst creating a detailed questionnaire for the industrial packaging sector.
        The main topic is derived from the SET benchmark question: "{main_topic_str}"

        Based on the following context extracted from company documents, formulate 2-3 specific and actionable sub-questions that explore the main topic in detail.

        CONTEXT:
        ---
        {context_str}
        ---

        INSTRUCTIONS:
        1.  Your primary goal is to create sub-questions that are directly answerable or verifiable from the PROVIDED CONTEXT.
        2.  If the context is sparse or not perfectly aligned, generate broader sub-questions that are still highly relevant to the main topic, rather than giving up. For example, ask about policies, management approaches, or data collection processes related to the main topic.
        3.  Ensure the questions are distinct and probe for different aspects (e.g., policy, performance, governance).
        4.  Return the result as a valid JSON object with a single key "sub_questions", which is a list of strings. Do not return an empty list unless the context is completely irrelevant.

        Example output:
        {{
            "sub_questions": [
                "What is the company's stated policy on water recycling mentioned in the report?",
                "According to the context, what were the total water withdrawal figures for the last fiscal year?"
            ]
        }}
        """

        try:
            response_object = await self.qg_llm.ainvoke(prompt)
            llm_content = response_object.content
            
            # เพิ่ม Log เพื่อดูผลลัพธ์ดิบจาก LLM
            self.logger.info(f"LLM Raw Output for SET ID {set_question.get('id')}: {llm_content}")

            result = self._extract_json_from_llm_output(llm_content)
            
            if not result or not isinstance(result.get("sub_questions"), list) or not result.get("sub_questions"):
                self.logger.warning(f"LLM did not return a valid list of sub-questions for SET ID: {set_question.get('id')}. Skipping.")
                return None

            sub_questions_list = result["sub_questions"]
            
            # (ส่วนที่เหลือของฟังก์ชันเหมือนเดิม)
            main_q_en = set_question.get('question_text_en', '')
            main_q_th = set_question.get('question_text_th', '') 
            desc_en = f"Questions generated to cover SET benchmark: {main_q_en}"
            sub_q_text_en = "\n".join(f"{i+1}. {q}" for i, q in enumerate(sub_questions_list))

            desc_th, sub_q_text_th = await asyncio.gather(
                self.translate_text_to_thai(desc_en, category_name=main_q_en),
                self.translate_text_to_thai(sub_q_text_en, category_name=main_q_en)
            )

            highest_relevance_score = max(chunk.get('score', 0.0) for chunk in validated_evidence)
            related_set_q_obj = RelatedSETQuestion(
                set_id=set_question.get('id'),
                title_th=main_q_th,
                relevance_score=highest_relevance_score
            )

            # Pydantic models need to be converted to dict for the final structure if the DB model expects dicts.
            # Assuming your _evaluate_and_integrate_new_question handles Pydantic objects or dicts correctly.
            sub_q_detail_obj = SubQuestionDetail(
                sub_question_text_en=sub_q_text_en,
                sub_question_text_th=sub_q_text_th,
                sub_theme_name=f"Details for {set_question.get('id')}",
                category_dimension=set_question.get('dimension'),
                detailed_source_info=f"Generated from {len(validated_evidence)} evidence chunks related to '{main_topic_str}'"
            )

            final_dict = {
                "theme": f"SET Coverage: {set_question.get('theme_set')} ({set_question.get('id')})",
                "category": set_question.get('dimension'),
                "keywords": set_question.get('theme_set'),
                "theme_description_en": desc_en,
                "theme_description_th": desc_th,
                "main_question_text_en": main_q_en,
                "main_question_text_th": main_q_th,
                "sub_questions_sets": [sub_q_detail_obj],
                "related_set_questions": [related_set_q_obj],
                "generation_method": "SET-Driven",
            }
            return final_dict

        except Exception as e:
            self.logger.error(f"Critical error during final dictionary generation for SET ID {set_question.get('id')}: {e}", exc_info=True)
            return None

    async def _run_targeted_gri_phase(self, target_standards: List[str], existing_used_entities: Set[str], existing_used_chunks: Set[str]) -> Tuple[List[Dict], Set[str], Set[str]]:
        final_questions, newly_used_entities, newly_used_chunks = [], set(), set()

        for standard_code in target_standards:
            self.logger.info(f"PHASE 2: Searching for target standard: {standard_code}")
            
            target_chunks = await self.neo4j_service.get_chunks_by_standard_code(standard_code)
            unseen_chunks = [chunk for chunk in target_chunks if chunk['chunk_id'] not in existing_used_chunks]
            
            if not unseen_chunks: continue

            all_entities_in_chunks = {eid for chunk in unseen_chunks for eid in chunk.get('entity_ids', [])}
            unseen_entity_ids = list(all_entities_in_chunks - existing_used_entities)
            
            if not unseen_entity_ids: continue

            subgraph_data = await self._get_entity_graph_data(include_only_entity_ids=unseen_entity_ids)
            
            theme_structures = await self.identify_hierarchical_themes_from_kg(
                entity_graph_data=subgraph_data, 
                min_first_order_community_size=3, 
                min_main_category_fo_community_count=1
            )
            
            if not theme_structures: continue

            self.logger.info(f"PHASE 2: Found {len(theme_structures)} potential themes for {standard_code}. Selecting the largest one.")
            largest_theme = max(theme_structures, key=lambda theme: len(theme.get('_constituent_entity_ids_in_mc', [])))
            
            q_sets, info_map = await self._generate_qsets_from_theme_structures([largest_theme])

            for q_set in q_sets:
                q_dict = await self._convert_gqs_to_final_dict(q_set, info_map[q_set.main_category_name])
                final_questions.append(q_dict)
            
            newly_used_chunks.update({chunk['chunk_id'] for chunk in unseen_chunks})
            newly_used_entities.update(unseen_entity_ids)

        return final_questions, newly_used_entities, newly_used_chunks

    async def _map_themes_to_set_benchmarks(
        self, 
        candidate_q_sets: List[GeneratedQuestionSet], 
        all_set_benchmarks: List[Dict[str, Any]]
    ) -> Tuple[Dict[str, str], List[Dict[str, Any]]]:
        """
        Maps candidate question sets to SET benchmarks to determine coverage.

        Args:
            candidate_q_sets: A list of GeneratedQuestionSet objects to evaluate.
            all_set_benchmarks: The full list of SET benchmark questions.

        Returns:
            A tuple containing:
            - A dictionary mapping covered SET question IDs to the theme name that covers them.
            - A list of SET benchmark questions that remain uncovered.
        """
        self.logger.info(f"Mapping {len(candidate_q_sets)} candidate themes to SET benchmarks...")
        
        set_coverage_map = {}  # { "SET_ID": "theme_name", ... }

        # Create a set of all benchmark IDs for quick lookup
        all_benchmark_ids = {q['id'] for q in all_set_benchmarks}

        for q_set in candidate_q_sets:
            # Find all relevant SET questions for the current candidate theme
            relevant_benchmarks = await self._find_relevant_set_benchmark_questions(
                mc_dimension=q_set.main_category_dimension,
                mc_name=q_set.main_category_name,
                mc_keywords=q_set.main_category_keywords,
                generated_main_q_text=q_set.main_question_text_en,
                generated_sub_q_text=q_set.rolled_up_sub_questions_text_en,
                all_set_benchmarks=all_set_benchmarks
            )
            
            for set_q in relevant_benchmarks:
                set_id = set_q['id']
                # If this SET question hasn't been covered yet, map it to this theme
                if set_id not in set_coverage_map:
                    set_coverage_map[set_id] = q_set.main_category_name
                    self.logger.info(f"SET question '{set_id}' is now covered by theme '{q_set.main_category_name}'.")

        # Determine which SET questions are still uncovered
        covered_set_ids = set(set_coverage_map.keys())
        uncovered_set_qs = [q for q in all_set_benchmarks if q['id'] not in covered_set_ids]
        
        self.logger.info(f"SET mapping complete. Covered: {len(covered_set_ids)}, Uncovered: {len(uncovered_set_qs)}")
        
        return set_coverage_map, uncovered_set_qs


        # --- NEW HELPER FOR PHASE 1 ---

    def _extract_json_from_llm_output(self, llm_output: str) -> Optional[Dict]:
        """
        Attempts to extract a JSON object from the LLM's raw text output.
        (This is a helper function assumed to be present in the class)
        """
        if not llm_output:
            return None
        try:
            match = re.search(r"```json\s*(\{[\s\S]*?\})\s*```", llm_output, re.DOTALL)
            if match:
                json_str = match.group(1)
            else:
                first_brace = llm_output.find('{')
                last_brace = llm_output.rfind('}')
                if first_brace != -1 and last_brace > first_brace:
                    json_str = llm_output[first_brace : last_brace + 1]
                else:
                    return None
            return json.loads(json_str)
        except json.JSONDecodeError:
            self.logger.warning(f"Could not decode JSON from LLM output: {llm_output[:200]}")
            return None

    # ในคลาส QuestionGenerationService
    async def _find_evidence_and_generate_for_set_q(self, set_question: Dict, custom_query: Optional[str] = None) -> Optional[Dict]:
        """
        (User's Trusted Logic)
        Finds evidence and generates a SET question. Returns None if sub-questions cannot be generated.
        This version is adapted to return used node IDs for the 4-phase architecture.
        """
        if custom_query:
            search_query = custom_query
            search_type = "Broad"
        else:
            search_query = f"{set_question.get('theme_set', '')}: {set_question.get('question_text_en', '')}"
            search_type = "Precision"

        self.logger.info(f"Phase 1 ({search_type} Search): Searching evidence for SET ID: {set_question.get('id')}")

        evidence_chunks = await self.neo4j_service.find_semantically_similar_chunks(query_text=search_query, top_k=3)
        if not evidence_chunks:
            return None

        # ใช้ Threshold เดิมที่คุณเชื่อมั่น
        SIMILARITY_THRESHOLD = 0.67
        validated_evidence = [chunk for chunk in evidence_chunks if chunk.get('score', 0.0) >= SIMILARITY_THRESHOLD]
        
        if not validated_evidence:
            self.logger.info(f"Phase 1 ({search_type} Search): Evidence for SET ID {set_question.get('id')} did not meet threshold {SIMILARITY_THRESHOLD}.")
            return None

        self.logger.info(f"Phase 1 ({search_type} Search): Found {len(validated_evidence)} validated chunks for SET ID: {set_question.get('id')}. Attempting generation...")
        context_str = "\n\n---\n\n".join([chunk['text'] for chunk in validated_evidence])

        prompt = f"""
        You are an expert ESG analyst creating a detailed questionnaire.
        The main topic is the SET benchmark question: "{search_query}"
        Based ONLY on the following context extracted from company documents, formulate 2-4 specific and actionable sub-questions that explore the main topic in detail. The sub-questions must be directly answerable from the provided context. Do not invent questions if the context is insufficient.
        CONTEXT:
        ---
        {context_str}
        ---
        Return the result as a JSON object with a single key "sub_questions", which is a list of strings. If you cannot formulate questions from the context, return an empty JSON object {{}} or a JSON with an empty list.
        """

        try:
            response_object = await self.qg_llm.ainvoke(prompt)
            result = self._extract_json_from_llm_output(response_object.content)
            
            # --- LOGIC เดิมของคุณที่เข้มงวด ---
            if not result or "sub_questions" not in result:
                self.logger.warning(f"LLM did not return a valid JSON with 'sub_questions' key for SET ID: {set_question.get('id')}. Skipping.")
                return None

            sub_questions_list = result.get("sub_questions", [])
            if not sub_questions_list:
                self.logger.warning(f"LLM returned an empty list for sub-questions for SET ID: {set_question.get('id')}. Skipping as per original logic.")
                return None
            # --- สิ้นสุด Logic เดิม ---

            # ถ้าผ่านจุดนี้มาได้ แสดงว่า LLM สร้างคำถามย่อยสำเร็จ
            self.logger.info(f"Phase 1 ({search_type} Search): LLM successfully generated {len(sub_questions_list)} sub-questions for SET ID: {set_question.get('id')}.")

            main_q_en = set_question.get('question_text_en', '')
            main_q_th = set_question.get('question_text_th', '')
            desc_en = f"Questions generated to cover SET benchmark: {main_q_en}"
            sub_q_text_en = "\n".join(f"{i+1}. {q}" for i, q in enumerate(sub_questions_list))

            desc_th, sub_q_text_th = await asyncio.gather(
                self.translate_text_to_thai(desc_en),
                self.translate_text_to_thai(sub_q_text_en)
            )

            related_set_q_obj = RelatedSETQuestion(
                set_id=set_question.get('id'),
                title_th=main_q_th,
                relevance_score=validated_evidence[0].get('score')
            )
            
            sub_q_detail_obj = SubQuestionDetail(
                sub_question_text_en=sub_q_text_en,
                sub_question_text_th=sub_q_text_th,
                sub_theme_name=f"Details for {set_question.get('id')}",
                category_dimension=set_question.get('dimension'),
                detailed_source_info=f"Generated from evidence chunks related to '{search_query}'"
            )
            
            question_dict = {
                "theme": f"SET Coverage: {set_question.get('theme_set')} ({set_question.get('id')})",
                "category": set_question.get('dimension'),
                "keywords": set_question.get('theme_set'),
                "theme_description_en": desc_en,
                "theme_description_th": desc_th,
                "main_question_text_en": main_q_en,
                "main_question_text_th": main_q_th,
                "sub_questions_sets": [sub_q_detail_obj],
                "related_set_questions": [related_set_q_obj],
                "generation_method": "SET-Driven"
            }

            # --- การปรับแก้ที่สำคัญ ---
            # รวบรวม ID ที่ใช้แล้วเพื่อส่งกลับ
            used_chunk_ids = {chunk['chunk_id'] for chunk in validated_evidence}
            used_entity_ids = {eid for chunk in validated_evidence for eid in chunk.get('entity_ids', [])}
            
            return {
                "question": question_dict,
                "used_entity_ids": used_entity_ids,
                "used_chunk_ids": used_chunk_ids
            }

        except Exception as e:
            self.logger.error(f"Error during LLM generation for SET ID {set_question.get('id')}: {e}", exc_info=True)
            return None

    # --- NEW HELPER FOR PHASE 2 ---

    async def _is_theme_redundant(self, organic_theme: Dict[str, Any], set_driven_questions: List[Dict[str, Any]]) -> bool:
        organic_theme_text = f"{organic_theme.get('theme', '')} {organic_theme.get('theme_description_en', '')}"
        for set_q_dict in set_driven_questions:
            set_q_text = f"{set_q_dict.get('theme', '')} {set_q_dict.get('theme_description_en', '')}"
            if await self.are_questions_substantially_similar(organic_theme_text, set_q_text, threshold=FINAL_VALIDATION_SIMILARITY_THRESHOLD):
                return True
        return False
        
    async def _evaluate_and_integrate_update(self, 
                                         candidate_qset: GeneratedQuestionSet,
                                         candidate_theme_info: Dict, 
                                         active_db_questions: List[ESGQuestion]):
        most_similar_db_q = await self._find_most_similar_db_question(candidate_qset, active_db_questions)
        
        if not most_similar_db_q:
            self.logger.info(f"UPDATE: Candidate '{candidate_qset.main_category_name}' is novel. Adding as new question.")
            new_question_dict = await self._convert_gqs_to_final_dict(candidate_qset, candidate_theme_info)
            await self.mongodb_service.store_esg_question(ESGQuestion(**new_question_dict))
            return

        self.logger.info(f"UPDATE: Candidate '{candidate_qset.main_category_name}' is similar to existing question '{most_similar_db_q.theme}'. Evaluating quality...")
        is_better, is_same_topic = await self._is_new_theme_better(most_similar_db_q, candidate_qset)

        if not is_same_topic:
            self.logger.info(f"UPDATE: LLM judged topics are different. Adding '{candidate_qset.main_category_name}' as new question.")
            new_question_dict = await self._convert_gqs_to_final_dict(candidate_qset, candidate_theme_info)
            await self.mongodb_service.store_esg_question(ESGQuestion(**new_question_dict))
        elif is_better:
            self.logger.warning(f"UPDATE: New candidate '{candidate_qset.main_category_name}' is BETTER. Replacing old question.")
            await self.mongodb_service.deactivate_question_set_in_db(str(most_similar_db_q.id))
            new_question_dict = await self._convert_gqs_to_final_dict(candidate_qset, candidate_theme_info)
            new_question_dict['version'] = most_similar_db_q.version + 1
            await self.mongodb_service.store_esg_question(ESGQuestion(**new_question_dict))
        else:
            self.logger.info(f"UPDATE: Existing question '{most_similar_db_q.theme}' is better or unchanged. Discarding new candidate.")

    # --- NEW HELPER TO CONVERT GQS to Dict ---

    async def _convert_gqs_to_final_dict(self, gqs: GeneratedQuestionSet, theme_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Converts a GeneratedQuestionSet Pydantic object into the final dictionary format,
        including translation.
        """
        # --- NEW: Translation Step ---
        theme_en = gqs.main_category_name
        desc_en = gqs.main_category_description
        main_q_en = gqs.main_question_text_en
        sub_q_en = gqs.rolled_up_sub_questions_text_en

        # Translate all relevant fields
        theme_th, desc_th, main_q_th, sub_q_th = await asyncio.gather(
            self.translate_text_to_thai(theme_en, category_name=theme_en),
            self.translate_text_to_thai(desc_en, category_name=theme_en),
            self.translate_text_to_thai(main_q_en, category_name=theme_en),
            self.translate_text_to_thai(sub_q_en, category_name=theme_en)
        )
        # --- End of Translation Step ---

        sub_questions_list = []
        if sub_q_en and "No specific sub-questions" not in sub_q_en:
            sub_q_detail = SubQuestionDetail(
                sub_question_text_en=sub_q_en,
                sub_question_text_th=sub_q_th, # Add translated field
                sub_theme_name=f"Detailed Inquiries for {theme_en}",
                category_dimension=gqs.main_category_dimension,
                keywords=gqs.main_category_keywords,
                theme_description_en=gqs.detailed_source_info_for_subquestions or "Generated based on document context."
            )
            sub_questions_list.append(sub_q_detail)

        return {
            "theme": theme_en,
            "theme_description_en": desc_en,
            "main_question_text_en": main_q_en,
            "category": gqs.main_category_dimension,
            "keywords": gqs.main_category_keywords,
            "theme_th": theme_th, # Add translated field
            "theme_description_th": desc_th, # Add translated field
            "main_question_text_th": main_q_th, # Add translated field
            "sub_questions_sets": sub_questions_list,
            "related_set_questions": [],
            "main_category_constituent_entity_ids": theme_info.get("_constituent_entity_ids_in_mc", []),
            "main_category_source_document_references": theme_info.get("_source_document_ids_for_mc", []),
            "generation_method": theme_info.get("generation_method", "KG-Driven")
        }

    async def _evaluate_and_integrate_new_question(self, candidate_dict: Dict[str, Any]):
        """
        Handles versioning for non-baseline runs. Compares a new candidate question
        with existing versions in the DB and decides whether to ADD or REPLACE.
        This version is updated for the new flat dictionary structure.
        """
        theme_name = candidate_dict.get('theme')
        if not theme_name:
            self.logger.error(f"Candidate dictionary is missing a 'theme'. Skipping evaluation. Data: {candidate_dict}")
            return

        self.logger.info(f"Evaluating candidate for theme: '{theme_name}'")

        # Find the latest active version of this theme in the database
        latest_version = await ESGQuestion.find_one(
            ESGQuestion.theme == theme_name, 
            ESGQuestion.is_active == True
        )

        if not latest_version:
            self.logger.info(f"Theme '{theme_name}' is new. Storing as v1.")
            await self.mongodb_service.store_esg_question(ESGQuestion(**candidate_dict))
            return

        # --- Content Comparison Logic ---
        # Compare main question text
        candidate_main_q = candidate_dict.get('main_question_text_en', '')
        is_main_q_different = not await self.are_questions_substantially_similar(
            candidate_main_q, latest_version.main_question_text_en
        )

        # Compare sub-questions text
        # We create a simple representative string for the sub_questions_sets for comparison
        def get_subq_text(subq_list):
            if not subq_list: return ""
            # The list contains SubQuestionDetail objects or dicts
            texts = [
                item.sub_question_text_en if isinstance(item, SubQuestionDetail) else item.get('sub_question_text_en', '') 
                for item in subq_list
            ]
            return " ".join(texts)

        candidate_sub_q_text = get_subq_text(candidate_dict.get('sub_questions_sets', []))
        latest_version_sub_q_text = get_subq_text(latest_version.sub_questions_sets)
        
        is_sub_q_different = not await self.are_questions_substantially_similar(
            candidate_sub_q_text, latest_version_sub_q_text
        )

        # If content has changed, create a new version
        if is_main_q_different or is_sub_q_different:
            self.logger.info(f"Content for theme '{theme_name}' has changed. Creating new version.")
            
            # Deactivate the old version
            await self.mongodb_service.deactivate_question_set_in_db(str(latest_version.id))
            
            # Prepare the new version
            next_version = await self.mongodb_service.get_next_question_version_for_theme(theme_name)
            candidate_dict['version'] = next_version
            await self.mongodb_service.store_esg_question(ESGQuestion(**candidate_dict))
        else:
            self.logger.info(f"Content for theme '{theme_name}' is unchanged. No new version needed.")

    async def _re_evaluate_set_coverage_for_question(self, esg_question: ESGQuestion):
        """
        Re-evaluates and updates the SET benchmark mapping for a given ESGQuestion object.
        """
        self.logger.info(f"Re-evaluating SET coverage for question ID: {esg_question.id}")
        # This function re-uses the logic from _map_themes_to_set_benchmarks
        # but for a single, already-created question object.
        
        # We need the full set of benchmark questions to compare against.
        set_benchmarks = self.mongodb_service.get_set_benchmark_questions()

        question_content = esg_question.main_question.question + "\n" + \
                           "\n".join([sub.question for sub in esg_question.sub_questions])

        related_set_questions = []
        # This logic should ideally be identical to the one in _map_themes_to_set_benchmarks
        # Consider refactoring to avoid code duplication.
        for set_q in set_benchmarks:
            # Using a mock theme structure to reuse the finding logic
            mock_theme = {
                'theme_name': esg_question.theme_name,
                'content_for_embedding': question_content
            }
            is_relevant, score = self._is_theme_relevant_to_set_q(mock_theme, set_q, threshold=0.7)
            if is_relevant:
                related_set_questions.append(RelatedSETQuestion(
                    set_id=set_q['id'],
                    title_th=set_q['title_th'],
                    relevance_score=score
                ))

        if related_set_questions:
            self.logger.info(f"Found {len(related_set_questions)} related SET questions. Updating in DB.")
            await self.mongodb_service.update_question_set_mappings(esg_question.id, related_set_questions)
        else:
            self.logger.info("No related SET questions found after re-evaluation.")

    async def _run_adaptive_theme_identification(self, entity_graph_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        MINIMUM_DESIRED_THEMES = 15
        
        self.logger.info("Attempt 1: Identifying themes with HIGH quality settings (4, 2)...")
        themes = await self.identify_hierarchical_themes_from_kg(
            entity_graph_data=entity_graph_data,
            min_first_order_community_size=4, 
            min_main_category_fo_community_count=2
        )

        if len(themes) < MINIMUM_DESIRED_THEMES:
            self.logger.warning(f"High-quality settings yielded only {len(themes)} themes. Retrying with MEDIUM settings (3, 2)...")
            themes = await self.identify_hierarchical_themes_from_kg(
                entity_graph_data=entity_graph_data,
                min_first_order_community_size=3, 
                min_main_category_fo_community_count=2
            )
        
        if not themes: self.logger.critical("No themes could be generated from any graph-based method.")
        return themes

    async def _generate_qsets_from_theme_structures(self, theme_structures: List[Dict[str, Any]]) -> Tuple[List[GeneratedQuestionSet], Dict[str, Dict[str, Any]]]:
        """Generates GeneratedQuestionSet objects from theme structure dictionaries."""
        q_sets, info_map = [], {}
        tasks = [self.generate_question_for_theme_level(info) for info in theme_structures]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for i, res in enumerate(results):
            if isinstance(res, GeneratedQuestionSet) and res.main_category_name:
                q_sets.append(res)
                info_map[res.main_category_name] = theme_structures[i]
        return q_sets, info_map

    async def _map_set_coverage(self, kg_q_sets: List[GeneratedQuestionSet], benchmarks: List[Dict[str, Any]]) -> Tuple[Dict[str, str], List[Dict[str, Any]]]:
        """Maps SET benchmarks to KG themes and returns the coverage map and a list of uncovered questions."""
        coverage_map = {}
        for q_set in kg_q_sets:
            relevant_qs = await self._find_relevant_set_benchmark_questions(
                mc_dimension=q_set.main_category_dimension, mc_name=q_set.main_category_name,
                mc_keywords=q_set.main_category_keywords, generated_main_q_text=q_set.main_question_text_en,
                generated_sub_q_text=q_set.rolled_up_sub_questions_text_en, all_set_benchmarks=benchmarks)
            for set_q in relevant_qs:
                coverage_map[set_q['id']] = q_set.main_category_name
        
        uncovered_qs = [bq for bq in benchmarks if bq['id'] not in coverage_map]
        print(f"[QG_SERVICE INFO] Initial SET mapping: {len(coverage_map)} covered, {len(uncovered_qs)} uncovered.")
        return coverage_map, uncovered_qs

    # --- ฟังก์ชัน Helper ใหม่สำหรับ `evolve_and_store_questions` ---
    async def _absorb_and_refine_uncovered_set_qs(self, uncovered_qs: List[Dict[str, Any]], kg_q_sets: List[GeneratedQuestionSet], coverage_map: Dict[str, str], info_map: Dict[str, Dict[str, Any]]) -> Tuple[List[GeneratedQuestionSet], Dict[str, str]]:
        """Absorbs uncovered SET questions by refining the best-fit KG theme."""
        q_set_dict = {q.main_category_name: q for q in kg_q_sets}
        for set_q in uncovered_qs:
            best_fit_theme = await self._find_best_fit_kg_theme_for_set_q(set_q, list(q_set_dict.values()))
            if best_fit_theme:
                print(f"[QG_SERVICE REFINE] Refining theme '{best_fit_theme.main_category_name}' to absorb SET ID '{set_q['id']}'...")
                feedback = f"Please revise or add sub-questions to specifically cover: '{set_q.get('question_text_en')}'"
                
                refined_sub_q, refined_info = await self._refine_sub_questions_based_on_feedback(
                    main_category_name=best_fit_theme.main_category_name,
                    main_question_text=best_fit_theme.main_question_text_en,
                    existing_sub_questions=best_fit_theme.rolled_up_sub_questions_text_en,
                    feedback_suggestions=feedback, missing_aspects=f"Specifics of SET Question: {set_q.get('question_text_en')}",
                    original_main_category_info=info_map[best_fit_theme.main_category_name],
                    knowledge_graph_context="N/A", standard_document_excerpts="N/A",
                    main_category_dimension=best_fit_theme.main_category_dimension)

                # Update the QSet object in our dictionary
                best_fit_theme.rolled_up_sub_questions_text_en = refined_sub_q
                best_fit_theme.detailed_source_info_for_subquestions = refined_info
                q_set_dict[best_fit_theme.main_category_name] = best_fit_theme
                
                coverage_map[set_q['id']] = best_fit_theme.main_category_name

        return list(q_set_dict.values()), coverage_map
    
    async def _find_best_fit_kg_theme_for_set_q(
        self,
        set_question: Dict[str, Any],
        generated_kg_q_sets: List[GeneratedQuestionSet]
    ) -> Optional[GeneratedQuestionSet]:
        """Finds the most semantically similar KG-generated theme for an uncovered SET question."""
        if not self.similarity_llm_embedding or not generated_kg_q_sets:
            return None

        set_q_text = set_question.get('question_text_en', '')
        if not set_q_text:
            return None

        # สร้าง Embedding ของ SET Question
        set_q_embedding_list = await asyncio.get_running_loop().run_in_executor(
            None, self.similarity_llm_embedding.embed_documents, [set_q_text]
        )
        if not set_q_embedding_list or not set_q_embedding_list[0]:
            return None
        set_q_embedding = np.array(set_q_embedding_list[0]).reshape(1, -1)

        # สร้าง Embeddings ของ KG Themes ทั้งหมด
        kg_q_set_texts = [f"{q.main_category_name} {q.main_category_description}" for q in generated_kg_q_sets]
        kg_embeddings_list = await asyncio.get_running_loop().run_in_executor(
            None, self.similarity_llm_embedding.embed_documents, kg_q_set_texts
        )
        
        best_match_q_set = None
        highest_score = -1.0

        for i, kg_embedding in enumerate(kg_embeddings_list):
            if kg_embedding:
                current_embedding = np.array(kg_embedding).reshape(1, -1)
                score = cosine_similarity(set_q_embedding, current_embedding)[0][0]
                if score > highest_score:
                    highest_score = score
                    best_match_q_set = generated_kg_q_sets[i]

        # เราอาจจะตั้ง Threshold ขั้นต่ำไว้ที่นี่ เพื่อไม่ให้จับคู่กับธีมที่ไม่เกี่ยวข้องกันเลย
        SIMILARITY_THRESHOLD_FOR_ABSORPTION = 0.5 
        if highest_score >= SIMILARITY_THRESHOLD_FOR_ABSORPTION:
            print(f"[QG_SERVICE ABSORB] Found best-fit KG theme '{best_match_q_set.main_category_name}' for SET ID '{set_question['id']}' with score {highest_score:.2f}")
            return best_match_q_set
        
        print(f"[QG_SERVICE ABSORB] No suitable KG theme found for SET ID '{set_question['id']}'. Max score was {highest_score:.2f}.")
        return None

    async def _prepare_and_upsert_theme_to_db(
        self,
        q_set: GeneratedQuestionSet,
        original_theme_info_from_source: Dict[str, Any], # Either from KG theme structure or mocked for gap-fill
        validation_status: str,
        validation_feedback_list: List[Dict[str, Any]],
        set_benchmark_ids_covered_list: List[str],
        all_historical_main_categories_map: Dict[str, List[ESGQuestion]],
        current_time_utc: datetime,
        api_response_questions_list: List[GeneratedQuestion], # To append for API response
        existing_theme_doc_to_update: Optional[ESGQuestion] = None # For KG theme evolution
    ):
        """Helper to translate, build DB model, and upsert, then add to API response."""
        
        # If updating, the theme name for DB lookup should be the old theme's name.
        # The content (q_set.main_category_name) might be new if the LLM rephrased it.
        db_lookup_theme_name = existing_theme_doc_to_update.theme if existing_theme_doc_to_update else q_set.main_category_name
        # The theme name stored in the new/updated document will be from q_set.main_category_name
        
        main_cat_name_for_translation = q_set.main_category_name # Use the new name for translation context
        
        main_q_text_th = await self.translate_text_to_thai(q_set.main_question_text_en, main_cat_name_for_translation, q_set.main_category_keywords)
        sub_q_text_th = await self.translate_text_to_thai(q_set.rolled_up_sub_questions_text_en, main_cat_name_for_translation, q_set.main_category_keywords)
        theme_desc_th = await self.translate_text_to_thai(q_set.main_category_description, main_cat_name_for_translation, q_set.main_category_keywords)

        main_q_text_th = await self.translate_text_to_thai(q_set.main_question_text_en, q_set.main_category_name, q_set.main_category_keywords)
        sub_q_text_th = await self.translate_text_to_thai(q_set.rolled_up_sub_questions_text_en, q_set.main_category_name, q_set.main_category_keywords)
        theme_desc_th = await self.translate_text_to_thai(q_set.main_category_description, q_set.main_category_name, q_set.main_category_keywords)

        sub_q_detail_list_for_db = []
        if q_set.rolled_up_sub_questions_text_en and \
           "no specific sub-questions" not in q_set.rolled_up_sub_questions_text_en.lower() and \
           q_set.rolled_up_sub_questions_text_en.strip():
            
            sub_theme_desc_en = f"Detailed inquiries related to {q_set.main_category_name}."
            if q_set.detailed_source_info_for_subquestions and "generated to cover SET benchmark" in q_set.detailed_source_info_for_subquestions: # More specific for gap-fill
                 sub_theme_desc_en = q_set.detailed_source_info_for_subquestions

            sub_theme_desc_th = await self.translate_text_to_thai(sub_theme_desc_en, q_set.main_category_name, q_set.main_category_keywords)
            
            current_sub_q_detail = SubQuestionDetail(
                sub_question_text_en=q_set.rolled_up_sub_questions_text_en,
                sub_question_text_th=sub_q_text_th,
                sub_theme_name=f"Overall Sub-Questions for {q_set.main_category_name}", # Or more specific if available
                category_dimension=q_set.main_category_dimension,
                keywords=q_set.main_category_keywords,
                theme_description_en=sub_theme_desc_en,
                theme_description_th=sub_theme_desc_th,
                constituent_entity_ids=q_set.main_category_constituent_entities or [],
                source_document_references=q_set.main_category_source_docs or [],
                detailed_source_info=q_set.detailed_source_info_for_subquestions
            )
            sub_q_detail_list_for_db.append(current_sub_q_detail.model_dump(exclude_none=True))

        db_doc_data = {
            "theme": q_set.main_category_name, "category": q_set.main_category_dimension,
            "main_question_text_en": q_set.main_question_text_en, "main_question_text_th": main_q_text_th,
            "keywords": q_set.main_category_keywords,
            "theme_description_en": q_set.main_category_description, "theme_description_th": theme_desc_th,
            "sub_questions_sets": sub_q_detail_list_for_db,
            "main_category_constituent_entity_ids": q_set.main_category_constituent_entities or [],
            "main_category_source_document_references": q_set.main_category_source_docs or [],
            "generation_method": original_theme_info_from_source.get("generation_method", "unknown_agentic"),
            "metadata_extras": {
                "_main_category_raw_id": original_theme_info_from_source.get("_main_category_raw_id"), # from KG themes
                "_set_gap_fill_id": original_theme_info_from_source.get("_set_gap_fill_id"), # from SET gap fill
                "validation_feedback": validation_feedback_list,
                "set_benchmark_ids_covered": set_benchmark_ids_covered_list,
                 # Store context if it was part of original_theme_info_from_source (e.g., for gap-fill debug)
                "_final_kg_context_used": original_theme_info_from_source.get("_final_kg_context_used"),
                "_final_chunk_context_used": original_theme_info_from_source.get("_final_chunk_context_used"),
            },
            "validation_status": validation_status,
            "updated_at": current_time_utc, "is_active": True, # Mark as active
        }
        await self._upsert_main_question_document_to_db(db_doc_data, all_historical_main_categories_map, current_time_utc, existing_theme_doc_to_update)

        # Add to API response
        api_response_questions_list.append(GeneratedQuestion(
            question_text_en=q_set.main_question_text_en, question_text_th=main_q_text_th,
            category=q_set.main_category_dimension, theme=q_set.main_category_name, is_main_question=True
        ))
        if sub_q_detail_list_for_db:
            sq_api_data = sub_q_detail_list_for_db[0]
            api_response_questions_list.append(GeneratedQuestion(
                question_text_en=sq_api_data["sub_question_text_en"], question_text_th=sq_api_data.get("sub_question_text_th"),
                category=sq_api_data["category_dimension"], theme=q_set.main_category_name,
                sub_theme_name=sq_api_data["sub_theme_name"], is_main_question=False,
                additional_info={"detailed_source_info_for_subquestions": sq_api_data.get("detailed_source_info")}
            ))    

    async def _upsert_main_question_document_to_db( 
        self, 
        main_category_db_data: Dict[str, Any], 
        all_historical_main_categories_map: Dict[str, List[ESGQuestion]], 
        current_time_utc: datetime,
        existing_theme_doc_to_update: Optional[ESGQuestion] = None
    ):
        # ... (เหมือนโค้ดที่คุณให้มาล่าสุด, ตรวจสอบ content_changed โดยเทียบ main_question_text_en และ sub_questions_sets) ...
        
        # Use db_lookup_theme_name for fetching history and potentially deactivating old versions
        # The actual theme name to be saved in the document is main_category_db_data["theme"] (which comes from q_set.main_category_name)
        theme_name_for_history_lookup = existing_theme_doc_to_update.theme if existing_theme_doc_to_update else main_category_db_data["theme"]
        
        historical_versions = all_historical_main_categories_map.get(theme_name_for_history_lookup, [])
        
        # latest_db_version_doc is the one we might be updating OR the one that is being superseded by a name change
        latest_db_version_doc: Optional[ESGQuestion] = None
        if existing_theme_doc_to_update:
            latest_db_version_doc = existing_theme_doc_to_update
        elif historical_versions: # This case is for new themes or SET themes where name doesn't change
            latest_db_version_doc = historical_versions[0]

        content_changed = False
        if latest_db_version_doc:
            if not await self.are_questions_substantially_similar(main_category_db_data["main_question_text_en"], latest_db_version_doc.main_question_text_en):
                content_changed = True
            if not content_changed: 
                current_sub_q_sets_dicts = main_category_db_data.get("sub_questions_sets", []) # List of Dict
                db_sub_q_sets_model = latest_db_version_doc.sub_questions_sets # List of SubQuestionDetail Model
                
                # Convert current dicts to a comparable form (e.g., tuple of sorted (name, text) for each sub_q_set)
                def get_sub_q_signatures_from_dicts(sub_q_sets_list_of_dicts):
                    signatures = []
                    for sq_dict in sub_q_sets_list_of_dicts:
                        text_en = sq_dict.get("sub_question_text_en", "")
                        signatures.append((sq_dict.get("sub_theme_name",""), text_en)) # Assuming sub_theme_name exists in dict
                    return sorted(signatures)

                def get_sub_q_signatures_from_models(sub_q_sets_list_of_models: List[SubQuestionDetail]):
                    signatures = []
                    for sq_model in sub_q_sets_list_of_models:
                        signatures.append((sq_model.sub_theme_name, sq_model.sub_question_text_en))
                    return sorted(signatures)

                current_signatures = get_sub_q_signatures_from_dicts(current_sub_q_sets_dicts)
                db_signatures = get_sub_q_signatures_from_models(db_sub_q_sets_model)

                if current_signatures != db_signatures:
                    content_changed = True
        else: content_changed = True

        upsert_data_for_document = main_category_db_data.copy()
        if latest_db_version_doc and not content_changed:
            # print(f"[QG_SERVICE INFO DB_UPSERT] Main Category '{theme_name_for_history_lookup}' content SIMILAR. Checking metadata/status.")
            update_payload = {}; needs_db_update = not latest_db_version_doc.is_active
            if not latest_db_version_doc.is_active: update_payload["is_active"] = True
            fields_to_check = ["category", "keywords", "theme_description_en", "theme_description_th", 
                               "main_category_constituent_entity_ids", "main_category_source_document_references",
                               "generation_method", "metadata_extras"]
            for field_key in fields_to_check:
                new_val, old_val = upsert_data_for_document.get(field_key), getattr(latest_db_version_doc, field_key, None)
                is_diff = False
                if isinstance(new_val, list) and isinstance(old_val, list): is_diff = sorted(new_val) != sorted(old_val)
                elif isinstance(new_val, dict) and isinstance(old_val, dict): is_diff = new_val != old_val
                elif new_val != old_val: is_diff = True
                if is_diff: update_payload[field_key] = new_val; needs_db_update = True
            if needs_db_update:
                update_payload["updated_at"] = current_time_utc
                await ESGQuestion.find_one(ESGQuestion.id == latest_db_version_doc.id).update({"$set": update_payload})
                # print(f"[QG_SERVICE INFO DB_UPSERT] Updated metadata/status for MC '{theme_name_for_history_lookup}'.")
                updated_doc = await ESGQuestion.get(latest_db_version_doc.id)
                if updated_doc: # Update local cache
                    idx = next((i for i,q in enumerate(all_historical_main_categories_map.get(theme_name_for_history_lookup,[])) if q.id == updated_doc.id),-1)
                    if idx != -1: all_historical_main_categories_map[theme_name_for_history_lookup][idx] = updated_doc
                    else: all_historical_main_categories_map.setdefault(theme_name_for_history_lookup, []).insert(0, updated_doc); all_historical_main_categories_map[theme_name_for_history_lookup].sort(key=lambda q:q.version, reverse=True)
        else:
            if latest_db_version_doc:
                # If existing_theme_doc_to_update is provided, we are updating it.
                # The theme name in main_category_db_data["theme"] might be different if LLM rephrased it.
                # We need to deactivate the old doc (identified by existing_theme_doc_to_update.id)
                # and then insert the new one with potentially a new theme name but incremented version from the old one.
                if existing_theme_doc_to_update and existing_theme_doc_to_update.is_active:
                     print(f"[QG_SERVICE INFO DB_UPSERT] Deactivating old version of theme '{existing_theme_doc_to_update.theme}' (ID: {existing_theme_doc_to_update.id}) before update.")
                     await ESGQuestion.find_one(ESGQuestion.id == existing_theme_doc_to_update.id).update({"$set": {"is_active": False, "updated_at": current_time_utc}})
                
                # If it's not an explicit update of an existing doc (e.g. new theme or SET theme matching by name)
                # and the content is different from the latest active version of that name.
                elif latest_db_version_doc.is_active and latest_db_version_doc.theme == main_category_db_data["theme"]: # Ensure we are deactivating the correct named theme
                    print(f"[QG_SERVICE INFO DB_UPSERT] Theme '{main_category_db_data['theme']}' content CHANGED. Deactivating old version.")
                    await ESGQuestion.find_one(ESGQuestion.id == latest_db_version_doc.id).update({"$set": {"is_active": False, "updated_at": current_time_utc}})
                
                upsert_data_for_document["version"] = latest_db_version_doc.version + 1
                print(f"[QG_SERVICE INFO DB_UPSERT] Creating new version {upsert_data_for_document['version']} for theme '{main_category_db_data['theme']}'.")

            else: # Absolutely new theme name
                print(f"[QG_SERVICE INFO DB_UPSERT] Theme '{main_category_db_data['theme']}' is NEW. Inserting v1.")
                upsert_data_for_document["version"] = 1
            
            upsert_data_for_document["generated_at"] = current_time_utc; upsert_data_for_document["updated_at"] = current_time_utc; upsert_data_for_document["is_active"] = True
            new_doc_to_insert = ESGQuestion(**upsert_data_for_document)
            try:
                inserted_doc = await new_doc_to_insert.insert()
                # Add to the map using the theme name that was actually saved.
                all_historical_main_categories_map.setdefault(inserted_doc.theme, []).insert(0, inserted_doc)
                all_historical_main_categories_map[inserted_doc.theme].sort(key=lambda q: q.version, reverse=True)
                print(f"[QG_SERVICE INFO DB_UPSERT] Successfully inserted/updated theme '{inserted_doc.theme}' as version {inserted_doc.version}.")
            except Exception as e: 
                print(f"[QG_SERVICE ERROR DB_UPSERT] Insert/Update for theme '{new_doc_to_insert.theme}': {e}"); 
                print(f"Data: {upsert_data_for_document}"); 
                traceback.print_exc()

    async def _is_new_theme_better(self, old_q_set: ESGQuestion, new_q_set: GeneratedQuestionSet) -> Tuple[bool, bool]:
        """
        Determines if the new theme is substantially about the same topic and better than the old theme.
        Returns: (is_better, is_same_topic)
        """
        old_main_q = old_q_set.main_question_text_en
        old_sub_q_parts = []
        for sub_set in old_q_set.sub_questions_sets:
            old_sub_q_parts.append(f"- {sub_set.sub_theme_name}: {sub_set.sub_question_text_en}")
        old_sub_q = "\n".join(old_sub_q_parts)
        old_desc = old_q_set.theme_description_en or ""

        new_main_q = new_q_set.main_question_text_en
        new_sub_q = new_q_set.rolled_up_sub_questions_text_en # This is already a formatted string
        new_desc = new_q_set.main_category_description or ""

        prompt_template_str = """
        You are a Senior ESG Reporting Analyst. Your task is to critically compare two versions of an ESG question set.

        Theme A (Old):
        Description: "{old_desc}"
        Main Question: "{old_main_q}"
        Sub-Questions: 
        {old_sub_q}

        Theme B (New):
        Description: "{new_desc}"
        Main Question: "{new_main_q}"
        Sub-Questions: 
        {new_sub_q}

        Evaluation Criteria:
        1.  **Topic Equivalence:** Is Theme B substantially addressing the same core ESG topic as Theme A? (Respond True/False)
        2.  **Quality Improvement**: If Topic Equivalence is True, does Theme B represent a **significant and meaningful improvement** over Theme A? 
            - A 'significant improvement' means Theme B introduces a **new material aspect, a new KPI, a new governance check, or a fundamentally clearer strategic angle**.
            - **Simple rephrasing, minor additions of common words, or slight restructuring of the same questions DO NOT count as a significant improvement.** (Respond True/False. If Topic Equivalence is False, this must also be False).

        Output ONLY a single, valid JSON object with keys: "is_same_topic" (boolean) and "new_is_better" (boolean).
        """
        prompt = PromptTemplate.from_template(prompt_template_str)
        formatted_prompt = prompt.format(
            old_desc=old_desc, old_main_q=old_main_q, old_sub_q=old_sub_q,
            new_desc=new_desc, new_main_q=new_main_q, new_sub_q=new_sub_q
        )

        try:
            response = await self.qg_llm.ainvoke(formatted_prompt)
            llm_output = response.content.strip()
            parsed_json = self._extract_json_from_llm_output(llm_output)
            
            if parsed_json and "is_same_topic" in parsed_json and "new_is_better" in parsed_json:
                is_same = bool(parsed_json["is_same_topic"])
                is_better = bool(parsed_json["new_is_better"])
                if not is_same: # If not same topic, new cannot be better in this context
                    return False, False 
                return is_better, is_same
            else:
                print(f"[QG_SERVICE KG_EVOLUTION_COMPARE_ERROR] LLM output parsing failed for theme comparison. Output: {llm_output}")
                return False, False # Default to not better, not same topic on error
        except Exception as e:
            print(f"[QG_SERVICE KG_EVOLUTION_COMPARE_ERROR] Exception during LLM theme comparison: {e}")
            return False, False

    async def cluster_consolidated_themes_with_llm(self, consolidated_themes: List[Dict[str, Any]], num_main_categories_target: int = 5) -> Optional[Dict[str, Any]]:
        # ... (เหมือนเดิม)
        if not consolidated_themes: return None
        print(f"[QG_SERVICE LOG] Fallback: LLM clustering for {len(consolidated_themes)} themes into ~{num_main_categories_target} main categories...")
        themes_context_for_llm = "".join([
            f"Theme {i+1}:\n  Name: {t.get('theme_name_en', 'N/A')}\n  Description: {t.get('description_en', 'N/A')}\n  Keywords: {t.get('keywords_en', 'N/A')}\n---\n"
            for i, t in enumerate(consolidated_themes)
        ])
        prompt_template_str = """
        You are an expert ESG strategist. Group the following detailed ESG themes into approximately {num_main_categories} broader main ESG categories for an industrial packaging factory.
        Detailed Themes:
        --- DETAILED THEMES START ---
        {detailed_themes_context}
        --- DETAILED THEMES END ---
        Output ONLY a single, valid JSON object. Top-level key "main_theme_categories", a list of objects. Each object: "main_category_name_en", "main_category_description_en", "consolidated_theme_names_in_category" (list of exact names from input).
        """
        prompt = PromptTemplate.from_template(prompt_template_str)
        formatted_prompt = prompt.format(detailed_themes_context=themes_context_for_llm, num_main_categories=num_main_categories_target)
        try:
            response = await self.qg_llm.ainvoke(formatted_prompt)
            llm_output = response.content.strip(); json_str = ""
            match = re.search(r"```json\s*(\{[\s\S]*?\})\s*```", llm_output, re.DOTALL | re.MULTILINE)
            if match: json_str = match.group(1)
            else:
                fb, lb = llm_output.find('{'), llm_output.rfind('}')
                if fb != -1 and lb != -1 and lb > fb: json_str = llm_output[fb : lb + 1]
                else: return None
            hierarchical_data = json.loads(json_str)
            return hierarchical_data if "main_theme_categories" in hierarchical_data else None
        except Exception as e: print(f"[QG_SERVICE ERROR] LLM clustering fallback failed: {e}"); return None

    async def _llm_as_evaluator(self, benchmark_question_detail: Dict[str, Any], 
                               generated_main_category_name: str, 
                               generated_main_question_en: str, 
                               generated_sub_questions_en: str, 
                               generated_dimension: str, 
                               knowledge_graph_context: Optional[str], 
                               standard_document_excerpts: Optional[str]) -> Dict[str, Any]:
        evaluator_prompt_template_str = """
        You are an expert ESG Reporting Analyst tasked with evaluating the coverage of generated questions against a benchmark question from the Stock Exchange of Thailand (SET).

        Benchmark SET Question Details:
        - Dimension: {benchmark_dimension}
        - Theme: {benchmark_theme_set}
        - Question (Thai): {benchmark_question_th}
        - Question (English): {benchmark_question_en}

        Generated Question Set for Main Category "{generated_main_category_name}" (Dimension: {generated_dimension}):
        - Generated Main Question (English): {generated_main_question_en}
        - Generated Rolled-up Sub-Questions (English):
        {generated_sub_questions_en}

        Supporting Context for "{generated_main_category_name}" (Use this to understand if missing aspects could be covered):
        Knowledge Graph Hints:
        {knowledge_graph_context}
        Relevant Document Excerpt Hints:
        {standard_document_excerpts}

        Evaluation Task:
        1.  **Coverage Assessment**: Does the "Generated Question Set" (both Main and Sub-Questions) adequately cover the INTENT and SCOPE of the "Benchmark SET Question"?
            Choose one: "Full", "Partial", "None".
        2.  **Missing Aspects**: If coverage is "Partial" or "None", what specific aspects, intents, data points, or nuances from the "Benchmark SET Question" are missing or inadequately covered by the "Generated Question Set"? Be specific.
        3.  **Redundancies**: Are there any obvious redundancies where the generated questions ask for the exact same information as the benchmark in a less effective way? (Briefly note if any).
        4.  **Suggestions for Improvement**: Based on the "Supporting Context", suggest specific new sub-questions (in English) or modifications to the existing "Generated Sub-Questions" to fill the identified gaps and achieve "Full" coverage for the "Benchmark SET Question". If coverage is already "Full", state "No suggestions needed".

        Output ONLY a single, valid JSON object with the following exact keys: "coverage_assessment", "missing_aspects", "redundancies", "suggested_improvements".
        The value for "suggested_improvements" should be a string containing actionable suggestions or new question texts.
        """
        prompt = PromptTemplate.from_template(evaluator_prompt_template_str)
        
        # Ensure benchmark_question_en exists, fallback to Thai if necessary
        benchmark_q_en_text = benchmark_question_detail.get('question_text_en', benchmark_question_detail['question_text_th'])

        formatted_prompt = prompt.format(
            benchmark_dimension=benchmark_question_detail['dimension'],
            benchmark_theme_set=benchmark_question_detail['theme_set'],
            benchmark_question_th=benchmark_question_detail['question_text_th'],
            benchmark_question_en=benchmark_q_en_text,
            generated_main_category_name=generated_main_category_name,
            generated_dimension=generated_dimension,
            generated_main_question_en=generated_main_question_en,
            generated_sub_questions_en=generated_sub_questions_en,
            knowledge_graph_context=knowledge_graph_context if knowledge_graph_context else "Not available.",
            standard_document_excerpts=standard_document_excerpts if standard_document_excerpts else "Not available."
        )
        try:
            # Assuming self.qg_llm is appropriate for evaluation, or you might have a dedicated evaluator LLM
            response = await self.qg_llm.ainvoke(formatted_prompt) 
            llm_output = response.content.strip()
            eval_data = self._extract_json_from_llm_output(llm_output) # Use the helper
            if eval_data and "coverage_assessment" in eval_data:
                return eval_data
            else:
                print(f"[QG_SERVICE AGENT EVAL] Failed to parse evaluation for MC '{generated_main_category_name}'. LLM output: {llm_output}")
                # Fallback structure
                return {"coverage_assessment": "Error_Parsing", "missing_aspects": "LLM output parsing failed.", "redundancies": "", "suggested_improvements": "No suggestions due to parsing error."}
        except Exception as e:
            print(f"[QG_SERVICE AGENT EVAL] Error during LLM evaluation for MC '{generated_main_category_name}': {e}")
            # Fallback structure
            return {"coverage_assessment": "Error_Exception", "missing_aspects": str(e), "redundancies": "", "suggested_improvements": "No suggestions due to exception."}
        
    def _extract_json_from_llm_output(self, llm_output: str) -> Optional[Dict[str, Any]]:
        """
        Attempts to extract a JSON object from the LLM's output string.
        Handles cases where JSON is embedded within triple backticks or is the main content.
        """
        try:
            # Try to find JSON within ```json ... ```
            match = re.search(r"```json\s*(\{[\s\S]*?\})\s*```", llm_output, re.DOTALL | re.MULTILINE)
            if match:
                json_str = match.group(1)
                return json.loads(json_str)
            else:
                # Try to find JSON that starts with { and ends with }
                first_brace = llm_output.find('{')
                last_brace = llm_output.rfind('}')
                if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
                    json_str = llm_output[first_brace : last_brace + 1]
                    return json.loads(json_str)
            print(f"[QG_SERVICE JSON_EXTRACT] No JSON object found in LLM output: {llm_output[:500]}") # Log if no JSON
            return None
        except json.JSONDecodeError as e:
            print(f"[QG_SERVICE JSON_EXTRACT] JSONDecodeError: {e}. LLM output: {llm_output[:500]}") # Log decode error
            return None

    async def _find_relevant_set_benchmark_questions(
            self,
            mc_dimension: str,
            mc_name: str,
            mc_keywords: Optional[str],
            generated_main_q_text: str,
            generated_sub_q_text: str,
            all_set_benchmarks: List[Dict[str, Any]],
            relevance_threshold: float = 0.65
        ) -> List[Dict[str, Any]]:
            """
            Finds relevant SET benchmark questions using a combination of keyword matching
            and semantic similarity on the content of generated main and sub-questions.
            """
            relevant_q_dict: Dict[str, Dict[str, Any]] = {} # ใช้ dict เพื่อป้องกันการเพิ่ม SET ID ซ้ำ

            # 1. ทำความสะอาดและรวมเนื้อหาคำถามที่ระบบสร้าง
            placeholder_main_q = f"What is the company's overall strategic approach and commitment to {mc_name}?"
            clean_generated_main_q = generated_main_q_text if generated_main_q_text != placeholder_main_q else ""
            
            placeholder_sub_q = "No specific sub-questions were generated for this main category."
            clean_generated_sub_q = generated_sub_q_text if placeholder_sub_q not in generated_sub_q_text else ""
            
            combined_generated_q_content = f"{clean_generated_main_q} {clean_generated_sub_q}".strip()
            combined_generated_q_content_lower = combined_generated_q_content.lower()

            # 2. เตรียม Keywords จาก Theme ที่ระบบสร้าง (สำหรับการ matching แบบเดิม)
            mc_name_lower = mc_name.lower()
            mc_keywords_list = [k.strip().lower() for k in (mc_keywords or "").split(',') if k.strip()]
            generated_content_keywords = set(re.findall(r'\b\w{4,}\b', combined_generated_q_content_lower))

            # 3. ถ้ามีเนื้อหาคำถามที่ระบบสร้างและมี embedding model, สร้าง embedding สำหรับเนื้อหานั้น
            generated_q_embedding = None
            if combined_generated_q_content and self.similarity_llm_embedding:
                try:
                    # embed_query is for single text, embed_documents for list
                    loop = asyncio.get_running_loop()
                    generated_q_embedding_list = await loop.run_in_executor(None, self.similarity_llm_embedding.embed_documents, [combined_generated_q_content_lower])
                    if generated_q_embedding_list and generated_q_embedding_list[0]:
                        generated_q_embedding = np.array(generated_q_embedding_list[0]).reshape(1, -1)
                except Exception as e_embed_gen:
                    print(f"[QG_SERVICE FIND_RELEVANT_SET] Error embedding generated question content for '{mc_name}': {e_embed_gen}")


            for set_q_benchmark in all_set_benchmarks:
                if set_q_benchmark['dimension'] != mc_dimension:
                    continue

                set_theme_lower = set_q_benchmark['theme_set'].lower()
                set_q_text_en_lower = set_q_benchmark.get('question_text_en', '').lower()
                set_question_keywords = set(re.findall(r'\b\w{4,}\b', set_q_text_en_lower))

                match_found = False

                # --- Matching Criteria ---

                # Criterion A: Original logic (Generated Theme Name/Keywords vs. SET Theme Name / SET Question Text)
                if (mc_name_lower in set_theme_lower or
                    any(kw in set_theme_lower for kw in mc_keywords_list) or
                    mc_name_lower in set_q_text_en_lower or
                    (mc_keywords_list and any(kw in set_q_text_en_lower for kw in mc_keywords_list))):
                    match_found = True

                # Criterion B: Keywords from Generated Question Content vs. SET Question Text Keywords
                if not match_found and generated_content_keywords and set_question_keywords:
                    common_keywords = generated_content_keywords.intersection(set_question_keywords)
                    if len(common_keywords) >= 2: # Example threshold
                        match_found = True
                
                # Criterion C: Direct substring check (SET question text within combined generated questions)
                if not match_found and set_q_text_en_lower and combined_generated_q_content_lower:
                    if set_q_text_en_lower in combined_generated_q_content_lower: # SET Q text is part of generated Q
                        match_found = True
                    # elif combined_generated_q_content_lower in set_q_text_en_lower: # Generated Q is part of SET Q (less likely for relevance)
                    #     match_found = True


                # Criterion D: Semantic Similarity (if generated question embedding is available)
                if not match_found and generated_q_embedding is not None and set_q_text_en_lower and self.similarity_llm_embedding:
                    try:
                        loop = asyncio.get_running_loop()
                        set_q_embedding_list = await loop.run_in_executor(None, self.similarity_llm_embedding.embed_documents, [set_q_text_en_lower])
                        if set_q_embedding_list and set_q_embedding_list[0]:
                            set_q_embedding = np.array(set_q_embedding_list[0]).reshape(1, -1)
                            similarity_score = cosine_similarity(generated_q_embedding, set_q_embedding)[0][0]
                            
                            # print(f"DEBUG: Semantic similarity for '{mc_name}' vs SET ID '{set_q_benchmark['id']}': {similarity_score:.4f} (Threshold: {relevance_threshold})")
                            if similarity_score >= relevance_threshold:
                                match_found = True
                                # print(f"DEBUG: MATCHED by semantic similarity: '{mc_name}' with SET '{set_q_benchmark['id']}'")
                    except Exception as e_sim_set:
                        print(f"[QG_SERVICE FIND_RELEVANT_SET] Error during semantic similarity for SET Q '{set_q_benchmark['id']}': {e_sim_set}")

                if match_found:
                    relevant_q_dict[set_q_benchmark['id']] = set_q_benchmark
            
            final_relevant_q_list = list(relevant_q_dict.values())
            if final_relevant_q_list: # Log only if any relevant questions were found
                print(f"[QG_SERVICE FIND_RELEVANT_SET] For MC '{mc_name}' (Dim: {mc_dimension}), found {len(final_relevant_q_list)} relevant SET Qs.")
                # for q_found in final_relevant_q_list:
                #     print(f"  - Relevant SET ID: {q_found['id']} - {q_found['theme_set']}")
            return final_relevant_q_list

    async def _refine_sub_questions_based_on_feedback(
        self, 
        main_category_name: str, 
        main_question_text: str, 
        existing_sub_questions: str, 
        feedback_suggestions: Optional[str], 
        missing_aspects: Optional[str], 
        original_main_category_info: Dict[str, Any], # This contains 'consolidated_themes'
        knowledge_graph_context: Optional[str], 
        standard_document_excerpts: Optional[str],
        main_category_dimension: str # Added to provide full context for refinement prompt
    ) -> Tuple[str, str]: # Returns (refined_sub_questions_text_en, refined_detailed_source_info)
        
        # Prepare super_context_from_sub_themes based on original_main_category_info
        super_context_from_sub_themes = "No specific sub-themes were detailed for this main category."
        consolidated_sub_themes_in_mc = original_main_category_info.get("consolidated_themes", [])
        if consolidated_sub_themes_in_mc:
            temp_super_context_parts = []
            char_count_super_ctx = 0
            # Max characters for sub-theme details in sub-question prompt (consistent with generate_question_for_theme_level)
            MAX_CHARS_SUPER_CTX = 10000 
            for sub_theme_data_for_ctx in consolidated_sub_themes_in_mc[:7]: # Sample up to 7
                s_name = sub_theme_data_for_ctx.get("theme_name_en", "N/A")
                s_desc = sub_theme_data_for_ctx.get("description_en", "N/A")
                s_keywords = sub_theme_data_for_ctx.get("keywords_en", "N/A")
                s_dim = sub_theme_data_for_ctx.get("dimension", "N/A")
                entry = f"Sub-Theme: {s_name}\n  Description: {s_desc}\n  Keywords: {s_keywords}\n  Dimension: {s_dim}\n"
                if char_count_super_ctx + len(entry) <= MAX_CHARS_SUPER_CTX:
                    temp_super_context_parts.append(entry)
                    char_count_super_ctx += len(entry)
                else:
                    break
            super_context_from_sub_themes = "\n---\n".join(temp_super_context_parts)
        
        if not super_context_from_sub_themes.strip() and consolidated_sub_themes_in_mc:
            super_context_from_sub_themes = f"This main category generally covers topics such as: {', '.join([st.get('theme_name_en', 'Unnamed Sub-Theme') for st in consolidated_sub_themes_in_mc[:3]])}."
        
        refinement_prompt_template_str = """
        You are an ESG Question Refinement Specialist for an industrial packaging factory.
        The Main ESG Category is: "{main_category_name}" (Dimension: {main_category_dimension})
        The Main Question for this category is: "{main_question_text}"

        Current Sub-Questions that need refinement:
        {existing_sub_questions}

        Feedback for Refinement:
        - Missing Aspects to Cover: {missing_aspects}
        - Suggestions for Improvement/New Questions: {feedback_suggestions}

        Supporting Context (use this to inform your refinements):
        Knowledge Graph Hints:
        {knowledge_graph_context}
        Relevant Document Excerpt Hints:
        {standard_document_excerpts}
        Overview of Sub-Themes in this Main Category: 
        {super_context_from_sub_themes} 

        Task:
        Revise the "Current Sub-Questions" to:
        1.  Address the "Missing Aspects to Cover".
        2.  Incorporate the "Suggestions for Improvement/New Questions".
        3.  Ensure the revised set still consists of 3-5 specific, actionable, and data-driven sub-questions relevant to the Main Category and its dimension.
        4.  Maintain relevance to an industrial packaging factory.
        5.  If possible, specify sources if evident from the context.

        Output ONLY a single, valid JSON object with these exact keys:
        - "refined_sub_questions_text_en": A string containing ONLY the revised 3-5 sub-questions, each numbered and on a new line. If no meaningful refinement is possible or original questions are already good after considering feedback, you can return the original sub-questions.
        - "refined_detailed_source_info": A brief textual summary of how the contexts and feedback were used for refinement, or specific sources if attributable for the refined questions.
        """
        prompt = PromptTemplate.from_template(refinement_prompt_template_str)
        formatted_prompt = prompt.format(
            main_category_name=main_category_name,
            main_category_dimension=main_category_dimension,
            main_question_text=main_question_text,
            existing_sub_questions=existing_sub_questions,
            missing_aspects=missing_aspects or "None specified.",
            feedback_suggestions=feedback_suggestions or "None specified.",
            knowledge_graph_context=knowledge_graph_context or "Not available.",
            standard_document_excerpts=standard_document_excerpts or "Not available.",
            super_context_from_sub_themes=super_context_from_sub_themes
        )

        default_source_info = f"Sub-questions refined for {main_category_name} based on feedback. Contexts were used for grounding."
        try:
            response = await self.qg_llm.ainvoke(formatted_prompt)
            llm_output = response.content.strip()
            json_output = self._extract_json_from_llm_output(llm_output)
            if json_output and "refined_sub_questions_text_en" in json_output:
                refined_text = json_output.get("refined_sub_questions_text_en", existing_sub_questions)
                refined_info = json_output.get("refined_detailed_source_info", default_source_info)
                return refined_text, refined_info
            else:
                print(f"[QG_SERVICE AGENT REFINE] Failed to get valid JSON refinement for MC '{main_category_name}'. LLM output: {llm_output}")
                return existing_sub_questions, "Refinement parsing failed, using existing sub-questions."
        except Exception as e:
            print(f"[QG_SERVICE AGENT REFINE] Error refining sub-questions for MC '{main_category_name}': {e}")
            return existing_sub_questions, "Refinement exception, using existing sub-questions."
        
    async def _generate_questions_for_set_gap_fill(
        self, set_question_to_cover: Dict[str, Any],
        main_question_text_en: str, 
        knowledge_graph_context: Optional[str],
        standard_document_excerpts: Optional[str],
        target_dimension: str
    ) -> Tuple[Optional[str], Optional[str]]: # (sub_questions_text_en, detailed_source_info)
        """
        Generates sub-questions for a SET benchmark. Emphasizes strict context adherence.
        """
        prompt_template_str = """
    You are an expert ESG consultant for an industrial packaging factory.
    You are tasked with generating detailed sub-questions to ensure full coverage of a specific Stock Exchange of Thailand (SET) benchmark question.

    The SET Benchmark Question to Cover:
    - Dimension: {set_dimension}
    - Theme: {set_theme}
    - English Text: "{set_question_en}"
    - Thai Text: "{set_question_th}"

    This SET question serves as the Main Question we need to elaborate on. It is: "{main_question_text_en}"

    Supporting Context (from available ESG documents relevant to an industrial packaging factory):
    Knowledge Graph Hints:
    {knowledge_graph_context}
    Relevant Document Excerpt Hints:
    {standard_document_excerpts}

    Task:
    Based STRICTLY AND SOLELY on the provided "Supporting Context" and the "SET Benchmark Question to Cover", formulate a set of 2-4 specific, actionable, and data-driven Sub-Questions.
    These Sub-Questions should:
    1.  Directly help answer or provide detailed supporting information for the "SET Benchmark Question to Cover" AND be grounded in the "Supporting Context".
    2.  Explore the MOST CRITICAL and REPRESENTATIVE aspects from the "Supporting Context" that DIRECTLY relate to the SET question.
    3.  CRITICAL INSTRUCTION: If the "Supporting Context" (Knowledge Graph Hints and Document Excerpts) does NOT contain specific, directly relevant information to formulate meaningful sub-questions for the SET benchmark, you MUST output the exact phrase "No specific sub-questions can be generated due to insufficient directly relevant context." in the "rolled_up_sub_questions_text_en" field. Do NOT generate generic questions or questions based on general knowledge if the provided context is lacking or irrelevant to the SET question.
    4.  Elicit a mix of (if context allows and is relevant):
        a. Policies & Commitments
        b. Strategies & Processes
        c. Performance & Metrics
        d. Governance & Oversight
    5.  Be relevant to an industrial packaging factory IF AND ONLY IF the context supports it.
    6.  **VERY IMPORTANT: Each sub-question MUST be a single, distinct question. DO NOT combine multiple questions into one sentence using "and" or by listing different topics.**


    Output ONLY a single, valid JSON object with these exact keys:
    - "rolled_up_sub_questions_text_en": A string containing ONLY 2 sub-questions, each numbered and on a new line, OR the specific insufficiency message mentioned in instruction 3.
    - "detailed_source_info_for_subquestions": A brief textual summary of how the contexts were used, or specific sources if attributable. If context was insufficient, state that clearly, referencing the lack of direct relevance in the provided context.
        """
        prompt = PromptTemplate.from_template(prompt_template_str)
        formatted_prompt = prompt.format(
            set_dimension=set_question_to_cover['dimension'],
            set_theme=set_question_to_cover['theme_set'],
            set_question_en=set_question_to_cover.get('question_text_en', main_question_text_en),
            set_question_th=set_question_to_cover['question_text_th'],
            main_question_text_en=main_question_text_en,
            knowledge_graph_context=knowledge_graph_context or "No specific KG context available.",
            standard_document_excerpts=standard_document_excerpts or "No specific document excerpts available."
        )

        default_source_info = f"Attempted to generate sub-questions for SET benchmark: {set_question_to_cover['id']}. Contextual information was evaluated."
        try:
            response = await self.qg_llm.ainvoke(formatted_prompt)
            llm_output = response.content.strip()
            json_output = self._extract_json_from_llm_output(llm_output)

            if json_output and "rolled_up_sub_questions_text_en" in json_output:
                sub_q_text = json_output["rolled_up_sub_questions_text_en"]
                source_info = json_output.get("detailed_source_info_for_subquestions", default_source_info)
                return sub_q_text, source_info # Return even if it's the insufficiency message
            else:
                print(f"[QG_SERVICE SET_GAP_FILL_GEN] Failed to get valid JSON for SET ID: {set_question_to_cover['id']}. LLM output: {llm_output[:300]}...")
                return None, f"JSON parsing failed during generation for SET ID {set_question_to_cover['id']}."
        except Exception as e:
            print(f"[QG_SERVICE SET_GAP_FILL_GEN] Error generating sub-questions for SET ID: {set_question_to_cover['id']}: {e}")
            traceback.print_exc()
            return None, f"Exception during sub-question generation for SET ID {set_question_to_cover['id']}."

    async def _validate_generated_gap_fill_qs(
        self,
        set_question_to_cover: Dict[str, Any],
        generated_sub_questions: Optional[str],
        context_used_kg: Optional[str],
        context_used_chunks: Optional[str]
    ) -> Tuple[bool, str]: # (is_meaningful_and_grounded, justification)
        """
        Uses an LLM to validate the quality and relevance of generated gap-fill sub-questions.
        """
        if not generated_sub_questions or not generated_sub_questions.strip():
            return False, "No sub-questions were provided for validation."

        insufficiency_messages = [
            "no specific sub-questions can be generated",
            "insufficient context",
            "not contain any information directly or indirectly related",
            "context is entirely focused on", # Added from user log
            "is irrelevant to the set benchmark question" # Added from user log
        ]
        if any(msg.lower() in generated_sub_questions.lower() for msg in insufficiency_messages):
            return False, f"Generated text indicates insufficient context: '{generated_sub_questions}'"
        
        if len(generated_sub_questions.strip()) <= 10: # Arbitrary short length
            return False, "Generated sub-questions are too short to be meaningful."

        validation_prompt_str = """
        You are an ESG Quality Assurance Analyst. Your task is to critically evaluate a set of auto-generated sub-questions.
        These sub-questions were created to help answer a specific SET Benchmark Question, based ONLY on the "Available Supporting Context" from a company's documents.

        SET Benchmark Question:
        - ID: {set_id}
        - Dimension: {set_dimension}
        - Theme: {set_theme}
        - English Text: "{set_question_en}"

        Available Supporting Context (used for generating the sub-questions):
        Knowledge Graph Hints:
        {context_kg}
        Relevant Document Excerpt Hints:
        {context_chunks}

        Generated Sub-Questions to Evaluate:
        {generated_sub_questions}

        Evaluation Criteria (Answer strictly based on the provided information):
        1.  Grounded in Context: Are the "Generated Sub-Questions" CLEARLY and DIRECTLY answerable or derived from specific information present ONLY in the "Available Supporting Context"? They should NOT require external knowledge or make assumptions beyond this context.
        2.  Relevance to SET Question: Do the "Generated Sub-Questions" directly help in answering or elaborating on the specific "SET Benchmark Question"?
        3.  Quality & Specificity: Are the "Generated Sub-Questions" clear, specific, and actionable? Avoid overly generic questions, questions that are just rephrasing the SET question, or questions that are unanswerable from any reasonable company's perspective given typical ESG data.

        Overall Assessment:
        Based on ALL criteria above, are these "Generated Sub-Questions" of sufficiently high quality, well-grounded in the "Available Supporting Context", AND directly relevant to the "SET Benchmark Question" to be considered a "Meaningful and Valid" set?

        Output ONLY a single, valid JSON object with the following exact keys:
        - "is_meaningful_and_grounded": boolean (true if all criteria are met positively, otherwise false).
        - "justification": A brief string (1-2 sentences) explaining your boolean choice. If false, clearly state which criteria were not met (e.g., "Not grounded in context", "Too generic", "Not relevant to SET Q").
        
        Example for good: {{"is_meaningful_and_grounded": true, "justification": "Questions are specific, based on the provided context, and directly address the SET benchmark."}}
        Example for bad (not grounded): {{"is_meaningful_and_grounded": false, "justification": "Questions are too generic and not clearly supported by the limited KG/chunk context provided, which was about a different topic."}}
        Example for bad (not relevant to SET Q): {{"is_meaningful_and_grounded": false, "justification": "Questions are based on context but do not help answer the specific SET benchmark question about SIA/EIA monitoring."}}
        """
        print("--- DEBUG: _validate_generated_gap_fill_qs ---")
        print("Validation Prompt Template String:")
        print(validation_prompt_str) # Log template string

        prompt = PromptTemplate.from_template(validation_prompt_str)
        print(f"Inferred input variables for validation prompt: {prompt.input_variables}") # Log inferred variables
        
        try:
            formatted_prompt = prompt.format(
                set_id=set_question_to_cover['id'],
                set_dimension=set_question_to_cover['dimension'],
                set_theme=set_question_to_cover['theme_set'],
                set_question_en=set_question_to_cover.get('question_text_en', 'N/A'),
                context_kg=context_used_kg or "Not available.",
                context_chunks=context_used_chunks or "Not available.",
                generated_sub_questions=generated_sub_questions or "No sub-questions provided for evaluation." # Ensure it's not None
            )
        except KeyError as e:
            print(f"ERROR formatting validation prompt: {e}")
            print(f"kwargs sent to format for validation: {{'set_id': ..., 'generated_sub_questions': '{generated_sub_questions}' ...}}") # Log some kwargs
            raise  # Re-raise after logging

        try:
            # Use a different LLM or same with different settings if needed for QA
            response = await self.qg_llm.ainvoke(formatted_prompt) 
            llm_output = response.content.strip()
            validation_data = self._extract_json_from_llm_output(llm_output)

            if validation_data and isinstance(validation_data.get("is_meaningful_and_grounded"), bool):
                is_valid = validation_data["is_meaningful_and_grounded"]
                justification = validation_data.get("justification", "No justification provided.")
                print(f"[QG_SERVICE GAP_FILL_VALIDATE] SET ID {set_question_to_cover['id']} - Valid: {is_valid}, Justification: {justification}")
                return is_valid, justification
            else:
                error_msg = f"Failed to parse validation or missing 'is_meaningful_and_grounded' boolean. LLM output: {llm_output}"
                print(f"[QG_SERVICE GAP_FILL_VALIDATE] SET ID {set_question_to_cover['id']} - Error: {error_msg}")
                return False, error_msg
        except Exception as e:
            error_msg = f"Exception during gap-fill validation: {e}"
            print(f"[QG_SERVICE GAP_FILL_VALIDATE] SET ID {set_question_to_cover['id']} - Error: {error_msg}")
            traceback.print_exc()
            return False, error_msg

    async def _llm_suggest_alternative_search_terms(
        self,
        set_question_to_cover: Dict[str, Any],
        initial_kg_context: Optional[str],
        initial_chunk_context: Optional[str]
    ) -> Optional[List[str]]:
        # ... (ส่วนต้นของฟังก์ชัน) ...
        prompt_template_str = """
        You are an ESG Research Strategist. For the following SET Benchmark Question, the initial context retrieval was insufficient to generate detailed sub-questions.
        Your task is to suggest 2-3 alternative search keywords, key concepts, or entity types that might be present in ESG-related documents (like GRI standards, sustainability reports for an industrial packaging factory) and could help find more relevant information for this SET question.

        SET Benchmark Question:
        - Dimension: {set_dimension}
        - Theme: {set_theme}
        - English Text: "{set_question_en}"
        - Thai Text: "{set_question_th}"

        Initial Knowledge Graph Context Found (may be sparse or irrelevant):
        {initial_kg_context}

        Initial Document Excerpt Context Found (may be sparse or irrelevant):
        {initial_chunk_context}

        Based on the SET question and the (potentially lacking) initial context, what alternative search terms (keywords, specific ESG metrics, policy names, process types, relevant GRI disclosures, etc.) would you recommend to try and find more specific information within a company's ESG knowledge base (documents, graph data)?
        Focus on terms that would likely appear in text if the company addresses this SET topic.

        Output ONLY a single, valid JSON object with the key "alternative_search_terms", which should be a list of 2-3 suggested string terms.
        Example: {{"alternative_search_terms": ["Waste reduction targets", "Circular economy initiatives", "GRI 306-2 Management of waste-related impacts"]}}
        """
        print("--- DEBUG: _llm_suggest_alternative_search_terms ---")
        print("Prompt Template String:")
        print(prompt_template_str)
        
        prompt = PromptTemplate.from_template(prompt_template_str)
        print(f"Inferred input variables: {prompt.input_variables}")

        try:
            formatted_prompt = prompt.format(
                set_dimension=set_question_to_cover['dimension'],
                set_theme=set_question_to_cover['theme_set'],
                set_question_en=set_question_to_cover.get('question_text_en', ''),
                set_question_th=set_question_to_cover['question_text_th'],
                initial_kg_context=initial_kg_context or "No specific KG context initially found.",
                initial_chunk_context=initial_chunk_context or "No specific document excerpts initially found."
            )
        except KeyError as e:
            print(f"ERROR during prompt.format in _llm_suggest_alternative_search_terms: {e}")
            print(f"kwargs sent to format: {{'set_dimension': ..., 'set_theme': ..., ...}}") # แสดง kwargs ที่ส่งไป
            raise e
        
        try:
            response = await self.qg_llm.ainvoke(formatted_prompt) # Use the main qg_llm
            llm_output = response.content.strip()
            json_data = self._extract_json_from_llm_output(llm_output)
            if json_data and "alternative_search_terms" in json_data and isinstance(json_data["alternative_search_terms"], list):
                return [str(term) for term in json_data["alternative_search_terms"] if str(term).strip()]
            else:
                print(f"[QG_SERVICE LLM_SUGGEST_TERMS] Failed to get valid alternative terms. LLM output: {llm_output[:200]}")
                return None
        except Exception as e:
            print(f"[QG_SERVICE LLM_SUGGEST_TERMS] Error: {e}")
            return None
        
    async def _iterative_search_and_generate_for_set_gap(self, set_question: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Attempts to find information for a missing SET question in the KG and generate a new question set.
        This version uses a multi-step search strategy.
        """
        set_topic = f"{set_question['id']}: {set_question['title_th']}"
        self.logger.info(f"Attempting to fill gap for SET question: {set_topic}")

        # Step 1: Baseline Semantic Search
        relevant_chunks = self.neo4j_service.find_semantically_similar_chunks(
            query_text=set_topic,
            top_k=5
        )

        # Step 2: LLM-Powered Query Expansion if baseline search is insufficient
        if not relevant_chunks or len(relevant_chunks) < 2:
            self.logger.info(f"Baseline search for '{set_topic}' yielded few results. Expanding search...")
            expanded_keywords = await self._llm_generate_search_keywords(set_topic)
            if expanded_keywords:
                # Assuming neo4j_service has a method to search by multiple keywords
                # This could search both text properties and semantic similarity
                additional_chunks = self.neo4j_service.find_chunks_by_keywords(
                    keywords=expanded_keywords, 
                    top_k=5
                )
                # Combine and deduplicate chunks
                all_chunk_ids = {chunk['chunk_id'] for chunk in relevant_chunks}
                for chunk in additional_chunks:
                    if chunk['chunk_id'] not in all_chunk_ids:
                        relevant_chunks.append(chunk)
                        all_chunk_ids.add(chunk['chunk_id'])

        if not relevant_chunks:
            self.logger.warning(f"Could not find any relevant information for '{set_topic}' after expanded search.")
            return None

        # Proceed with question generation if relevant chunks are found
        self.logger.info(f"Found {len(relevant_chunks)} relevant chunks for '{set_topic}'. Generating new question.")
        context_str = "\n\n".join([chunk['text'] for chunk in relevant_chunks])
        
        # This part re-uses the logic from `generate_question_for_theme_level`
        # to create a question from the found context.
        # This might need to be refactored into a shared helper function.
        prompt = f"""
        Based on the following context from a company's documents, create a comprehensive question set about the ESG topic: "{set_topic}".
        The main question should be a high-level strategic question.
        The sub-questions should be specific, answerable questions based on the provided text.

        Context:
        ---
        {context_str}
        ---

        Return the result as a JSON object with keys "main_question" and "sub_questions" (a list of strings).
        """
        try:
            await self.rate_limiter.wait()
            response = await self.llm.acomplete(prompt)
            new_question_data = json.loads(response.text)
            
            main_q_text = new_question_data.get("main_question", "")
            sub_q_list = new_question_data.get("sub_questions", [])
            sub_q_text_en = "\n".join(f"{i+1}. {q}" for i, q in enumerate(sub_q_list))

            candidate_question = {
                "theme": f"SET Coverage: {set_question['id']}",
                "category": set_question['dimension'],
                "keywords": set_question['theme_set'],
                "theme_description_en": f"Questions generated to cover SET benchmark ID: {set_question['id']}",
                "main_question_text_en": main_q_text,
                "sub_questions_sets": [
                    SubQuestionDetail(
                        sub_question_text_en=sub_q_text_en,
                        sub_theme_name=f"Details for SET {set_question['id']}",
                        category_dimension=set_question['dimension']
                    )
                ],
                "related_set_questions": [
                    RelatedSETQuestion(
                        set_id=set_question['id'],
                        title_th=set_question['title_th']
                    )
                ]
            }
            return candidate_question
        except Exception as e:
            self.logger.error(f"Error generating question for gap '{set_topic}': {e}")
            return None

    def _add_existing_theme_to_api_response(self, existing_doc: ESGQuestion, api_response_list: List[GeneratedQuestion]):
        """Adds questions from an existing active DB document to the API response list if not already present by theme."""
        if any(q.theme == existing_doc.theme for q in api_response_list):
            return # Already added or processed

        api_response_list.append(GeneratedQuestion(
            question_text_en=existing_doc.main_question_text_en,
            question_text_th=existing_doc.main_question_text_th,
            category=existing_doc.category,
            theme=existing_doc.theme,
            is_main_question=True
        ))
        if existing_doc.sub_questions_sets:
            for sq_set_model in existing_doc.sub_questions_sets:
                api_response_list.append(GeneratedQuestion(
                    question_text_en=sq_set_model.sub_question_text_en,
                    question_text_th=sq_set_model.sub_question_text_th,
                    category=sq_set_model.category_dimension,
                    theme=existing_doc.theme, # Belongs to the same Main Category
                    sub_theme_name=sq_set_model.sub_theme_name,
                    is_main_question=False,
                    additional_info={"detailed_source_info_for_subquestions": sq_set_model.detailed_source_info}
                ))