import asyncio
import os
import io
import re
import tempfile
from typing import List, Dict, Any, Optional, Union, Tuple, Set
from dotenv import load_dotenv
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

load_dotenv()

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
    def __init__(self,
                 neo4j_service: Neo4jService,
                 similarity_embedding_model: Embeddings
                ):
        self.neo4j_service = neo4j_service
        self.similarity_llm_embedding = similarity_embedding_model # Used for are_questions_substantially_similar
        self.qg_llm = ChatGoogleGenerativeAI( # LLM for theme naming, question generation
            model=os.getenv("QUESTION_GENERATION_MODEL", "gemini-2.5-flash-preview-05-20"), # Updated model name
            temperature=0.8, # Slightly lower for more deterministic theme naming/question gen
            max_retries=3,
        )
        self.translation_llm = ChatGoogleGenerativeAI(
            model=os.getenv("TRANSLATION_MODEL", "gemini-2.5-flash-preview-05-20"), # Updated model name
            temperature=0.8,
            max_retries=3,
        )

        # self._load_graph_schema() # Remove this call or its usage for themes

        if self.similarity_llm_embedding:
            print("[QG_SERVICE LOG] Similarity embedding model received and configured.")
        else:
            print("[QG_SERVICE WARNING] Similarity embedding model was NOT provided. Similarity checks might fallback or be less accurate.")

    # _load_graph_schema method can be removed if no longer used elsewhere.

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

    async def _get_entity_graph_data(self) -> Optional[Dict[str, Any]]:
        # ... (Method from your latest code, ensure query_nodes and query_edges are correct) ...
        print("[QG_SERVICE LOG] Fetching __Entity__ graph data from Neo4j...")
        try:
            query_nodes = """
            MATCH (e:__Entity__)
            WHERE e.id IS NOT NULL
            OPTIONAL MATCH (sc:StandardChunk)-[:CONTAINS_ENTITY]->(e) // <<< ส่วนนี้มีปัญหา
            RETURN
                e.id AS id,
                COALESCE(e.description, '') AS description,
                [lbl IN labels(e) WHERE lbl <> '__Entity__'] AS specific_labels,
                COALESCE(sc.doc_id, '') AS source_document_doc_id // <<< ทำให้ได้ค่าว่าง
            """
            query_edges = """
            MATCH (e1:__Entity__)-[r]-(e2:__Entity__)
            WHERE e1.id IS NOT NULL AND e2.id IS NOT NULL AND e1.id <> e2.id
            RETURN DISTINCT e1.id AS source, e2.id AS target
            """
            loop = asyncio.get_running_loop()
            nodes_result = await loop.run_in_executor(None, self.neo4j_service.graph.query, query_nodes)
            edges_result = await loop.run_in_executor(None, self.neo4j_service.graph.query, query_edges)

            if not nodes_result: print("[QG_SERVICE WARNING] No __Entity__ nodes found."); return None
            nodes_map: Dict[str, Dict[str, Any]] = {}
            for record in nodes_result:
                if record and record.get('id'):
                    node_id = record['id']
                    if node_id in nodes_map:
                        if not nodes_map[node_id]['source_document_doc_id'] and record['source_document_doc_id']:
                            nodes_map[node_id]['source_document_doc_id'] = record['source_document_doc_id']
                    else:
                        nodes_map[node_id] = {
                            'id': node_id,
                            'description': record['description'],
                            'labels': record['specific_labels'] if record['specific_labels'] else [DEFAULT_NODE_TYPE],
                            'source_document_doc_id': record['source_document_doc_id']
                        }
            edges: List[Tuple[str, str]] = []
            if edges_result:
                for record in edges_result:
                    source, target = record.get('source'), record.get('target')
                    if source and target and source in nodes_map and target in nodes_map:
                        edges.append((source, target))
            print(f"[QG_SERVICE INFO] Fetched {len(nodes_map)} __Entity__ nodes and {len(edges)} edges (before unique).")
            if not nodes_map: print("[QG_SERVICE WARNING] No valid __Entity__ nodes after processing."); return None
            return {"nodes_map": nodes_map, "edges": list(set(tuple(sorted(edge)) for edge in edges))}
        except Exception as e:
            print(f"[QG_SERVICE ERROR] Error fetching __Entity__ graph data: {e}"); traceback.print_exc(); return None

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
        min_first_order_community_size: int = 2,
        min_main_category_fo_community_count: int = 1, 
    ) -> List[Dict[str, Any]]:
        print("[QG_SERVICE LOG] Starting Hierarchical Theme Identification from KG...")
        entity_graph_data = await self._get_entity_graph_data()
        if not entity_graph_data or not entity_graph_data.get("nodes_map"):
            print("[QG_SERVICE WARNING] Could not retrieve entity graph data. Aborting."); return []
        all_entity_data_map = entity_graph_data["nodes_map"]

        first_order_communities_map = await self._detect_first_order_communities(entity_graph_data, min_community_size=min_first_order_community_size)
        if not first_order_communities_map:
            print("[QG_SERVICE WARNING] No first-order communities found. Aborting."); return []

        consolidated_themes_tasks = []
        for fo_comm_id, entity_ids_in_fo_comm in first_order_communities_map.items():
            context_for_llm_ct, constituent_ids_ct, source_docs_ct = self._prepare_context_for_consolidated_theme(fo_comm_id, entity_ids_in_fo_comm, all_entity_data_map)
            if not context_for_llm_ct.strip(): continue
            prompt_template_str_ct = """
            You are an expert ESG analyst. A group of semantically related entities and concepts has been identified from an ESG knowledge graph. 
            Your task is to define a consolidated ESG reporting theme based on these entities, specifically for the **industrial packaging sector**.
            The following are representative entities/concepts and their descriptions from this group:
            --- REPRESENTATIVE ENTITIES/CONCEPTS START ---
            {representative_entity_content}
            --- REPRESENTATIVE ENTITIES/CONCEPTS END ---
            These entities and concepts primarily originate from or relate to standards document(s) such as: {source_standards_list_str}
            Based on this information:
            1.  Propose a concise and descriptive **Theme Name** (in English) for this group of entities/concepts.
            2.  Provide a brief **Theme Description** (in English, 1-2 sentences).
            3.  Determine the most relevant primary **ESG Dimension** ('E', 'S', or 'G').
            4.  Suggest 3-5 relevant and specific **Keywords** (in English, comma-separated).
            Output ONLY a single, valid JSON object with the following exact keys: "theme_name_en", "description_en", "dimension", "keywords_en".
            """
            prompt_ct = PromptTemplate.from_template(prompt_template_str_ct)
            formatted_prompt_ct = prompt_ct.format(
                representative_entity_content=context_for_llm_ct,
                source_standards_list_str=", ".join(sorted(list(source_docs_ct))[:3]) if source_docs_ct else "various ESG standards"
            )
            consolidated_themes_tasks.append(self._process_first_order_community_for_consolidated_theme(formatted_prompt_ct, constituent_ids_ct, sorted(list(source_docs_ct)), fo_comm_id))
        
        all_consolidated_themes: List[Dict[str, Any]] = []
        if consolidated_themes_tasks:
            results = await asyncio.gather(*consolidated_themes_tasks, return_exceptions=True)
            for res in results:
                if isinstance(res, dict) and res.get("theme_name"): all_consolidated_themes.append(res)
        
        if not all_consolidated_themes:
            print("[QG_SERVICE WARNING] No consolidated themes (sub-themes) generated."); return []
        print(f"[QG_SERVICE INFO] Generated {len(all_consolidated_themes)} consolidated themes (sub-themes).")

        meta_graph = await self._create_community_meta_graph(first_order_communities_map, entity_graph_data, min_inter_community_edges_for_link=1)
        
        main_category_groupings = await self._detect_main_categories_from_meta_graph(
            meta_graph, 
            min_main_category_fo_community_count=min_main_category_fo_community_count # *** CORRECTED HERE ***
        )
        
        if not main_category_groupings:
            print("[QG_SERVICE WARNING] No main category groupings from meta-graph. Fallback strategy.")
            if 0 < len(all_consolidated_themes) <= 25:
                 hierarchical_output_fallback = await self.cluster_consolidated_themes_with_llm(all_consolidated_themes, num_main_categories_target=max(1, int(len(all_consolidated_themes) / 4) + 1))
                 if hierarchical_output_fallback and hierarchical_output_fallback.get("main_theme_categories"):
                     print("[QG_SERVICE INFO] Fallback to LLM clustering of consolidated themes succeeded.")
                     return [
                         {
                             **mc_data, 
                             "_constituent_entity_ids_in_mc": list(set(eid for tn in mc_data.get("consolidated_theme_names_in_category", []) for ct in all_consolidated_themes if ct.get("theme_name") == tn for eid in ct.get("constituent_entity_ids",[]))),
                             "_source_document_ids_for_mc": list(set(sid for tn in mc_data.get("consolidated_theme_names_in_category", []) for ct in all_consolidated_themes if ct.get("theme_name") == tn for sid in ct.get("source_standard_document_ids",[]))),
                             "consolidated_themes": [ct for tn in mc_data.get("consolidated_theme_names_in_category", []) for ct in all_consolidated_themes if ct.get("theme_name") == tn]
                         } for mc_data in hierarchical_output_fallback.get("main_theme_categories",[])
                     ]
            print("[QG_SERVICE WARNING] Fallback to LLM clustering failed or too many themes. Structuring flat themes as main categories.")
            return [self._structure_flat_theme_as_main_category(ct, idx) for idx, ct in enumerate(all_consolidated_themes)]

        main_categories_final: List[Dict[str, Any]] = []
        llm_main_category_tasks = []
        original_fo_comm_id_to_consolidated_theme: Dict[int, Dict[str, Any]] = {}
        for ct_theme in all_consolidated_themes:
            original_id = ct_theme.get("_first_order_community_id_source")
            if isinstance(original_id, int): original_fo_comm_id_to_consolidated_theme[original_id] = ct_theme

        for mc_raw_id, original_fo_community_ids_in_mc_group in main_category_groupings.items():
            consolidated_themes_for_this_mc: List[Dict[str, Any]] = []
            current_mc_entities: Set[str] = set(); current_mc_source_docs: Set[str] = set()
            for fo_comm_id in original_fo_community_ids_in_mc_group:
                consolidated_theme = original_fo_comm_id_to_consolidated_theme.get(fo_comm_id)
                if consolidated_theme:
                    consolidated_themes_for_this_mc.append(consolidated_theme)
                    current_mc_entities.update(consolidated_theme.get("constituent_entity_ids", []))
                    current_mc_source_docs.update(consolidated_theme.get("source_standard_document_ids", []))
            if not consolidated_themes_for_this_mc: continue
            llm_main_category_tasks.append(self._generate_main_category_details_with_llm(mc_raw_id, consolidated_themes_for_this_mc, current_mc_source_docs, current_mc_entities))
        
        if llm_main_category_tasks:
            main_category_llm_results = await asyncio.gather(*llm_main_category_tasks, return_exceptions=True)
            for mc_result_item in main_category_llm_results:
                if isinstance(mc_result_item, dict) and mc_result_item.get("main_category_name_en"):
                    main_categories_final.append(mc_result_item)
        print(f"[QG_SERVICE LOG] Identified {len(main_categories_final)} hierarchical main categories after LLM processing.")
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
                node_desc = entity_data.get('description', 'No description.')
                if not node_desc.strip(): node_desc = "No substantive description."
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
                LIMIT 2
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
        You are an expert ESG consultant for an **industrial packaging factory**.
        For the Main ESG Category named "{main_category_name}" (Description: {main_category_description}; Keywords: {main_category_keywords}):

        For your general awareness, core concepts and document excerpts potentially related to this category include:
        Knowledge Graph Hints: {concise_graph_context_summary}
        Relevant Document Excerpt Hint: {concise_chunk_context_summary}

        Based on the Main Category's name, description, keywords, and these general hints, formulate ONE very broad, open-ended Main Question.
        This single question should:
        - Act as an umbrella for the entire Main Category.
        - Capture its core essence and strategic importance.
        - Be a high-level strategic inquiry suitable for a foundational overview.
        - Be answerable with "Yes/No" and a brief explanation, or a general policy/approach statement (e.g., "Does the company have a comprehensive strategy for X?").
        - AVOID asking for multiple distinct pieces of information, detailed lists, or quantitative data in this Main Question.
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
            LIMIT 15
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

        Task: Based on the Main Question and ALL provided context (sub-theme overview, knowledge graph, and standard documents), formulate a set of 3-5 specific, actionable, and data-driven Sub-Questions.
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

        Output ONLY a single, valid JSON object with these exact keys:
        - "rolled_up_sub_questions_text_en": A string containing ONLY 3-5 sub-questions, each numbered and on a new line.
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
    
    async def evolve_and_store_questions(self, uploaded_file_content_bytes: Optional[bytes] = None) -> List[GeneratedQuestion]: # API Response
        print(f"[QG_SERVICE LOG] Orchestrating Hierarchical Question (Denormalized Roll-up) Evolution. File uploaded: {'Yes' if uploaded_file_content_bytes else 'No'}")
        current_time_utc = datetime.now(timezone.utc)
        
        # --- Scope Determination ---
        # ... (Your existing scope determination logic - ensure it's robust) ...
        process_scope = PROCESS_SCOPE_KG_UPDATED_FULL_SCAN_FROM_CONTENT # Example
        initial_neo4j_empty_check_result = await self.neo4j_service.is_graph_substantially_empty()
        active_questions_in_db_count = await ESGQuestion.find(ESGQuestion.is_active == True).count()
        if initial_neo4j_empty_check_result:
            process_scope = PROCESS_SCOPE_KG_INITIAL_FULL_SCAN_FROM_CONTENT
            if active_questions_in_db_count > 0:
                await ESGQuestion.find(ESGQuestion.is_active == True).update({"$set": {"is_active": False, "updated_at": current_time_utc}})
        elif not uploaded_file_content_bytes and active_questions_in_db_count == 0 and not initial_neo4j_empty_check_result:
            process_scope = PROCESS_SCOPE_KG_FULL_SCAN_DB_EMPTY_FROM_CONTENT
        elif not uploaded_file_content_bytes and active_questions_in_db_count > 0 :
             process_scope = PROCESS_SCOPE_KG_SCHEDULED_REFRESH_FROM_CONTENT
        print(f"[QG_SERVICE INFO] Determined process_scope: {process_scope}")
        # --- End Scope Determination ---

        hierarchical_themes_structure: List[Dict[str, Any]] = await self.identify_hierarchical_themes_from_kg()
        if not hierarchical_themes_structure:
            print("[QG_SERVICE WARNING] No hierarchical theme structure identified. Aborting question evolution.")
            # ... (Fallback logic) ...
            return []

        all_db_main_category_docs: List[ESGQuestion] = await ESGQuestion.find_all().to_list()
        all_historical_main_categories_map: Dict[str, List[ESGQuestion]] = {
            doc.theme: [d for d in all_db_main_category_docs if d.theme == doc.theme]
            for doc in all_db_main_category_docs
        }
        for theme_key in all_historical_main_categories_map:
            all_historical_main_categories_map[theme_key].sort(key=lambda q: q.version, reverse=True)
        print(f"[QG_SERVICE INFO] Fetched {len(all_db_main_category_docs)} historical Main Category docs, grouped into {len(all_historical_main_categories_map)} unique Main Categories.")

        question_set_generation_tasks = []
        all_current_run_main_category_names_generated: Set[str] = set()

        for main_category_info in hierarchical_themes_structure:
            main_category_name = main_category_info.get("main_category_name_en")
            if not main_category_name: continue
            all_current_run_main_category_names_generated.add(main_category_name)
            
            question_set_generation_tasks.append(
                self.generate_question_for_theme_level(theme_info=main_category_info) # No is_main_category needed now
            )
        
        generated_question_sets_with_exc: List[Union[GeneratedQuestionSet, Exception, None]] = []
        if question_set_generation_tasks:
            generated_question_sets_with_exc = await asyncio.gather(*question_set_generation_tasks, return_exceptions=True)

        # --- DB Operations for Main Category Documents (Denormalized) ---
        api_response_questions: List[GeneratedQuestion] = [] # For API response

        for i, q_set_result_item in enumerate(generated_question_sets_with_exc):
            original_main_category_info = hierarchical_themes_structure[i] # Assuming order is maintained

            if isinstance(q_set_result_item, GeneratedQuestionSet):
                generated_q_set = q_set_result_item # This is a GeneratedQuestionSet Pydantic object
                
                main_cat_name_from_q_set = generated_q_set.main_category_name # Should match original_main_category_info's name
                
                # Prepare data for ESGQuestion (Denormalized MongoDB Model)
                main_question_text_th = await self.translate_text_to_thai(
                    generated_q_set.main_question_text_en,
                    category_name=main_cat_name_from_q_set,
                    keywords=original_main_category_info.get("main_category_keywords_en", "")
                )
                sub_questions_text_th = await self.translate_text_to_thai(generated_q_set.rolled_up_sub_questions_text_en, 
                    category_name=main_cat_name_from_q_set, 
                    keywords=original_main_category_info.get("main_category_keywords_en", "")
                )

                # Create the SubQuestionDetail for the rolled-up sub-questions
                rolled_up_sub_q_detail = SubQuestionDetail(
                    sub_question_text_en=generated_q_set.rolled_up_sub_questions_text_en,
                    sub_question_text_th=sub_questions_text_th,
                    sub_theme_name=f"Overall Sub-Questions for {main_cat_name_from_q_set}", # Generic name
                    category_dimension=generated_q_set.main_category_dimension, # Dimension of the main category
                    # Keywords/Desc for sub_questions_sets can be derived from main_category_info or be generic
                    keywords=original_main_category_info.get("main_category_keywords_en"), 
                    theme_description_en=f"Detailed inquiries related to {main_cat_name_from_q_set}.",
                    constituent_entity_ids=original_main_category_info.get("_constituent_entity_ids_in_mc", []),
                    source_document_references=original_main_category_info.get("_source_document_ids_for_mc", []),
                    detailed_source_info=generated_q_set.detailed_source_info_for_subquestions
                )
                if rolled_up_sub_q_detail.theme_description_en:
                    rolled_up_sub_q_detail.theme_description_th = await self.translate_text_to_thai(rolled_up_sub_q_detail.theme_description_en, category_name=main_cat_name_from_q_set, keywords=original_main_category_info.get("main_category_keywords_en", ""))

                db_main_category_doc_data = {
                    "theme": main_cat_name_from_q_set,
                    "category": generated_q_set.main_category_dimension,
                    "main_question_text_en": generated_q_set.main_question_text_en,
                    "main_question_text_th": main_question_text_th,
                    "keywords": original_main_category_info.get("main_category_keywords_en", ""),
                    "theme_description_en": original_main_category_info.get("main_category_description_en", ""),
                    "theme_description_th": await self.translate_text_to_thai(original_main_category_info.get("main_category_description_en", "")),
                    "sub_questions_sets": [rolled_up_sub_q_detail.model_dump(exclude_none=True)] if rolled_up_sub_q_detail.sub_question_text_en.strip() and "No specific sub-questions" not in rolled_up_sub_q_detail.sub_question_text_en else [], # Only add if not empty
                    "main_category_constituent_entity_ids": original_main_category_info.get("_constituent_entity_ids_in_mc", []),
                    "main_category_source_document_references": original_main_category_info.get("_source_document_ids_for_mc", []),
                    "generation_method": original_main_category_info.get("generation_method", "kg_hierarchical_rollup"),
                    "metadata_extras": {"_main_category_raw_id": original_main_category_info.get("_main_category_raw_id")},
                    "updated_at": current_time_utc, "is_active": True,
                }
                
                theme_desc_en_for_translation = original_main_category_info.get("main_category_description_en", "")
                theme_desc_th_for_translation = None
                if theme_desc_en_for_translation:
                    theme_desc_th_for_translation = await self.translate_text_to_thai(
                        theme_desc_en_for_translation,
                        category_name=main_cat_name_from_q_set,
                        keywords=original_main_category_info.get("main_category_keywords_en") # Get keywords from original_main_category_info
                    )
                
                db_main_category_doc_data["theme_description_th"] = theme_desc_th_for_translation

                await self._upsert_main_question_document_to_db(
                    db_main_category_doc_data, 
                    all_historical_main_categories_map, 
                    current_time_utc
                )
                
                # Populate API response (flat list of GeneratedQuestion)
                api_response_questions.append(GeneratedQuestion(
                    question_text_en=generated_q_set.main_question_text_en,
                    question_text_th=main_question_text_th,
                    category=generated_q_set.main_category_dimension,
                    theme=main_cat_name_from_q_set,
                    is_main_question=True,
                    additional_info=None # Main question usually doesn't have detailed source here
                ))
                if rolled_up_sub_q_detail.sub_question_text_en.strip() and "No specific sub-questions" not in rolled_up_sub_q_detail.sub_question_text_en:
                    api_response_questions.append(GeneratedQuestion(
                        question_text_en=rolled_up_sub_q_detail.sub_question_text_en,
                        question_text_th=rolled_up_sub_q_detail.sub_question_text_th,
                        category=rolled_up_sub_q_detail.category_dimension,
                        theme=main_cat_name_from_q_set, # Belongs to the same Main Category
                        sub_theme_name=rolled_up_sub_q_detail.sub_theme_name, # "Overall Sub-Questions..."
                        is_main_question=False,
                        additional_info={"detailed_source_info_for_subquestions": rolled_up_sub_q_detail.detailed_source_info}
                    ))

            elif isinstance(q_set_result_item, Exception):
                mc_name_err = original_main_category_info.get("main_category_name_en", f"Unknown MainCat (index {i})")
                print(f"[QG_SERVICE ERROR] Question set generation for MainCat '{mc_name_err}' failed: {q_set_result_item}")
            elif q_set_result_item is None:
                mc_name_err = original_main_category_info.get("main_category_name_en", f"Unknown MainCat (index {i})")
                print(f"[QG_SERVICE WARNING] Question set generation for MainCat '{mc_name_err}' returned None.")
        
        # --- Deactivation Logic for Main Categories (เหมือนเดิม) ---
        if process_scope in [PROCESS_SCOPE_KG_INITIAL_FULL_SCAN_FROM_CONTENT, PROCESS_SCOPE_KG_UPDATED_FULL_SCAN_FROM_CONTENT, PROCESS_SCOPE_KG_FULL_SCAN_DB_EMPTY_FROM_CONTENT, PROCESS_SCOPE_KG_SCHEDULED_REFRESH_FROM_CONTENT]:
            # ... (Deactivation logic using all_current_run_main_category_names_generated) ...
            current_active_main_cat_docs_in_db: List[ESGQuestion] = await ESGQuestion.find(ESGQuestion.is_active == True).to_list()
            active_main_cat_names_in_db_for_deact: Set[str] = {q.theme for q in current_active_main_cat_docs_in_db} # Assuming theme is main_category_name
            for active_mc_name_in_db in active_main_cat_names_in_db_for_deact:
                if active_mc_name_in_db not in all_current_run_main_category_names_generated:
                    print(f"[QG_SERVICE INFO] Main Category '{active_mc_name_in_db}' no longer identified. Deactivating.")
                    await ESGQuestion.find(ESGQuestion.theme == active_mc_name_in_db, ESGQuestion.is_active == True).update(
                        {"$set": {"is_active": False, "updated_at": current_time_utc}}
                    )

        print(f"[QG_SERVICE LOG] Hierarchical evolution (denormalized roll-up) complete. API response contains {len(api_response_questions)} question entries.")
        return api_response_questions


    async def _upsert_main_question_document_to_db( 
        self, 
        main_category_db_data: Dict[str, Any], 
        all_historical_main_categories_map: Dict[str, List[ESGQuestion]], 
        current_time_utc: datetime
    ):
        # ... (เหมือนโค้ดที่คุณให้มาล่าสุด, ตรวจสอบ content_changed โดยเทียบ main_question_text_en และ sub_questions_sets) ...
        main_category_name = main_category_db_data["theme"]
        historical_versions = all_historical_main_categories_map.get(main_category_name, [])
        latest_db_version_doc: Optional[ESGQuestion] = historical_versions[0] if historical_versions else None
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
            # print(f"[QG_SERVICE INFO DB_UPSERT] Main Category '{main_category_name}' content SIMILAR. Checking metadata/status.")
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
                # print(f"[QG_SERVICE INFO DB_UPSERT] Updated metadata/status for MC '{main_category_name}'.")
                updated_doc = await ESGQuestion.get(latest_db_version_doc.id)
                if updated_doc: # Update local cache
                    idx = next((i for i,q in enumerate(all_historical_main_categories_map.get(main_category_name,[])) if q.id == updated_doc.id),-1)
                    if idx != -1: all_historical_main_categories_map[main_category_name][idx] = updated_doc
                    else: all_historical_main_categories_map.setdefault(main_category_name, []).insert(0, updated_doc); all_historical_main_categories_map[main_category_name].sort(key=lambda q:q.version, reverse=True)
        else:
            if latest_db_version_doc:
                print(f"[QG_SERVICE INFO DB_UPSERT] MC '{main_category_name}' content CHANGED. New version.")
                if latest_db_version_doc.is_active: await ESGQuestion.find_one(ESGQuestion.id == latest_db_version_doc.id).update({"$set": {"is_active": False, "updated_at": current_time_utc}})
                upsert_data_for_document["version"] = latest_db_version_doc.version + 1
            else:
                print(f"[QG_SERVICE INFO DB_UPSERT] MC '{main_category_name}' is NEW. Inserting v1.")
                upsert_data_for_document["version"] = 1
            upsert_data_for_document["generated_at"] = current_time_utc; upsert_data_for_document["updated_at"] = current_time_utc; upsert_data_for_document["is_active"] = True
            new_doc = ESGQuestion(**upsert_data_for_document)
            try:
                inserted = await new_doc.insert()
                all_historical_main_categories_map.setdefault(main_category_name, []).insert(0, inserted)
                all_historical_main_categories_map[main_category_name].sort(key=lambda q: q.version, reverse=True)
            except Exception as e: print(f"[QG_SERVICE ERROR DB_UPSERT] Insert MC '{main_category_name}': {e}"); print(f"Data: {upsert_data_for_document}"); traceback.print_exc()

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


