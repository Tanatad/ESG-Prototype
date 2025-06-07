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
            temperature=0.4, 
            top_p=0.9,
            top_k=40,
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
        min_first_order_community_size: int = 3,
        min_main_category_fo_community_count: int = 2, 
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
    
    async def evolve_and_store_questions(self, uploaded_file_content_bytes: Optional[bytes] = None) -> List[GeneratedQuestion]:
        print(f"[QG_SERVICE LOG] Orchestrating Hierarchical Question Evolution with Agentic Behavior. File uploaded: {'Yes' if uploaded_file_content_bytes else 'No'}")
        current_time_utc = datetime.now(timezone.utc)

        # --- Scope Determination (existing logic) ---
        process_scope = PROCESS_SCOPE_KG_UPDATED_FULL_SCAN_FROM_CONTENT # Default
        initial_neo4j_empty_check_result = await self.neo4j_service.is_graph_substantially_empty()
        active_questions_in_db_count = await ESGQuestion.find(ESGQuestion.is_active == True).count()

        if initial_neo4j_empty_check_result:
            process_scope = PROCESS_SCOPE_KG_INITIAL_FULL_SCAN_FROM_CONTENT
            if active_questions_in_db_count > 0:
                print("[QG_SERVICE SCOPE] Initial KG scan, deactivating all existing questions.")
                await ESGQuestion.find(ESGQuestion.is_active == True).update({"$set": {"is_active": False, "updated_at": current_time_utc}})
        elif not uploaded_file_content_bytes and active_questions_in_db_count == 0 and not initial_neo4j_empty_check_result:
            process_scope = PROCESS_SCOPE_KG_FULL_SCAN_DB_EMPTY_FROM_CONTENT
        elif not uploaded_file_content_bytes and active_questions_in_db_count > 0:
            process_scope = PROCESS_SCOPE_KG_SCHEDULED_REFRESH_FROM_CONTENT
        print(f"[QG_SERVICE INFO] Determined process_scope: {process_scope}")
        # --- End Scope Determination ---

        # 1. Load SET Benchmark Questions
        benchmark_set_questions = self.load_set_benchmark_questions()
        if not benchmark_set_questions:
            print("[QG_SERVICE WARNING] SET Benchmark questions could not be loaded. Validation against SET will be skipped.")

        # 2. Identify Themes from KG
        hierarchical_themes_structure: List[Dict[str, Any]] = await self.identify_hierarchical_themes_from_kg()
        if not hierarchical_themes_structure:
            print("[QG_SERVICE WARNING] No hierarchical theme structure identified from KG.")
            # Fallback: If no themes from KG, directly try to cover SET questions if benchmarks are available
            if benchmark_set_questions:
                 pass # This path will be handled by the gap-filling loop later
            else:
                print("[QG_SERVICE WARNING] No KG themes and no benchmarks. Aborting question evolution.")
                return []


        # Load historical questions from DB
        all_db_main_category_docs: List[ESGQuestion] = await ESGQuestion.find_all().to_list()
        all_historical_main_categories_map: Dict[str, List[ESGQuestion]] = {
            doc.theme: [d for d in all_db_main_category_docs if d.theme == doc.theme]
            for doc in all_db_main_category_docs
        }
        for theme_key in all_historical_main_categories_map:
            all_historical_main_categories_map[theme_key].sort(key=lambda q: q.version, reverse=True)
        
        active_theme_names_at_start_of_run: Set[str] = {
            theme_name for theme_name, docs in all_historical_main_categories_map.items() if docs and docs[0].is_active
        }

        api_response_questions: List[GeneratedQuestion] = []
        # Tracks SET Question IDs covered by any theme (KG-derived or gap-filled) in this run
        successfully_covered_set_question_ids_this_run: Set[str] = set()
        # Tracks theme names that are successfully processed and should be active at the end of this run
        themes_to_be_active_this_run: Set[str] = set()


        # --- AGENTIC LOOP FOR KG-DERIVED THEMES ---
        print(f"[QG_SERVICE AGENT LOOP] Processing {len(hierarchical_themes_structure)} themes derived from KG.")
        for main_category_info_from_kg in hierarchical_themes_structure:
            mc_name_kg = main_category_info_from_kg.get("main_category_name_en")
            if not mc_name_kg:
                print("[QG_SERVICE WARNING] KG theme info missing 'main_category_name_en'. Skipping.")
                continue

            # Generate initial question set for this KG theme
            current_q_set_for_kg_theme: Optional[GeneratedQuestionSet] = \
                await self.generate_question_for_theme_level(theme_info=main_category_info_from_kg)

            if not current_q_set_for_kg_theme:
                print(f"[QG_SERVICE WARNING] Initial QSet generation failed for KG theme '{mc_name_kg}'. Skipping this theme.")
                continue
            
            # These will be updated in the refinement loop
            current_main_q_text = current_q_set_for_kg_theme.main_question_text_en
            current_sub_q_text = current_q_set_for_kg_theme.rolled_up_sub_questions_text_en
            current_sub_q_source_info = current_q_set_for_kg_theme.detailed_source_info_for_subquestions

            max_refinement_iterations = int(os.getenv("QG_MAX_REFINEMENT_ITERATIONS", "1")) # e.g., 0 for no refinement, 1 for one try
            current_iteration = 0
            
            validation_status_for_this_kg_theme = "Pending_Validation"
            all_feedback_for_this_kg_theme: List[Dict[str, Any]] = []
            set_ids_covered_by_this_kg_theme: Set[str] = set()

            relevant_set_benchmarks_for_kg_theme = []
            if benchmark_set_questions:
                relevant_set_benchmarks_for_kg_theme = await self._find_relevant_set_benchmark_questions(
                    mc_dimension=current_q_set_for_kg_theme.main_category_dimension,
                    mc_name=mc_name_kg, 
                    mc_keywords=current_q_set_for_kg_theme.main_category_keywords,
                    generated_main_q_text=current_main_q_text,
                    generated_sub_q_text=current_sub_q_text,
                    all_set_benchmarks=benchmark_set_questions,
                    relevance_threshold=0.65
                )

            is_fully_validated_against_relevant_set = False
            if not relevant_set_benchmarks_for_kg_theme: # No relevant SET Qs for this KG theme
                is_fully_validated_against_relevant_set = True 
                validation_status_for_this_kg_theme = "Validated_Extended_Coverage" # Covers beyond SET or SET not applicable
            
            while current_iteration < max_refinement_iterations and not is_fully_validated_against_relevant_set:
                print(f"[QG_SERVICE AGENT LOOP - KG THEME] '{mc_name_kg}', Iteration {current_iteration + 1}/{max_refinement_iterations}")
                
                # Assume all relevant benchmarks will be covered in this iteration pass
                all_benchmarks_covered_this_iteration_pass = True 
                
                # Fetch fresh context for evaluation and potential refinement
                # Context can be more targeted here if needed based on previous feedback (more advanced)
                refine_kg_ctx = await self.neo4j_service.get_graph_context_for_theme_chunks_v2(
                    theme_name=mc_name_kg, theme_keywords=current_q_set_for_kg_theme.main_category_keywords or "",
                    max_central_entities=2, max_hops_for_central_entity=1, max_relations_to_collect_per_central_entity=3,
                    max_total_context_items_str_len=700000 
                ) or "No specific KG context for refinement."
                refine_chunk_ctx = "No specific document excerpts for refinement." # Simplified
                # (Add logic to fetch relevant chunk context if desired, similar to generate_question_for_theme_level)

                temp_set_ids_covered_this_iteration_pass: Set[str] = set()

                for benchmark_q in relevant_set_benchmarks_for_kg_theme:
                    if benchmark_q['id'] in set_ids_covered_by_this_kg_theme: # Already fully covered by this theme in a previous iteration
                        temp_set_ids_covered_this_iteration_pass.add(benchmark_q['id'])
                        continue

                    eval_result = await self._llm_as_evaluator(
                        benchmark_question_detail=benchmark_q,
                        generated_main_category_name=mc_name_kg,
                        generated_main_question_en=current_main_q_text,
                        generated_sub_questions_en=current_sub_q_text,
                        generated_dimension=current_q_set_for_kg_theme.main_category_dimension,
                        knowledge_graph_context=refine_kg_ctx,
                        standard_document_excerpts=refine_chunk_ctx
                    )
                    eval_result_with_id = {**eval_result, "benchmark_id_evaluated": benchmark_q['id']}
                    all_feedback_for_this_kg_theme.append(eval_result_with_id)

                    if eval_result.get("coverage_assessment", "None").lower() == "full":
                        temp_set_ids_covered_this_iteration_pass.add(benchmark_q['id'])
                    else: # Partial or None coverage for this benchmark
                        all_benchmarks_covered_this_iteration_pass = False
                        print(f"[QG_SERVICE AGENT LOOP - KG THEME] '{mc_name_kg}' needs refinement for SET Q: {benchmark_q['id']}.")
                        current_sub_q_text, current_sub_q_source_info = await self._refine_sub_questions_based_on_feedback(
                            main_category_name=mc_name_kg, main_question_text=current_main_q_text,
                            existing_sub_questions=current_sub_q_text,
                            feedback_suggestions=eval_result.get("suggested_improvements"),
                            missing_aspects=eval_result.get("missing_aspects"),
                            original_main_category_info=main_category_info_from_kg, # Pass the original KG theme info
                            knowledge_graph_context=refine_kg_ctx, standard_document_excerpts=refine_chunk_ctx,
                            main_category_dimension=current_q_set_for_kg_theme.main_category_dimension
                        )
                        # After one refinement for a specific benchmark, break to re-evaluate all benchmarks for this KG theme
                        break 
                
                set_ids_covered_by_this_kg_theme.update(temp_set_ids_covered_this_iteration_pass)
                
                if all_benchmarks_covered_this_iteration_pass and len(set_ids_covered_by_this_kg_theme) == len(relevant_set_benchmarks_for_kg_theme):
                    is_fully_validated_against_relevant_set = True
                    validation_status_for_this_kg_theme = "Validated_SET_Covered"
                
                current_iteration += 1
            
            if not is_fully_validated_against_relevant_set and relevant_set_benchmarks_for_kg_theme:
                 validation_status_for_this_kg_theme = "Validated_Partial_SET_Coverage"
            
            successfully_covered_set_question_ids_this_run.update(set_ids_covered_by_this_kg_theme)
            themes_to_be_active_this_run.add(mc_name_kg) # Mark this KG theme as processed for activation

            # Upsert the (potentially refined) KG-derived theme
            final_kg_theme_q_set = GeneratedQuestionSet(
                main_question_text_en=current_main_q_text, rolled_up_sub_questions_text_en=current_sub_q_text,
                main_category_name=mc_name_kg, main_category_dimension=current_q_set_for_kg_theme.main_category_dimension,
                main_category_keywords=current_q_set_for_kg_theme.main_category_keywords,
                main_category_description=current_q_set_for_kg_theme.main_category_description,
                main_category_constituent_entities=current_q_set_for_kg_theme.main_category_constituent_entities,
                main_category_source_docs=current_q_set_for_kg_theme.main_category_source_docs,
                detailed_source_info_for_subquestions=current_sub_q_source_info
            )
            await self._prepare_and_upsert_theme_to_db(final_kg_theme_q_set, main_category_info_from_kg, 
                                                      validation_status_for_this_kg_theme, all_feedback_for_this_kg_theme, 
                                                      list(set_ids_covered_by_this_kg_theme), 
                                                      all_historical_main_categories_map, current_time_utc, api_response_questions)
        # --- END LOOP for KG-derived themes ---


        # --- STEP 7 (from plan): Addressing Uncovered SET Questions (Gap-Filling Loop) ---
        if benchmark_set_questions: # Only run if benchmarks are loaded
            print(f"[QG_SERVICE AGENT GAP-FILL] Checking for SET questions not covered by KG themes. Total KG-covered SET IDs: {len(successfully_covered_set_question_ids_this_run)}")
            uncovered_set_qs = [
                bq for bq in benchmark_set_questions 
                if bq['id'] not in successfully_covered_set_question_ids_this_run
            ]

            if uncovered_set_qs:
                print(f"[QG_SERVICE AGENT GAP-FILL] Found {len(uncovered_set_qs)} SET questions for direct gap-filling.")
                for set_q_to_cover in uncovered_set_qs:
                    print(f"[QG_SERVICE AGENT GAP-FILL] Attempting for SET ID: {set_q_to_cover['id']} - {set_q_to_cover['theme_set']}")
                    
                    gap_fill_theme_name = f"SET Coverage: {set_q_to_cover['theme_set']} (ID: {set_q_to_cover['id']})"
                    
                    # Check if this gap-fill theme already exists and is active from a previous successful run
                    # This avoids re-processing if the content for this specific SET ID is already optimally generated
                    existing_gap_theme_versions = all_historical_main_categories_map.get(gap_fill_theme_name, [])
                    if existing_gap_theme_versions and existing_gap_theme_versions[0].is_active and \
                       (existing_gap_theme_versions[0].validation_status == "Validated_SET_Targeted" or 
                        existing_gap_theme_versions[0].validation_status == "Validated_SET_Targeted_No_SubQs"):
                        print(f"[QG_SERVICE AGENT GAP-FILL] SET ID {set_q_to_cover['id']} already covered by active theme '{gap_fill_theme_name}'. Adding to response.")
                        successfully_covered_set_question_ids_this_run.add(set_q_to_cover['id'])
                        themes_to_be_active_this_run.add(gap_fill_theme_name) # Ensure it's considered active
                        # Add existing questions to API response if not already added by KG theme processing
                        # (this ensures UI gets all relevant questions if a KG theme coincidentally had same name)
                        if not any(q.theme == gap_fill_theme_name for q in api_response_questions):
                            self._add_existing_theme_to_api_response(existing_gap_theme_versions[0], api_response_questions)
                        continue

                    # Use iterative search and generation for this SET gap
                    max_search_iter_for_gap = int(os.getenv("QG_MAX_SEARCH_ITERATIONS_GAP_FILL", "1")) # 0 means one attempt, 1 means initial + 1 retry
                    generated_sub_q_for_gap, source_info_for_gap, final_kg_ctx_gap, final_chunk_ctx_gap = \
                        await self._iterative_search_and_generate_for_set_gap(set_q_to_cover, max_search_iterations=max_search_iter_for_gap)

                    if generated_sub_q_for_gap: # Meaningful sub-questions were generated
                        print(f"[QG_SERVICE AGENT GAP-FILL] Successfully generated questions for SET ID: {set_q_to_cover['id']}")
                        
                        gap_main_q_en = set_q_to_cover.get('question_text_en', f"What is the company's approach regarding {set_q_to_cover['theme_set']}?")
                        gap_theme_keywords = f"{set_q_to_cover['theme_set']}, {set_q_to_cover.get('question_text_en','')}"
                        gap_theme_desc = f"Questions specifically generated to cover SET benchmark ID: {set_q_to_cover['id']} - {set_q_to_cover['question_text_th']}"
                        
                        gap_fill_q_set = GeneratedQuestionSet(
                            main_question_text_en=gap_main_q_en,
                            rolled_up_sub_questions_text_en=generated_sub_q_for_gap,
                            main_category_name=gap_fill_theme_name,
                            main_category_dimension=set_q_to_cover['dimension'],
                            main_category_keywords=gap_theme_keywords,
                            main_category_description=gap_theme_desc,
                            # constituent_entities and source_docs might be inferred from context if needed, or left empty
                            detailed_source_info_for_subquestions=source_info_for_gap
                        )
                        
                        original_info_for_gap_fill = { # Mocking structure for _prepare_and_upsert_theme_to_db
                            "main_category_name_en": gap_fill_theme_name, # Important: use the unique gap-fill theme name
                            "main_category_description_en": gap_theme_desc,
                            "dimension": set_q_to_cover['dimension'],
                            "main_category_keywords_en": gap_theme_keywords,
                            "generation_method": "set_gap_filling_agentic_loop",
                             "_main_category_raw_id": f"set_gap_{set_q_to_cover['id']}",
                            # Add KG/Chunk context used, if desired for metadata
                            "_final_kg_context_used": final_kg_ctx_gap,
                            "_final_chunk_context_used": final_chunk_ctx_gap
                        }
                        
                        await self._prepare_and_upsert_theme_to_db(
                            q_set=gap_fill_q_set,
                            original_theme_info_from_source=original_info_for_gap_fill, # This is mocked for gap-fill
                            validation_status="Validated_SET_Targeted",
                            validation_feedback_list=[{"benchmark_id_evaluated": set_q_to_cover['id'], "coverage_assessment": "Full_Via_Gap_Fill"}],
                            set_benchmark_ids_covered_list=[set_q_to_cover['id']],
                            all_historical_main_categories_map=all_historical_main_categories_map,
                            current_time_utc=current_time_utc,
                            api_response_questions_list=api_response_questions
                        )
                        successfully_covered_set_question_ids_this_run.add(set_q_to_cover['id'])
                        themes_to_be_active_this_run.add(gap_fill_theme_name)
                    else:
                        print(f"[QG_SERVICE AGENT GAP-FILL] Could not generate or context insufficient for SET ID: {set_q_to_cover['id']}. Details: {source_info_for_gap}")
                        # Optionally, store a placeholder "theme" indicating this SET question was attempted but not covered
                        # For now, we just log and it remains "uncovered" for this run.
            else: # No benchmarks loaded or all covered
                print("[QG_SERVICE AGENT GAP-FILL] No uncovered SET questions to process or benchmarks not loaded.")
        # --- END GAP-FILLING LOOP ---

        # --- Final Deactivation Logic ---
        if process_scope in [PROCESS_SCOPE_KG_INITIAL_FULL_SCAN_FROM_CONTENT, PROCESS_SCOPE_KG_UPDATED_FULL_SCAN_FROM_CONTENT, PROCESS_SCOPE_KG_FULL_SCAN_DB_EMPTY_FROM_CONTENT, PROCESS_SCOPE_KG_SCHEDULED_REFRESH_FROM_CONTENT]:
            for theme_name_to_check_deact in active_theme_names_at_start_of_run:
                if theme_name_to_check_deact not in themes_to_be_active_this_run:
                    print(f"[QG_SERVICE DEACTIVATION] Theme '{theme_name_to_check_deact}' was active but not processed/validated in this run. Deactivating.")
                    await ESGQuestion.find(
                        ESGQuestion.theme == theme_name_to_check_deact, 
                        ESGQuestion.is_active == True
                    ).update({"$set": {"is_active": False, "updated_at": current_time_utc}})
        
        print(f"[QG_SERVICE LOG] Evolution with Agentic Behavior complete. API response questions: {len(api_response_questions)}. Total unique SET Qs covered this run: {len(successfully_covered_set_question_ids_this_run)}")
        return api_response_questions

    async def _prepare_and_upsert_theme_to_db(
        self,
        q_set: GeneratedQuestionSet,
        original_theme_info_from_source: Dict[str, Any], # Either from KG theme structure or mocked for gap-fill
        validation_status: str,
        validation_feedback_list: List[Dict[str, Any]],
        set_benchmark_ids_covered_list: List[str],
        all_historical_main_categories_map: Dict[str, List[ESGQuestion]],
        current_time_utc: datetime,
        api_response_questions_list: List[GeneratedQuestion] # To append for API response
    ):
        """Helper to translate, build DB model, and upsert, then add to API response."""
        main_cat_name = q_set.main_category_name

        main_q_text_th = await self.translate_text_to_thai(q_set.main_question_text_en, main_cat_name, q_set.main_category_keywords)
        sub_q_text_th = await self.translate_text_to_thai(q_set.rolled_up_sub_questions_text_en, main_cat_name, q_set.main_category_keywords)
        theme_desc_th = await self.translate_text_to_thai(q_set.main_category_description, main_cat_name, q_set.main_category_keywords)

        sub_q_detail_list_for_db = []
        if q_set.rolled_up_sub_questions_text_en and \
           "no specific sub-questions" not in q_set.rolled_up_sub_questions_text_en.lower() and \
           q_set.rolled_up_sub_questions_text_en.strip():
            
            sub_theme_desc_en = f"Detailed inquiries related to {main_cat_name}."
            if q_set.detailed_source_info_for_subquestions and "generated to cover SET benchmark" in q_set.detailed_source_info_for_subquestions: # More specific for gap-fill
                 sub_theme_desc_en = q_set.detailed_source_info_for_subquestions

            sub_theme_desc_th = await self.translate_text_to_thai(sub_theme_desc_en, main_cat_name, q_set.main_category_keywords)
            
            current_sub_q_detail = SubQuestionDetail(
                sub_question_text_en=q_set.rolled_up_sub_questions_text_en,
                sub_question_text_th=sub_q_text_th,
                sub_theme_name=f"Overall Sub-Questions for {main_cat_name}", # Or more specific if available
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
            "theme": main_cat_name, "category": q_set.main_category_dimension,
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
        await self._upsert_main_question_document_to_db(db_doc_data, all_historical_main_categories_map, current_time_utc)

        # Add to API response
        api_response_questions_list.append(GeneratedQuestion(
            question_text_en=q_set.main_question_text_en, question_text_th=main_q_text_th,
            category=q_set.main_category_dimension, theme=main_cat_name, is_main_question=True
        ))
        if sub_q_detail_list_for_db:
            sq_api_data = sub_q_detail_list_for_db[0]
            api_response_questions_list.append(GeneratedQuestion(
                question_text_en=sq_api_data["sub_question_text_en"], question_text_th=sq_api_data.get("sub_question_text_th"),
                category=sq_api_data["category_dimension"], theme=main_cat_name,
                sub_theme_name=sq_api_data["sub_theme_name"], is_main_question=False,
                additional_info={"detailed_source_info_for_subquestions": sq_api_data.get("detailed_source_info")}
            ))    

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

    def load_set_benchmark_questions(self) -> List[Dict[str, Any]]:
        """
        Loads the SET benchmark questions with Thai and rephrased English versions.
        The English questions are framed to be more open-ended.
        """
        benchmark_questions = [
            # --- Environmental Dimension ---
            {
                "id": "SET_E_1", "dimension": "Environmental", "theme_set": "Reporting on SIA/EIA",
                "question_text_th": "บริษัทมีการการเปิดเผยข้อมูลการประเมินผลกระทบด้านสังคมและสิ่งแวดล้อม (SIA/EIA) หรือไม่?",
                "question_text_en": "Does the company disclose its social and environmental impact assessment (SIA/EIA)?"
            },
            {
                "id": "SET_E_2", "dimension": "Environmental", "theme_set": "Reporting on SIA/EIA",
                "question_text_th": "บริษัทมีการการเปิดเผยข้อมูลเกี่ยวกับกระบวนการติดตามผลกระทบด้านสังคมและสิ่งแวดล้อม (SIA/EIA) หรือไม่?",
                "question_text_en": "Does the company disclose its process for monitoring social and environmental impacts (SIA/EIA)?"
            },
            {
                "id": "SET_E_3", "dimension": "Environmental", "theme_set": "Environmentally Friendly Products", # Mapping to "Policy and guidelines for preventing contamination..."
                "question_text_th": "บริษัทมีนโยบายและแนวปฏิบัติเกี่ยวกับการป้องกันการปนเปื้อนหรือรั่วไหลจากกระบวนการผลิตหรือไม่?",
                "question_text_en": "Does the company have a policy and guidelines for preventing contamination or leakage from its production processes?"
            },
            {
                "id": "SET_E_4", "dimension": "Environmental", "theme_set": "Environmentally Friendly Products", # Mapping to "The life cycle impact assessment of products"
                "question_text_th": "บริษัทมีการการประเมินผลกระทบและวัฏจักรชีวิตของผลิตภัณฑ์ (life cycle impact assessment) หรือไม่?",
                "question_text_en": "Has the company conducted a life cycle impact assessment of its products?"
            },
            {
                "id": "SET_E_5", "dimension": "Environmental", "theme_set": "Environmentally Friendly Products", # Mapping to "Percentage of sales for environmentally friendly products..."
                "question_text_th": "ร้อยละของยอดขายผลิตภัณฑ์ที่เป็นมิตรต่อสิ่งแวดล้อม (eco products) ต่อยอดขายผลิตภัณฑ์ทั้งหมด เท่าไหร่?",
                "question_text_en": "Does the company report the percentage of sales from environmentally friendly products (eco products) compared to total product sales?"
            },
            {
                "id": "SET_E_6", "dimension": "Environmental", "theme_set": "Biodiversity and Cessation of Deforestation",
                "question_text_th": "บริษัทมีนโยบายและแนวปฏิบัติเกี่ยวกับการอนุรักษ์ความหลากหลายทางชีวภาพและยุติการตัดไม้ทำาลายป่า โดยครอบคลุมกระบวนการดำาเนินธุรกิจและห่วงโซ่อุปทานของบริษัทหรือไม่?",
                "question_text_en": "Does the company have a policy and guidelines regarding the conservation of biodiversity and cessation of deforestation, covering its business operations and supply chain?"
            },
            {
                "id": "SET_E_7", "dimension": "Environmental", "theme_set": "Biodiversity and Cessation of Deforestation",
                "question_text_th": "บริษัทมีการการประเมินความเสี่ยงและผลกระทบต่อความหลากหลายทางชีวภาพจากการดำเนินธุรกิจ หรือไม่?",
                "question_text_en": "Has the company assessed the risks and impacts on biodiversity resulting from its business operations?"
            },
            {
                "id": "SET_E_8", "dimension": "Environmental", "theme_set": "Biodiversity and Cessation of Deforestation",
                "question_text_th": "จำนวนพื้นที่การดำเนินธุรกิจของบริษัทที่มีการอนุรักษ์ความหลากหลายทางชีวภาพ เป็นจำนวนเท่าใด(ตารางเมตร)?",
                "question_text_en": "Does the company report the area of its business operations with biodiversity conservation efforts (in square meters)?"
            },
            {
                "id": "SET_E_9", "dimension": "Environmental", "theme_set": "Biodiversity and Cessation of Deforestation",
                "question_text_th": "จำนวนพื้นที่ที่ได้รับการรักษาภายใต้การดูแลของบริษัท เป็นจำนวนเท่าใด(ตารางเมตร)?", # Assuming "รักษา" means conserved/protected
                "question_text_en": "Does the company report the area of land conserved or protected under its care (in square meters)?"
            },
            {
                "id": "SET_E_10", "dimension": "Environmental", "theme_set": "Biodiversity and Cessation of Deforestation",
                "question_text_th": "บริษัทมีแผนงานหรือโครงการอนุรักษ์ความหลากหลายทางชีวภาพในการดำเนินธุรกิจ หรือไม่?",
                "question_text_en": "Does the company have biodiversity conservation plans or projects in its business operations?"
            },
            {
                "id": "SET_E_11", "dimension": "Environmental", "theme_set": "Biodiversity and Cessation of Deforestation",
                "question_text_th": "บริษัทมีแผนงานหรือโครงการอนุรักษ์พื้นที่ป่าในการดำเนินธุรกิจ หรือไม่?",
                "question_text_en": "Does the company have forest conservation plans or projects in its business operations?"
            },
            {
                "id": "SET_E_12", "dimension": "Environmental", "theme_set": "Environmentally Friendly Materials",
                "question_text_th": "ปริมาณน้ำหนักรวมของวัสดุทั้งหมด เป็นเท่าใด (กิโลกรัม)?",
                "question_text_en": "Does the company report the total weight of all materials used (in kilograms), optionally classified by type (e.g., non-renewable, renewable)?"
            },
            {
                "id": "SET_E_13", "dimension": "Environmental", "theme_set": "Environmentally Friendly Materials",
                "question_text_th": "ร้อยละของวัสดุรีไซเคิลที่นำกลับมาใช้ในการพัฒนาผลิตภัณฑ์ เท่าไหร่?",
                "question_text_en": "Does the company report the percentage of recycled input materials used in its product development?"
            },
            {
                "id": "SET_E_14", "dimension": "Environmental", "theme_set": "Environmentally Friendly Materials",
                "question_text_th": "ร้อยละของวัสดุจากของเหลือหมดอายุหรือเสื่อมคุณภาพ (reclaimed) และถูกนำกลับมาใช้ในการพัฒนาผลิตภัณฑ์ เท่าไหร่?",
                "question_text_en": "Does the company report the percentage of reclaimed materials (from expired or deteriorated sources) reused in its product development?"
            },
            {
                "id": "SET_E_15", "dimension": "Environmental", "theme_set": "Air Pollution",
                "question_text_th": "ปริมาณมลพิษทางอากาศจากการดำเนินธุรกิจ - Nitrogen oxide (NOₓ), Sulfur dioxide (SOₓ), Persistent organic pollutants (POP), Volatile organic compounds (VOC), Hazardous air pollutants (HAP), Particulate matter (PM), อื่น ๆ เป็นเท่าใด?",
                "question_text_en": "Does the company report the volume of its air pollutants from business operations (e.g., NOₓ, SOₓ, POPs, VOCs, HAPs, PM, others)?"
            },
            {
                "id": "SET_E_16", "dimension": "Environmental", "theme_set": "Climate Change Risks",
                "question_text_th": "บริษัทมีการการประเมินความเสี่ยงจากการเปลี่ยนแปลงสภาพภูมิอากาศ โดยอธิบายผลกระทบที่อาจส่งผลต่อการดำเนินธุรกิจ หรือไม่?",
                "question_text_en": "Has the company conducted a climate change risk assessment, including an explanation of potential impacts on its business operations?"
            },
            {
                "id": "SET_E_17", "dimension": "Environmental", "theme_set": "Climate Change Risks",
                "question_text_th": "บริษัทมีเป้าหมาย แผนงาน และมาตรการบรรเทาความเสี่ยงจากการเปลี่ยนแปลงสภาพภูมิอากาศ หรือไม่?",
                "question_text_en": "Does the company have established goals, plans, and measures to mitigate climate change risks?"
            },

            # --- Social Dimension ---
            {
                "id": "SET_S_1", "dimension": "Social", "theme_set": "Local Employment",
                "question_text_th": "บริษัทมีนโยบายและแนวปฏิบัติเกี่ยวกับการจ้างแรงงานท้องถิ่น หรือไม่?",
                "question_text_en": "Does the company have a policy and guidelines regarding local employment?"
            },
            {
                "id": "SET_S_2", "dimension": "Social", "theme_set": "Local Employment",
                "question_text_th": "ร้อยละของพนักงานที่มาจากชุมชนท้องถิ่น เท่าไหร่?",
                "question_text_en": "Does the company report the percentage of its employees from local communities?"
            },
            {
                "id": "SET_S_3", "dimension": "Social", "theme_set": "Respecting Diversity and Equality",
                "question_text_th": "บริษัทมีนโยบายและแนวปฏิบัติเกี่ยวกับการเคารพความแตกต่างและความเสมอภาคภายในองค์กรและห่วงโซ่อุปทาน โดยไม่แบ่งแยกเพศ อายุ เชื้อชาติ ความพิการ ศาสนา หรืออื่น ๆ หรือไม่?",
                "question_text_en": "Does the company have a policy and guidelines regarding respect for diversity and equality within the organization and its supply chain (e.g., non-discrimination based on gender, age, race, disability, religion, or other factors)?"
            },
            {
                "id": "SET_S_4", "dimension": "Social", "theme_set": "Respecting Diversity and Equality",
                "question_text_th": "สถิติจำนวนพนักงานตามเพศและสัญชาติ เป็นจำนวนเท่าใด?",
                "question_text_en": "Does the company disclose employee statistics categorized by gender and nationality?"
            },
            {
                "id": "SET_S_5", "dimension": "Social", "theme_set": "Respecting Diversity and Equality",
                "question_text_th": "จำนวนเหตุการณ์หรือข้อร้องเรียนเกี่ยวกับการละเมิดสิทธิ ความเสมอภาค และการปฏิบัติต่อแรงงานอย่างไม่เป็นธรรม พร้อมมาตรการแก้ไขและเยียวยา เป็นจำนวนเท่าใด?",
                "question_text_en": "Does the company report the number of incidents or complaints related to violations of rights, equality, and unfair labor treatment, along with corresponding remediation and mitigation measures?"
            },
            {
                "id": "SET_S_6", "dimension": "Social", "theme_set": "Promotion of Female Workforce",
                "question_text_th": "บริษัทมีนโยบายและแนวปฏิบัติเกี่ยวกับการส่งเสริมแรงงานสตรีในสถานประกอบการอย่างเท่าเทียมกัน หรือไม่?",
                "question_text_en": "Does the company have a policy and guidelines related to promoting equal opportunities for the female workforce in the workplace?"
            },
            {
                "id": "SET_S_7", "dimension": "Social", "theme_set": "Promotion of Female Workforce",
                "question_text_th": "ข้อมูลพนักงานหญิงแบ่งตามตำแหน่งงานมีจำนวนเท่าใด?",
                "question_text_en": "Does the company disclose the number of female employees categorized by employment level (e.g., senior management, management, staff level)?"
            },
            {
                "id": "SET_S_8", "dimension": "Social", "theme_set": "Monitoring and Assessing Impacts on Communities",
                "question_text_th": "บริษัทมีการการติดตามและประเมินผลกระทบต่อชุมชนจากการดำเนินธุรกิจของบริษัท หรือไม่?",
                "question_text_en": "Does the company monitor and assess the impacts of its business operations on communities?"
            },
            {
                "id": "SET_S_9", "dimension": "Social", "theme_set": "Monitoring and Assessing Impacts on Communities",
                "question_text_th": "จำนวนกรณีพิพาทหรือเหตุการณ์ร้องเรียนเกี่ยวกับการละเมิดสิทธิชุมชน พร้อมมาตรการแก้ไขและเยียวยา เป็นจำนวนเท่าใด?",
                "question_text_en": "Does the company report the number of disputes or complaints regarding community rights violations, along with corresponding remediation and mitigation measures?"
            },

            # --- Governance Dimension ---
            {
                "id": "SET_G_1", "dimension": "Governance", "theme_set": "Cybersecurity and Personal Data Protection",
                "question_text_th": "บริษัทมีนโยบายและแนวปฏิบัติเกี่ยวกับด้านความปลอดภัยทางไซเบอร์และการป้องกันข้อมูลส่วนบุคคล หรือไม่?",
                "question_text_en": "Does the company have a policy and guidelines on cybersecurity and personal data protection?"
            },
            {
                "id": "SET_G_2", "dimension": "Governance", "theme_set": "Cybersecurity and Personal Data Protection",
                "question_text_th": "ร้อยละของจำนวนโครงสร้างพื้นฐานด้านเทคโนโลยีที่ได้รับการรับรองมาตรฐานด้านความปลอดภัยทางไซเบอร์ เช่น ISO 27001 หรือมาตรฐานอื่น ๆ เป็นต้น เท่าไหร่?",
                "question_text_en": "Does the company report the percentage of its technology infrastructures that have been certified with cybersecurity standards (e.g., ISO 27001 or other relevant standards)?"
            },
            {
                "id": "SET_G_3", "dimension": "Governance", "theme_set": "Cybersecurity and Personal Data Protection",
                "question_text_th": "บริษัทมีมาตรการและแนวปฏิบัติเกี่ยวกับการใช้ข้อมูลส่วนบุคคล หรือไม่?",
                "question_text_en": "Does the company have established measures and guidelines related to personal data usage?"
            },
            {
                "id": "SET_G_4", "dimension": "Governance", "theme_set": "Cybersecurity and Personal Data Protection",
                "question_text_th": "ร้อยละของพนักงานที่ได้รับการอบรมด้านความปลอดภัยทางไซเบอร์และการใช้ข้อมูลส่วนบุคคล เท่าไหร่?",
                "question_text_en": "Does the company report the percentage of its employees who have received training in cybersecurity and personal data usage?"
            },
            {
                "id": "SET_G_5", "dimension": "Governance", "theme_set": "Cybersecurity and Personal Data Protection",
                "question_text_th": "จำนวนเหตุการณ์หรือกรณีที่บริษัทถูกโจมตีทางไซเบอร์ พร้อมมาตรการแก้ไข เป็นจำนวนเท่าใด?",
                "question_text_en": "Does the company report the number of incidents or cases of cyberattacks against the company, along with corresponding mitigation measures?"
            },
            {
                "id": "SET_G_6", "dimension": "Governance", "theme_set": "Cybersecurity and Personal Data Protection",
                "question_text_th": "จำนวนเหตุการณ์หรือกรณีข้อมูลส่วนบุคคลรั่วไหล พร้อมมาตรการแก้ไข เป็นจำนวนเท่าใด?",
                "question_text_en": "Does the company report the number of incidents or cases of personal data breaches, along with corresponding mitigation measures?"
            },
            {
                "id": "SET_G_7", "dimension": "Governance", "theme_set": "Product Quality and Recall",
                "question_text_th": "บริษัทมีนโยบายและแนวปฏิบัติเกี่ยวกับการจัดการด้านคุณภาพของผลิตภัณฑ์ ตามมาตรฐานสากล เช่น ISO 9001:2015 หรือมาตรฐานอื่น ๆ เป็นต้น หรือไม่?",
                "question_text_en": "Does the company have a policy and guidelines for product quality management according to international standards (e.g., ISO 9001:2015 or other standards)?"
            },
            {
                "id": "SET_G_8", "dimension": "Governance", "theme_set": "Product Quality and Recall",
                "question_text_th": "บริษัทมีแผนการเรียกคืนผลิตภัณฑ์ หรือไม่?",
                "question_text_en": "Does the company have a product recall plan?"
            },
            {
                "id": "SET_G_9", "dimension": "Governance", "theme_set": "Product Quality and Recall",
                "question_text_th": "จำนวนกรณีหรือเหตุการณ์เรียกคืนผลิตภัณฑ์ พร้อมมาตรการแก้ไขและเยียวยา เป็นจำนวนเท่าใด?",
                "question_text_en": "Does the company report the number of cases or incidents of product recall, along with corresponding remediation and mitigation measures?"
            }
        ]
        return benchmark_questions

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

    Output ONLY a single, valid JSON object with these exact keys:
    - "rolled_up_sub_questions_text_en": A string containing ONLY 2-4 sub-questions, each numbered and on a new line, OR the specific insufficiency message mentioned in instruction 3.
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
        
    async def _iterative_search_and_generate_for_set_gap(
        self,
        set_q_to_cover: Dict[str, Any],
        max_search_iterations: int = 1 # Default to 0 retries (1 attempt total for search)
    ) -> Tuple[Optional[str], Optional[str], str, str]:
        print(f"[QG_SERVICE ITERATIVE_SEARCH] Starting for SET ID: {set_q_to_cover['id']}")
        current_iteration = 0
        generated_sub_q: Optional[str] = None
        source_info: Optional[str] = "Attempted generation, but no meaningful questions were ultimately validated."
        
        current_search_keywords = f"{set_q_to_cover['theme_set']} {set_q_to_cover.get('question_text_en','')}"
        
        kg_context_for_set_gap = "Context not fetched or deemed irrelevant in initial attempts."
        chunk_context_for_set_gap = "Context not fetched or deemed irrelevant in initial attempts."

        while current_iteration <= max_search_iterations:
            print(f"[QG_SERVICE ITERATIVE_SEARCH] Iteration {current_iteration + 1}/{max_search_iterations + 1} for SET ID: {set_q_to_cover['id']} using keywords: '{current_search_keywords[:100]}...'")

            # 1. Fetch Context (Consider adding entity relevance filter here or in neo4j_service if feasible)
            # For now, we rely on get_graph_context_for_theme_chunks_v2 using the SET theme/keywords.
            # A more advanced step would be to get entities, filter them for relevance to SET Q, then build context.
            current_kg_context = await self.neo4j_service.get_graph_context_for_theme_chunks_v2(
                theme_name=set_q_to_cover['theme_set'],
                theme_keywords=current_search_keywords,
                max_central_entities=2, max_hops_for_central_entity=1,
                max_relations_to_collect_per_central_entity=3,
                max_total_context_items_str_len=800000
            )
            current_chunk_context = "No specific document excerpts found for current search terms."
            try:
                similar_chunks = await self.neo4j_service.get_relate(query=current_search_keywords, k=1) # Reduced to k=1 for more focused context
                if similar_chunks:
                    chunk_texts_list = [f"Excerpt from '{sc.metadata.get('doc_id', 'Unknown')}': \"{sc.page_content[:1500]}...\"" for sc in similar_chunks] # Increased preview
                    current_chunk_context = "\n---\n".join(chunk_texts_list)
            except Exception as e_set_chunk_ctx:
                print(f"[QG_SERVICE ITERATIVE_SEARCH] Error fetching chunk context for SET Q {set_q_to_cover['id']}: {e_set_chunk_ctx}")

            # Store context used in this iteration for potential return if successful
            kg_context_for_set_gap = current_kg_context or "No KG context found."
            chunk_context_for_set_gap = current_chunk_context

            # 2. Attempt to generate questions with current context
            set_gap_main_q_text_en = set_q_to_cover.get('question_text_en', f"Regarding {set_q_to_cover['theme_set']}, what is the company's approach or disclosure?")
            
            temp_generated_sub_q, temp_source_info = await self._generate_questions_for_set_gap_fill(
                set_question_to_cover=set_q_to_cover,
                main_question_text_en=set_gap_main_q_text_en,
                knowledge_graph_context=kg_context_for_set_gap,
                standard_document_excerpts=chunk_context_for_set_gap,
                target_dimension=set_q_to_cover['dimension']
            )

            # 3. Validate the quality and relevance of generated questions
            is_valid_and_grounded = False
            validation_justification = "Generation did not produce output for validation."

            if temp_generated_sub_q: # Only validate if something was generated
                is_valid_and_grounded, validation_justification = await self._validate_generated_gap_fill_qs(
                    set_question_to_cover=set_q_to_cover,
                    generated_sub_questions=temp_generated_sub_q,
                    context_used_kg=kg_context_for_set_gap,
                    context_used_chunks=chunk_context_for_set_gap
                )
            
            if is_valid_and_grounded and temp_generated_sub_q is not None: # Ensure temp_generated_sub_q is not None
                generated_sub_q = temp_generated_sub_q
                source_info = temp_source_info if temp_source_info else validation_justification
                print(f"[QG_SERVICE ITERATIVE_SEARCH] Validated meaningful sub-questions generated for SET ID {set_q_to_cover['id']} in iteration {current_iteration + 1}.")
                return generated_sub_q, source_info, kg_context_for_set_gap, chunk_context_for_set_gap # Success

            # 4. If not successful and more iterations are allowed, ask LLM for new search terms
            source_info = validation_justification # Update source_info with the latest justification if not successful
            if current_iteration < max_search_iterations:
                print(f"[QG_SERVICE ITERATIVE_SEARCH] Attempting to get alternative search terms for SET ID {set_q_to_cover['id']} due to: {validation_justification}")
                alternative_terms = await self._llm_suggest_alternative_search_terms(
                    set_q_to_cover,
                    initial_kg_context=kg_context_for_set_gap,
                    initial_chunk_context=chunk_context_for_set_gap
                )
                if alternative_terms:
                    current_search_keywords = ", ".join(alternative_terms)
                    print(f"[QG_SERVICE ITERATIVE_SEARCH] New search terms for next iteration: {current_search_keywords}")
                else:
                    print(f"[QG_SERVICE ITERATIVE_SEARCH] LLM could not suggest alternative search terms. Ending search for SET ID {set_q_to_cover['id']}.")
                    break 
            
            current_iteration += 1

        final_justification = source_info if source_info else "Max iterations reached, and no validated meaningful questions generated."
        print(f"[QG_SERVICE ITERATIVE_SEARCH] Failed to generate validated meaningful sub-questions for SET ID {set_q_to_cover['id']} after all iterations. Last justification: {final_justification}")
        return None, final_justification, kg_context_for_set_gap, chunk_context_for_set_gap
    
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

