import asyncio
import os
import io
import re
import tempfile
from typing import List, Dict, Any, Optional, Union
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
from app.models.esg_question_model import ESGQuestion
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
    category: str = PydanticField(..., description="The ESG category (E, S, or G).") # Category for the question/theme
    theme: str = PydanticField(..., description="The ESG theme or area of inquiry this question relates to.") # Consolidated theme name
    additional_info: Optional[Dict[str, Any]] = PydanticField(None, description="Additional non-standard information carried with the question.")


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
            model=os.getenv("TRANSLATION_MODEL", "gemini-2.0-flash"), # Updated model name
            temperature=0.8,
            max_retries=3,
        )
        # self.graph_schema_content is NO LONGER USED for theme identification
        # self._load_graph_schema() # Remove this call or its usage for themes

        if self.similarity_llm_embedding:
            print("[QG_SERVICE LOG] Similarity embedding model received and configured.")
        else:
            print("[QG_SERVICE WARNING] Similarity embedding model was NOT provided. Similarity checks might fallback or be less accurate.")

    # _load_graph_schema method can be removed if no longer used elsewhere.

    async def translate_text_to_thai(self, text_to_translate: str) -> Optional[str]:
        # ... (existing implementation is fine) ...
        if not text_to_translate: return None
        prompt_template = PromptTemplate.from_template(
            "Translate the following English text to Thai. Provide only the Thai translation.\n\nEnglish Text: \"{english_text}\"\n\nThai Translation:"
        )
        formatted_prompt = prompt_template.format(english_text=text_to_translate)
        try:
            response = await self.translation_llm.ainvoke(formatted_prompt)
            return response.content.strip()
        except Exception as e:
            print(f"[QG_SERVICE ERROR] Error during translation to Thai for text '{text_to_translate[:30]}...': {e}")
            return None

    # analyze_pdf_impact might still be useful as a preliminary check, but not for primary theme ID.
    # The current implementation of analyze_pdf_impact expects existing theme names as input.
    # For now, we'll bypass its direct use in evolve_and_store_questions' main theme generation path.
    # async def analyze_pdf_impact(self, ...):
    # ... (existing implementation can remain for now if used elsewhere or for light impact assessment)

    async def _get_graph_data_for_community_detection(self) -> Optional[Dict[str, List[Any]]]:
        print("[QG_SERVICE LOG] Fetching graph data for community detection from Neo4j...")
        try:
            query_nodes = """
            MATCH (e:__Entity__)
            WHERE e.id IS NOT NULL
            OPTIONAL MATCH (sc:StandardChunk)-[:MENTIONS]->(e)
            RETURN
                e.id AS id,
                COALESCE(e.description, '') AS description,
                [lbl IN labels(e) WHERE lbl <> '__Entity__'] AS specific_labels,
                COALESCE(sc.doc_id, '') AS source_document_doc_id
            """
            query_edges = """
            MATCH (e1:__Entity__)-[r]-(e2:__Entity__)
            WHERE e1.id IS NOT NULL AND e2.id IS NOT NULL AND e1.id <> e2.id
            RETURN DISTINCT e1.id AS source, e2.id AS target
            """
            loop = asyncio.get_running_loop()
            nodes_result = await loop.run_in_executor(None, self.neo4j_service.graph.query, query_nodes)
            edges_result = await loop.run_in_executor(None, self.neo4j_service.graph.query, query_edges)

            if not nodes_result:
                print("[QG_SERVICE WARNING] No __Entity__ nodes found for community detection.")
                return None

            nodes = [
                {
                    'id': record['id'],
                    'description': record['description'],
                    'labels': record['specific_labels'] if record['specific_labels'] else [DEFAULT_NODE_TYPE],
                    'source_document_doc_id': record['source_document_doc_id']
                }
                for record in nodes_result if record and record.get('id')
            ]
            valid_node_ids = {node['id'] for node in nodes}
            edges = []
            if edges_result:
                for record in edges_result:
                    source = record.get('source')
                    target = record.get('target')
                    if source and target and source in valid_node_ids and target in valid_node_ids:
                        edges.append((source, target))
            
            unique_edges = list(set(tuple(sorted(edge_pair)) for edge_pair in edges))
            print(f"[QG_SERVICE INFO] Fetched {len(nodes)} nodes and {len(unique_edges)} unique edges for community detection.")
            if not nodes:
                 print("[QG_SERVICE WARNING] No valid nodes after processing for community detection.")
                 return None
            return {"nodes": nodes, "edges": unique_edges}
        except Exception as e:
            print(f"[QG_SERVICE ERROR] Error fetching graph data: {e}")
            traceback.print_exc()
            return None
            
    async def identify_themes_with_igraph_louvain(self, graph_data: Dict[str, List[Any]]) -> Dict[int, List[str]]:
        if not graph_data or not graph_data.get("nodes"):
            print("[QG_SERVICE WARNING] No nodes data provided for igraph Louvain.")
            return {}
        nodes_info = graph_data["nodes"]
        edges_info = graph_data.get("edges", [])
        if not nodes_info:
            print("[QG_SERVICE WARNING] Node list is empty for igraph Louvain.")
            return {}

        node_id_to_idx = {node['id']: i for i, node in enumerate(nodes_info)}
        idx_to_node_id = {i: node['id'] for i, node in enumerate(nodes_info)}
        igraph_edges = []
        for source_id, target_id in edges_info:
            if source_id in node_id_to_idx and target_id in node_id_to_idx:
                 igraph_edges.append((node_id_to_idx[source_id], node_id_to_idx[target_id]))

        num_vertices = len(nodes_info)
        if num_vertices == 0:
            print("[QG_SERVICE WARNING] No vertices to build graph for igraph.")
            return {}

        g = ig.Graph(n=num_vertices, edges=igraph_edges, directed=False)
        print(f"[QG_SERVICE INFO] Running igraph Louvain (community_multilevel) on a graph with {g.vcount()} vertices and {g.ecount()} edges.")
        
        if g.ecount() == 0 and g.vcount() > 1:
            print(f"[QG_SERVICE WARNING] Graph has {g.vcount()} nodes but no edges. Louvain might not be effective.")
            if 1 < g.vcount() <= 20: # Increased limit slightly for this fallback
                 print(f"[QG_SERVICE INFO] Grouping all {g.vcount()} isolated nodes into a single community as fallback.")
                 return {0: [node['id'] for node in nodes_info]}
            return {}
        try:
            vc = g.community_multilevel(return_levels=False)
            communities: Dict[int, List[str]] = {}
            for community_id, members_indices in enumerate(vc):
                if not members_indices: continue
                node_ids_in_this_community = [idx_to_node_id[idx] for idx in members_indices if idx in idx_to_node_id]
                if node_ids_in_this_community:
                    communities[community_id] = node_ids_in_this_community
            print(f"[QG_SERVICE INFO] igraph Louvain found {len(communities)} communities.")
            return communities
        except Exception as e:
            print(f"[QG_SERVICE ERROR] igraph Louvain community detection error: {e}")
            traceback.print_exc()
            if 1 < g.vcount() <= 30 : # Increased limit for error fallback
                 print(f"[QG_SERVICE WARNING] Fallback due to Louvain error: Treating up to 30 nodes as individual themes.")
                 return {i: [idx_to_node_id[i]] for i in range(num_vertices) if i in idx_to_node_id and idx_to_node_id.get(i)}
            return {}
        
    async def identify_consolidated_themes_from_kg(
        self,
        existing_active_theme_names: Optional[List[str]] = None,
        min_nodes_per_community_for_theme: int = 3
    ) -> List[Dict[str, Any]]:
        print("[QG_SERVICE LOG] Starting Consolidated Theme Identification using Graph Community Detection (igraph)...")
        graph_data = await self._get_graph_data_for_community_detection()
        if not graph_data or not graph_data.get("nodes"):
            print("[QG_SERVICE WARNING] Could not retrieve graph data (or no nodes found) for theme identification.")
            return []

        node_communities = await self.identify_themes_with_igraph_louvain(graph_data)
        
        if not node_communities:
            print("[QG_SERVICE WARNING] No communities found using graph-based community detection.")
            if graph_data and graph_data.get("nodes") and 1 < len(graph_data["nodes"]) <= 15: # Adjusted fallback
                print("[QG_SERVICE INFO] Fallback: No communities, but few nodes. Treating each node as a potential theme seed.")
                node_communities = {i: [node['id']] for i, node in enumerate(graph_data["nodes"])}
            else:
                return []

        all_entity_data_map = {entity_data['id']: entity_data for entity_data in graph_data.get("nodes", [])}
        identified_themes_from_kg: List[Dict[str, Any]] = []
        llm_tasks = []

        for community_id, node_ids_in_community in node_communities.items():
            if not node_ids_in_community or len(node_ids_in_community) < min_nodes_per_community_for_theme:
                # print(f"[QG_SERVICE INFO] Skipping community {community_id} as it has fewer than {min_nodes_per_community_for_theme} nodes ({len(node_ids_in_community)} nodes).")
                continue

            # print(f"[QG_SERVICE INFO] Processing community {community_id} with {len(node_ids_in_community)} nodes.")
            context_for_llm = ""
            char_count = 0
            max_chars_for_context = 70000 # Adjusted
            num_nodes_for_context = 0
            MAX_NODES_SAMPLE = 15 

            temp_context_parts = []
            constituent_entity_ids_for_theme = [] 
            source_doc_names_in_theme = set()

            for node_id in node_ids_in_community:
                entity_data = all_entity_data_map.get(node_id) 
                if entity_data:
                    constituent_entity_ids_for_theme.append(entity_data['id'])
                    node_type_display_list = entity_data.get('labels', [DEFAULT_NODE_TYPE])
                    node_type_display = ", ".join(node_type_display_list) if node_type_display_list else DEFAULT_NODE_TYPE
                    node_desc = entity_data.get('description', 'No description provided for this entity.')
                    if not node_desc.strip():
                        node_desc = "No substantive description provided."

                    doc_id_from_entity = entity_data.get('source_document_doc_id')
                    if doc_id_from_entity and isinstance(doc_id_from_entity, str) and doc_id_from_entity.strip():
                        source_doc_names_in_theme.add(doc_id_from_entity)
                    
                    if num_nodes_for_context < MAX_NODES_SAMPLE:
                        node_info_sample = f"ENTITY_ID: {node_id}\nENTITY_TYPE(S): {node_type_display}\nENTITY_DESCRIPTION: {node_desc}\n\n"
                        if char_count + len(node_info_sample) <= max_chars_for_context:
                            temp_context_parts.append(node_info_sample)
                            char_count += len(node_info_sample)
                            num_nodes_for_context += 1
                        else:
                            # print(f"[QG_SERVICE INFO] Max character limit for LLM context reached for community {community_id}. Stopping at {num_nodes_for_context} entities.")
                            break 
                    else:
                        # print(f"[QG_SERVICE INFO] Max entities sample limit ({MAX_NODES_SAMPLE}) reached for LLM context for community {community_id}.")
                        break 
            
            context_for_llm = "".join(temp_context_parts)
            if not context_for_llm.strip() and node_ids_in_community: 
                first_node_id = node_ids_in_community[0]
                first_entity_data = all_entity_data_map.get(first_node_id)
                if first_entity_data:
                    context_for_llm = f"ENTITY_ID: {first_node_id}\nENTITY_DESCRIPTION: {first_entity_data.get('description', 'N/A')}"[:max_chars_for_context]
            
            if not context_for_llm.strip():
                # print(f"[QG_SERVICE WARNING] No context could be generated for LLM for community {community_id}. Skipping.")
                continue

            source_standards_list_str_for_prompt = sorted(list(s for s in source_doc_names_in_theme if s))

            prompt_template_str = """
            You are an expert ESG analyst. A group of semantically related entities and concepts has been identified from an ESG knowledge graph. 
            Your task is to define a consolidated ESG reporting theme based on these entities, specifically for the **industrial packaging sector**.

            The following are representative entities/concepts and their descriptions from this group:
            --- REPRESENTATIVE ENTITIES/CONCEPTS START ---
            {representative_entity_content}
            --- REPRESENTATIVE ENTITIES/CONCEPTS END ---

            These entities and concepts primarily originate from or relate to standards document(s) such as: {source_standards_list_str}

            Based on this information:
            1.  Propose a concise and descriptive **Theme Name** (in English) for this group of entities/concepts. The theme name should be suitable for an ESG report.
            2.  Provide a brief **Theme Description** (in English, 1-2 sentences) that explains what this theme covers in the context of ESG for industrial packaging.
            3.  Determine the most relevant primary **ESG Dimension** for this theme ('E' for Environmental, 'S' for Social, or 'G' for Governance).
            4.  Suggest 3-5 relevant and specific **Keywords** (in English, comma-separated) that capture the essence of this theme.

            Output ONLY a single, valid JSON object with the following exact keys: "theme_name_en", "description_en", "dimension", "keywords_en".
            Example of a valid JSON output:
            {{"theme_name_en": "Sustainable Water Management", "description_en": "Covers policies, practices, and performance related to responsible water withdrawal, consumption, discharge, and risk mitigation in industrial packaging facilities.", "dimension": "E", "keywords_en": "water consumption, wastewater treatment, water recycling, water stewardship, water risk"}}
            """
            
            prompt = PromptTemplate.from_template(prompt_template_str)
            formatted_prompt = prompt.format(
                representative_entity_content=context_for_llm,
                source_standards_list_str=", ".join(source_standards_list_str_for_prompt[:3]) if source_standards_list_str_for_prompt else "various ESG standards"
            )
            
            llm_tasks.append(
                self._process_single_cluster_with_llm(
                    formatted_prompt,
                    constituent_entity_ids_for_theme, 
                    source_standards_list_str_for_prompt, 
                    f"community_{community_id}" 
                )
            )

        if llm_tasks:
            llm_results_with_exceptions = await asyncio.gather(*llm_tasks, return_exceptions=True)
            for result_item in llm_results_with_exceptions:
                # *** CRITICAL FIX: Check for "theme_name" as this is what _process_single_cluster_with_llm now returns as the primary theme identifier ***
                if isinstance(result_item, dict) and result_item.get("theme_name"): 
                    identified_themes_from_kg.append(result_item)
                elif isinstance(result_item, Exception):
                    print(f"[QG_SERVICE ERROR] LLM processing for a community failed: {result_item}")
                    # traceback.print_exc() # Consider conditional traceback
                elif result_item is None:
                    print(f"[QG_SERVICE WARNING] LLM processing for a community returned None.")
                else:
                    print(f"[QG_SERVICE WARNING] LLM processing for a community returned an unexpected item: {type(result_item)} with content {str(result_item)[:200]}")
        
        print(f"[QG_SERVICE LOG] Identified {len(identified_themes_from_kg)} consolidated themes from KG (igraph Community Detection) after LLM processing.")
        return identified_themes_from_kg

    async def _process_single_cluster_with_llm(
        self, 
        formatted_prompt: str, 
        constituent_ids: List[str], 
        source_doc_names: List[str],
        cluster_or_community_id: Union[int, str]
    ) -> Optional[Dict[str, Any]]:
        print(f"[QG_SERVICE INFO] Sending prompt to LLM for cluster/community ID: {cluster_or_community_id}")
        try:
            # print(f"DEBUG: Formatted prompt for LLM (community {cluster_or_community_id}):\n{formatted_prompt[:1000]}...\n")
            
            # Use self.qg_llm (defined in __init__)
            response = await self.qg_llm.ainvoke(formatted_prompt)
            llm_output = response.content.strip()
            
            json_str = ""
            # Enhanced JSON cleaning
            match = re.search(r"```json\s*(\{[\s\S]*?\})\s*```", llm_output, re.DOTALL | re.MULTILINE)
            if match:
                json_str = match.group(1)
            else:
                # Try to find the first '{' and last '}'
                first_brace = llm_output.find('{')
                last_brace = llm_output.rfind('}')
                if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
                    json_str = llm_output[first_brace : last_brace + 1]
                else:
                    print(f"[QG_SERVICE ERROR / Community {cluster_or_community_id}] LLM output does not appear to be valid JSON or wrapped JSON: {llm_output}")
                    return None
            
            llm_theme_data = json.loads(json_str)
            
            theme_name_en = llm_theme_data.get("theme_name_en")
            if not theme_name_en or not isinstance(theme_name_en, str) or not theme_name_en.strip(): 
                print(f"[QG_SERVICE WARNING / Community {cluster_or_community_id}] LLM did not provide a valid 'theme_name_en'. Raw: '{theme_name_en}'. Skipping.")
                return None

            theme_name_th = await self.translate_text_to_thai(theme_name_en)
            description_en = llm_theme_data.get("description_en", "")
            
            dimension = str(llm_theme_data.get("dimension", "G")).upper() # Ensure string before upper()
            if dimension not in ['E', 'S', 'G']: dimension = "G"

            keywords_val = llm_theme_data.get("keywords_en", "")
            if isinstance(keywords_val, list): # If LLM returns a list of keywords
                keywords_en_str = ", ".join(str(k) for k in keywords_val)
            elif isinstance(keywords_val, str):
                keywords_en_str = keywords_val
            else:
                keywords_en_str = ""


            # *** KEY CHANGE: Return 'theme_name' as the primary identifier, along with 'theme_name_en' ***
            final_theme_info = {
                "theme_name": theme_name_en.strip(), # This is the key evolve_and_store_questions expects
                "theme_name_en": theme_name_en.strip(),
                "theme_name_th": theme_name_th,
                "description_en": description_en,
                "dimension": dimension,
                "keywords_en": keywords_en_str, # Use the processed string
                "constituent_entity_ids": constituent_ids, # Correct key name
                "source_standard_document_ids": list(set(s for s in source_doc_names if s)), # Ensure unique and non-empty
                "generation_method": "kg_community_detection_igraph",
                "_community_id_source": str(cluster_or_community_id) # Ensure string
            }
            print(f"[QG_SERVICE INFO] Successfully processed community {cluster_or_community_id} into theme: {final_theme_info['theme_name_en']}")
            return final_theme_info

        except json.JSONDecodeError as e_json:
            print(f"[QG_SERVICE ERROR / Community {cluster_or_community_id}] Failed to parse JSON from LLM: {e_json}. Cleaned_JSON_Attempt: '{json_str}'. Original_LLM_Output: '{llm_output}'")
            return None
        except Exception as e_llm:
            print(f"[QG_SERVICE ERROR / Community {cluster_or_community_id}] LLM call or processing error: {e_llm}")
            traceback.print_exc()
            return None

    # generate_question_for_theme is renamed to generate_question_for_consolidated_theme
    async def generate_question_for_consolidated_theme(
        self, 
        theme_info: Dict[str, Any] 
    ) -> Optional[GeneratedQuestion]:
        
        # *** KEY CHANGE: Use "theme_name" which should be populated from "theme_name_en" ***
        theme_name = theme_info.get("theme_name") 
        if not theme_name: # Critical check
             print(f"[QG_SERVICE ERROR /gen_q_consolidated] Critical: 'theme_name' is missing in theme_info. Data: {str(theme_info)[:500]}")
             return None

        theme_description = theme_info.get("description_en", "")
        # *** KEY CHANGE: Use "keywords_en" from the processed theme_info ***
        theme_keywords = theme_info.get("keywords_en", "") 
        # *** KEY CHANGE: Use "constituent_entity_ids" and adjust query/logic if needed ***
        constituent_entity_ids = theme_info.get("constituent_entity_ids", [])
        source_standards_involved = theme_info.get("source_standard_document_ids", []) # Corrected key
        
        # Log for debugging the received theme_info
        # print(f"[QG_SERVICE DEBUG /gen_q_consolidated] Received theme_info: theme_name='{theme_name}', entity_ids_count={len(constituent_entity_ids)}")

        chunk_texts_list = []
        retrieved_chunks_count = 0
        valuable_chunks_count = 0
        MIN_VALUABLE_CHUNK_LENGTH = 30 
        LOW_VALUE_SECTION_TYPES = ["header", "footer", "title", "page_number_text_unstructured", "figurecaption", "tablecaption", "bibliography", "index"]

        # This check now uses constituent_entity_ids
        if not theme_name or not constituent_entity_ids: # constituent_entity_ids are crucial
            print(f"[QG_SERVICE ERROR /gen_q_consolidated] Insufficient theme_info for theme: '{theme_name}'. Missing name or constituent_entity_ids.")
            return None
        
        if constituent_entity_ids:
            # OPTION 1: Query StandardChunks linked to these __Entity__ nodes
            # This query assumes a :CONTAINS_ENTITY relationship from StandardChunk to __Entity__
            # You might need to adjust this based on how LLMGraphTransformer creates links.
            # If __Entity__ nodes have their original chunk_id in a property, that's another way.
            # Let's assume LLMGraphTransformer's output (GraphDocument) had gd.source.metadata['chunk_id']
            # and you linked StandardChunk to __Entity__ (node_in_gd.id) via :CONTAINS_ENTITY
            # The `constituent_entity_ids` are IDs of __Entity__ nodes.
            
            # We need to get the CHUNKS that CONTAIN these entities.
            # This might mean multiple chunks if an entity appears in several,
            # or if multiple entities in a community come from different chunks.
            
            # Query to get StandardChunks that are linked to the entities in this theme's community
            # We should aim to get the original chunk_id from the __Entity__ node if it was stored there
            # or traverse back. Let's assume the __Entity__ node has a direct link or property to its source StandardChunk.
            # For simplicity, IF constituent_entity_ids ARE ACTUALLY StandardChunk IDs from a previous step, the old query works.
            # BUT, based on `identify_consolidated_themes_from_kg`, constituent_entity_ids ARE __Entity__ IDs.
            
            # Let's refine the query to get chunks associated with these entities.
            # This requires knowing how StandardChunk and __Entity__ are linked.
            # Assuming `(sc:StandardChunk)-[:CONTAINS_ENTITY]->(e:__Entity__)`
            
            query = """
            UNWIND $entity_ids AS target_entity_id
            MATCH (e:__Entity__ {id: target_entity_id})
            MATCH (sc:StandardChunk)-[:CONTAINS_ENTITY]->(e) // หรือ (e)<-[:CONTAINS_ENTITY]-(sc) ขึ้นอยู่กับทิศทาง
            WITH DISTINCT sc // เอา chunk ที่ไม่ซ้ำกัน
            // OPTIONAL: เพิ่มการเรียงลำดับถ้าต้องการ chunk ที่ "สำคัญ" ก่อน
            // WITH sc, COALESCE(sc.doc_id, "z") AS doc_order, COALESCE(sc.sequence_in_document, 99999) AS seq_order
            // ORDER BY doc_order, seq_order
            RETURN sc.text AS text, 
                sc.original_section_id AS original_section, 
                sc.doc_id AS source_doc, 
                sc.page_number AS page_number
            LIMIT 20 // จำกัดจำนวน chunk ที่ดึงมาเบื้องต้น
            """
            loop = asyncio.get_running_loop()
            try:
                query_callable = functools.partial(
                    self.neo4j_service.graph.query, 
                    query, 
                    params={'entity_ids': constituent_entity_ids} # Pass entity IDs
                )
                chunk_results = await loop.run_in_executor(None, query_callable)

                if chunk_results:
                    retrieved_chunks_count = len(chunk_results)
                    for res_idx, res in enumerate(chunk_results):
                        source_doc_val = res.get('source_doc', 'Unknown Document')
                        original_section_val = res.get('original_section', 'N/A')
                        page_number_val = str(res.get('page_number', 'N/A'))
                        text_val = res.get('text', '').strip()

                        if not text_val or len(text_val) < MIN_VALUABLE_CHUNK_LENGTH:
                            continue
                        current_section_type_lower = str(original_section_val).lower()
                        if any(low_val_type in current_section_type_lower for low_val_type in LOW_VALUE_SECTION_TYPES):
                            if len(text_val.split()) < 5:
                                continue
                        
                        source_str = f"Doc='{source_doc_val}', Sec/Page='{original_section_val}/{page_number_val}'"
                        # Limit the number of chunks actually added to the prompt context
                        if valuable_chunks_count < 5: # Only take top 5 most relevant (after ordering)
                            chunk_texts_list.append(f"CONTEXT_CHUNK {valuable_chunks_count+1} (Source: {source_str}):\n{text_val}\n---\n")
                            valuable_chunks_count += 1
                        else:
                            break # Stop adding chunks if we have enough
                    
                    print(f"[QG_SERVICE INFO] Retrieved {retrieved_chunks_count} raw chunks, selected {valuable_chunks_count} valuable chunks for theme '{theme_name}' context.")
                else:
                     print(f"[QG_SERVICE WARNING] No chunks retrieved from DB for entities in theme '{theme_name}'")
            except Exception as e_ctx:
                print(f"[QG_SERVICE ERROR] Error retrieving or filtering chunk context for theme '{theme_name}': {e_ctx}")
                traceback.print_exc()
        
        # MAX_CONTEXT_LENGTH (for chunk context) - this should be smaller than the overall LLM limit
        # to leave room for graph context and other prompt elements.
        MAX_CHUNK_CONTEXT_CHARS = 15000 # Example: ~3-4k tokens from chunks
        context_from_chunks = "".join(chunk_texts_list)
        if len(context_from_chunks) > MAX_CHUNK_CONTEXT_CHARS:
            print(f"[QG_SERVICE WARNING] Chunk Context for theme '{theme_name}' is too long ({len(context_from_chunks)} chars). Truncating to {MAX_CHUNK_CONTEXT_CHARS}.")
            context_from_chunks = context_from_chunks[:MAX_CHUNK_CONTEXT_CHARS] + "\n... [CHUNK CONTEXT TRUNCATED]"
        
        if not context_from_chunks.strip():
            context_from_chunks = "No specific content chunks available for this theme. Focus on the theme name and description."

        graph_structure_context_str = "Graph structure context was not retrieved or is unavailable for this theme." 
        if theme_keywords or theme_name:
            try:
                # *** KEY CHANGE: Use more hops for graph context and potentially more central entities ***
                graph_context_result = await self.neo4j_service.get_graph_context_for_theme_chunks_v2(
                    theme_name=theme_name,
                    theme_keywords=theme_keywords,
                    max_central_entities=3,  # Increase to get a slightly broader seed
                    max_hops_for_central_entity=3, # *** INCREASED TO 2 HOPS ***
                    max_relations_to_collect_per_central_entity=7, # Increase slightly
                    max_total_context_items_str_len=4000 # Allow slightly more for graph context
                )
                if graph_context_result:
                    graph_structure_context_str = graph_context_result
                # print(f"[QG_SERVICE INFO] Graph structure context (v2) for theme '{theme_name}': {graph_structure_context_str[:300]}...")
            except Exception as e_graph_ctx:
                print(f"[QG_SERVICE ERROR] Failed to get graph structure context (v2) for theme '{theme_name}': {e_graph_ctx}")
                # traceback.print_exc() # Keep this for debugging

        # Prompt is fine, ensure variables passed to .format() are correct
        prompt_template_str = """
        You are an expert ESG consultant assisting an **industrial packaging factory** in preparing its **Sustainability Report**.
        Your task is to formulate a set of 2-5 specific, actionable, and data-driven sub-questions for the given consolidated ESG theme. These sub-questions are crucial for gathering the precise information needed for comprehensive sustainability reporting.
        
        For EACH sub-question:
        - It should be a direct question that elicits factual data, policies, processes, or performance metrics.
        - If evident from the provided context (either Standard Documents or Knowledge Graph), specify its most direct source standard, entity, or clause/section using the format (Source: DocumentName - SectionInfo OR Source: KG - EntityName). If the source is not clear for a sub-question, omit this source annotation.

        Consolidated ESG Theme: "{theme_name}"
        Theme Description: {theme_description}
        Keywords for this theme: {theme_keywords}
        Relevant Source Standards involved (if known from theme generation): {source_standards_list_str}

        Context from Knowledge Graph (Key Entities and their Relationships - Prioritize this for understanding connections):
        ---CONTEXT (GRAPH STRUCTURE) START---
        {graph_structure_context_str} 
        ---CONTEXT (GRAPH STRUCTURE) END---

        Context from Standard Documents (Use for specific details, definitions, or direct source attribution):
        ---CONTEXT (STANDARD DOCUMENTS) START---
        {context_from_chunks}
        ---CONTEXT END---

        Instructions for Output:
        Return ONLY a single, valid JSON object with these exact keys:
        - "question_text_en": A string containing ONLY the sub-questions, each numbered and on a new line (e.g., "1. Sub-question 1 text... (Source: DocName - SectionX OR Source: KG - EntityA)\\n2. Sub-question 2 text..."). DO NOT include a primary question.
        - "category": The overall ESG category for this theme ('E', 'S', or 'G').
        - "theme": Must be exactly "{theme_name}".
        - "detailed_source_info_for_subquestions": A brief textual summary of the source standards and clauses explicitly identified for the sub-questions (e.g., "Sub-Q1 based on GRI X.Y, Sub-Q2 on ISO Z:A.B, Sub-Q3 from KG Entity 'Climate Risk'"). This is for internal documentation.

        Example for "question_text_en" (containing only sub-questions):
        "1. What were the total direct (Scope 1) greenhouse gas (GHG) emissions in the reporting period, in metric tons of CO2 equivalent? (Source: GRI 305-1 - Direct GHG Emissions)\\n2. What methodology was used to calculate these Scope 1 GHG emissions? (Source: GRI 305-1 - Calculation Methodology)\\n3. What were the total indirect (Scope 2) GHG emissions from purchased electricity, heat, or steam in the reporting period? (Source: GRI 305-2 - Indirect GHG Emissions)"
        
        If context is sparse, formulate broader sub-questions suitable for initial data gathering for sustainability reporting based on the theme name, description, and keywords.
        Focus on questions that help disclose performance and management approaches.
        Prioritize information from the Knowledge Graph context if available, then use Standard Document context for specifics.
        """
        prompt = PromptTemplate.from_template(prompt_template_str)
        
        formatted_prompt = prompt.format(
            theme_name=theme_name,
            theme_description=theme_description if theme_description else "Not specified.",
            theme_keywords=theme_keywords if theme_keywords else "Not specified.",
            source_standards_list_str=", ".join(source_standards_involved) if source_standards_involved else "Various ESG standards.",
            context_from_chunks=context_from_chunks, # Limited and filtered chunks
            graph_structure_context_str=graph_structure_context_str # Potentially richer graph context
        )
        
        llm_output = "" 
        json_str = "" 
        try:
            # print(f"[QG_SERVICE LOG /gen_q_consolidated] Generating question for consolidated theme: {theme_name}")
            # print(f"DEBUG Prompt for {theme_name}:\n{formatted_prompt}\n------------------")
            response = await self.qg_llm.ainvoke(formatted_prompt)
            llm_output = response.content.strip()
            
            match = re.search(r"```json\s*(\{[\s\S]*?\})\s*```", llm_output, re.DOTALL | re.MULTILINE)
            if match: json_str = match.group(1)
            else:
                first_brace = llm_output.find('{'); last_brace = llm_output.rfind('}')
                if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
                    json_str = llm_output[first_brace : last_brace+1]
                else:
                    print(f"[QG_SERVICE ERROR /gen_q_consolidated] LLM output not valid JSON for '{theme_name}': {llm_output}")
                    return None
            
            question_data = json.loads(json_str)
            
            if not all(k in question_data for k in ["question_text_en", "category", "theme"]):
                print(f"[QG_SERVICE ERROR /gen_q_consolidated] LLM JSON output missing required keys for theme '{theme_name}'. Got: {question_data.keys()}")
                return None

            final_theme_name = theme_name 
            final_category = str(question_data.get("category", "G")).upper() # Ensure string
            if final_category not in ['E', 'S', 'G']: final_category = "G"
            
            detailed_source_info = question_data.get("detailed_source_info_for_subquestions")

            generated_q_obj = GeneratedQuestion(
                 question_text_en=question_data.get("question_text_en", "Error: Question text missing from LLM"),
                 category=final_category,
                 theme=final_theme_name,
                 additional_info={"detailed_source_info_for_subquestions": detailed_source_info} if detailed_source_info else None
            )
            return generated_q_obj
        except json.JSONDecodeError as e:
            print(f"[QG_SERVICE ERROR /gen_q_consolidated] JSON parse error for '{theme_name}': {e}. Cleaned Str: {json_str}. Original LLM Output: {llm_output}")
            return None
        except Exception as e:
            print(f"[QG_SERVICE ERROR /gen_q_consolidated] Question formulation error for '{theme_name}': {e}")
            traceback.print_exc()
            return None


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
    
    # evolve_and_store_questions: The full version provided in the previous message.
    # Ensure it uses the updated `identify_consolidated_themes_from_kg` and
    # `generate_question_for_consolidated_theme`.
    # Also ensure it correctly maps data from the theme_info dict (returned by 
    # identify_consolidated_themes_from_kg) to the new fields in the ESGQuestion MongoDB model.
    async def evolve_and_store_questions(self, uploaded_file_content_bytes: Optional[bytes] = None) -> List[GeneratedQuestion]:
        print(f"[QG_SERVICE LOG] Orchestrating Question AI Evolution (Consolidated Themes). File uploaded: {'Yes' if uploaded_file_content_bytes else 'No'}")
        current_time_utc = datetime.now(timezone.utc)
        initial_neo4j_empty_check_result = await self.neo4j_service.is_graph_substantially_empty()
        print(f"[QG_SERVICE INFO] Neo4j graph substantially empty check (initial): {initial_neo4j_empty_check_result}")

        active_questions_in_db: List[ESGQuestion] = await ESGQuestion.find(ESGQuestion.is_active == True).to_list()
        active_themes_map: Dict[str, ESGQuestion] = {q.theme: q for q in active_questions_in_db}
        print(f"[QG_SERVICE INFO] Found {len(active_questions_in_db)} active questions in DB, covering {len(active_themes_map)} unique themes.")

        process_scope: str = PROCESS_SCOPE_NO_ACTION_DEFAULT
        themes_info_for_question_generation: List[Dict[str, Any]] = [] 

        if initial_neo4j_empty_check_result:
            process_scope = PROCESS_SCOPE_KG_INITIAL_FULL_SCAN_FROM_CONTENT
            # ... (rest of scope determination logic remains the same) ...
            if active_questions_in_db: 
                print(f"[QG_SERVICE INFO] Deactivating ALL {len(active_questions_in_db)} existing questions as KG is considered empty for new generation.")
                for q_to_deactivate in active_questions_in_db:
                    await ESGQuestion.find_one(ESGQuestion.id == q_to_deactivate.id).update(
                        {"$set": {"is_active": False, "updated_at": current_time_utc}}
                    )
                active_questions_in_db = [] 
                active_themes_map.clear()
            themes_info_for_question_generation = await self.identify_consolidated_themes_from_kg()
        elif uploaded_file_content_bytes: 
            process_scope = PROCESS_SCOPE_KG_UPDATED_FULL_SCAN_FROM_CONTENT
            themes_info_for_question_generation = await self.identify_consolidated_themes_from_kg(
                existing_active_theme_names=list(active_themes_map.keys())
            )
        elif not uploaded_file_content_bytes: 
            if not active_questions_in_db and not initial_neo4j_empty_check_result:
                process_scope = PROCESS_SCOPE_KG_FULL_SCAN_DB_EMPTY_FROM_CONTENT
                themes_info_for_question_generation = await self.identify_consolidated_themes_from_kg()
            elif active_questions_in_db: 
                process_scope = PROCESS_SCOPE_KG_SCHEDULED_REFRESH_FROM_CONTENT
                themes_info_for_question_generation = await self.identify_consolidated_themes_from_kg(
                    existing_active_theme_names=list(active_themes_map.keys())
                )
            else: 
                 process_scope = PROCESS_SCOPE_NO_ACTION_KG_AND_DB_EMPTY_ON_SCHEDULED_RUN
        else:
            process_scope = PROCESS_SCOPE_NO_ACTION_UNEXPECTED_STATE
        
        print(f"[QG_SERVICE INFO] Final Determined process_scope: {process_scope}")
        
        if not themes_info_for_question_generation:
            print(f"[QG_SERVICE WARNING] No themes were identified from KG content for scope '{process_scope}'.")
            if process_scope != PROCESS_SCOPE_KG_INITIAL_FULL_SCAN_FROM_CONTENT and active_questions_in_db :
                 print("[QG_SERVICE WARNING] Returning existing active questions as fallback.")
                 return [GeneratedQuestion(**q.model_dump(exclude_none=True, exclude={'id','_id', 'revision_id', 'created_at', 'generated_at', 'source_standard_document_references'})) for q in active_questions_in_db]
            return []

        print(f"[QG_SERVICE INFO] Generating English questions for {len(themes_info_for_question_generation)} consolidated themes.")
        english_question_tasks = [
            self.generate_question_for_consolidated_theme(theme_info)
            for theme_info in themes_info_for_question_generation # themes_info_for_question_generation SHOULD be clean now
        ]
        generated_english_q_results_with_exc = await asyncio.gather(*english_question_tasks, return_exceptions=True)
        
        successful_english_q_objs_map_to_theme_info: Dict[int, Dict[str, Any]] = {}
        candidate_bilingual_pydantic_questions: List[GeneratedQuestion] = []
        temp_english_questions_for_translation: List[GeneratedQuestion] = []

        for i, result_item in enumerate(generated_english_q_results_with_exc):
            original_theme_info = themes_info_for_question_generation[i] # This should be a valid dict
            
            # *** CRITICAL CHECK: Ensure original_theme_info has "theme_name" before proceeding ***
            if not isinstance(original_theme_info, dict) or not original_theme_info.get("theme_name"):
                print(f"[QG_SERVICE ERROR] Skipping result item {i} because its corresponding original_theme_info is invalid or missing 'theme_name'.")
                continue

            if isinstance(result_item, GeneratedQuestion):
                temp_english_questions_for_translation.append(result_item)
                successful_english_q_objs_map_to_theme_info[id(result_item)] = original_theme_info
            elif isinstance(result_item, Exception):
                theme_name_for_err = original_theme_info.get("theme_name", f"Unknown Theme (index {i})")
                print(f"[QG_SERVICE ERROR] English question generation task failed for theme '{theme_name_for_err}': {result_item}")
                # traceback.print_exc()
        
        if temp_english_questions_for_translation:
            print(f"[QG_SERVICE INFO] Translating {len(temp_english_questions_for_translation)} English questions to Thai...")
            translation_tasks = [self.translate_text_to_thai(q.question_text_en) for q in temp_english_questions_for_translation]
            translated_th_texts_with_exc = await asyncio.gather(*translation_tasks, return_exceptions=True)

            for i, en_question_obj in enumerate(temp_english_questions_for_translation):
                th_text_result = translated_th_texts_with_exc[i]
                question_text_th = th_text_result if isinstance(th_text_result, str) else None
                if isinstance(th_text_result, Exception):
                    print(f"[QG_SERVICE ERROR] Translation failed for EN q '{en_question_obj.question_text_en[:50]}...': {th_text_result}")
                
                pydantic_q = GeneratedQuestion(
                    question_text_en=en_question_obj.question_text_en,
                    question_text_th=question_text_th,
                    category=en_question_obj.category,
                    theme=en_question_obj.theme,
                    additional_info=en_question_obj.additional_info # Preserve additional_info
                )
                candidate_bilingual_pydantic_questions.append(pydantic_q)
        
        print(f"[QG_SERVICE INFO] Prepared {len(candidate_bilingual_pydantic_questions)} candidate bilingual questions for DB evolution.")

        all_db_questions_docs: List[ESGQuestion] = await ESGQuestion.find_all().to_list()
        all_historical_questions_by_theme: Dict[str, List[ESGQuestion]] = {}
        for q_doc_hist in all_db_questions_docs:
            all_historical_questions_by_theme.setdefault(q_doc_hist.theme, []).append(q_doc_hist)
        for theme_name_hist_sort in all_historical_questions_by_theme:
            all_historical_questions_by_theme[theme_name_hist_sort].sort(key=lambda q: q.version, reverse=True)
        
        print(f"[QG_SERVICE INFO] Fetched and prepared {len(all_db_questions_docs)} historical questions from DB, grouped into {len(all_historical_questions_by_theme)} themes.")

        # --- DB Operations Loop ---
        for pydantic_candidate_q in candidate_bilingual_pydantic_questions:
            theme_name_cand = pydantic_candidate_q.theme

            original_theme_info_for_db_op: Optional[Dict[str, Any]] = None
            # Find the original_theme_info using the object ID map
            # This relies on pydantic_candidate_q being derived from an object in temp_english_questions_for_translation
            # We need a more robust way to link pydantic_candidate_q back to original_theme_info
            # For now, we assume pydantic_candidate_q.theme and .question_text_en are sufficient for this lookup
            # in successful_english_q_objs_map_to_theme_info (but the key is object ID)

            # Let's try to find the original English GeneratedQuestion object
            # that corresponds to this pydantic_candidate_q
            found_original_eng_q_obj = None
            for eng_q_obj_ref in temp_english_questions_for_translation:
                 if eng_q_obj_ref.theme == pydantic_candidate_q.theme and \
                    eng_q_obj_ref.question_text_en == pydantic_candidate_q.question_text_en and \
                    eng_q_obj_ref.category == pydantic_candidate_q.category: # Add category for better match
                    found_original_eng_q_obj = eng_q_obj_ref
                    break
            
            if found_original_eng_q_obj and id(found_original_eng_q_obj) in successful_english_q_objs_map_to_theme_info:
                original_theme_info_for_db_op = successful_english_q_objs_map_to_theme_info[id(found_original_eng_q_obj)]
            
            if not original_theme_info_for_db_op:
                print(f"[QG_SERVICE WARNING] Critical: Could not map Pydantic candidate Q for theme '{theme_name_cand}' back to its original full theme_info. Skipping DB ops for this question.")
                continue
            
            # Now original_theme_info_for_db_op is the dict from identify_consolidated_themes_from_kg
            # which contains "theme_name", "theme_name_en", "description_en", "keywords_en", etc.

            historical_q_versions_for_theme = all_historical_questions_by_theme.get(theme_name_cand, [])
            latest_db_q_for_theme = historical_q_versions_for_theme[0] if historical_q_versions_for_theme else None
            
            detailed_source_info_from_additional = None
            if pydantic_candidate_q.additional_info: 
                detailed_source_info_from_additional = pydantic_candidate_q.additional_info.get("detailed_source_info_for_subquestions")

            # *** KEY MAPPING FOR DB ***
            db_question_data = {
                "question_text_en": pydantic_candidate_q.question_text_en,
                "question_text_th": pydantic_candidate_q.question_text_th,
                "category": pydantic_candidate_q.category,
                "theme": theme_name_cand, # This is theme_name_en / primary theme identifier
                "keywords": original_theme_info_for_db_op.get("keywords_en", ""), # Use "keywords_en"
                "theme_description_en": original_theme_info_for_db_op.get("description_en"),
                "theme_description_th": None, # Placeholder, TODO: translate description_en if needed
                "dimension": original_theme_info_for_db_op.get("dimension"),
                "source_document_references": original_theme_info_for_db_op.get("source_standard_document_ids", []), # Correct key
                "constituent_entity_ids": original_theme_info_for_db_op.get("constituent_entity_ids", []), # Correct key
                "detailed_source_info_for_subquestions": detailed_source_info_from_additional,
                "updated_at": current_time_utc,
                "is_active": True,
                "generation_method": original_theme_info_for_db_op.get("generation_method", "unknown"),
                "metadata_extras": { # Example for storing other info from theme generation
                    "_community_id_source": original_theme_info_for_db_op.get("_community_id_source")
                }
            }
            # TODO: Translate theme_description_en to theme_description_th if required by your ESGQuestion model
            if db_question_data["theme_description_en"]:
                 translated_desc_th = await self.translate_text_to_thai(db_question_data["theme_description_en"])
                 if translated_desc_th:
                     db_question_data["theme_description_th"] = translated_desc_th


            if latest_db_q_for_theme: 
                is_content_similar = await self.are_questions_substantially_similar(
                    pydantic_candidate_q.question_text_en, latest_db_q_for_theme.question_text_en
                )
                if not is_content_similar: 
                    # ... (logic for new version - seems okay) ...
                    if latest_db_q_for_theme.is_active:
                        await ESGQuestion.find_one(ESGQuestion.id == latest_db_q_for_theme.id).update(
                            {"$set": {"is_active": False, "updated_at": current_time_utc}}
                        )
                    db_question_data["version"] = latest_db_q_for_theme.version + 1
                    db_question_data["generated_at"] = current_time_utc
                    new_q_to_insert = ESGQuestion(**db_question_data)
                    try: await new_q_to_insert.insert()
                    except Exception as e_ins_v: print(f"[QG_SERVICE ERROR] DB Insert (new ver) for Q '{theme_name_cand}': {e_ins_v}")

                else: 
                    # ... (logic for updating similar - seems okay, ensure field names in fields_to_check_for_update match ESGQuestion model) ...
                    fields_to_check_for_update = [
                        "keywords", "theme_description_en", "theme_description_th", "dimension", 
                        "source_document_references", "constituent_entity_ids", 
                        "detailed_source_info_for_subquestions", "question_text_th", "category",
                        "generation_method", "metadata_extras" # Add new fields if they should be updated
                    ]
                    update_payload = {"is_active": True, "updated_at": current_time_utc}
                    needs_db_update = not latest_db_q_for_theme.is_active
                    for field_key in fields_to_check_for_update:
                        new_value = db_question_data.get(field_key)
                        old_value = getattr(latest_db_q_for_theme, field_key, None)
                        if isinstance(new_value, list) and isinstance(old_value, list):
                            if sorted(new_value) != sorted(old_value):
                                update_payload[field_key] = new_value
                                needs_db_update = True
                        elif isinstance(new_value, dict) and isinstance(old_value, dict): # For metadata_extras
                            if new_value != old_value:
                                update_payload[field_key] = new_value
                                needs_db_update = True
                        elif new_value != old_value:
                            update_payload[field_key] = new_value
                            needs_db_update = True
                    
                    if needs_db_update:
                        await ESGQuestion.find_one(ESGQuestion.id == latest_db_q_for_theme.id).update({"$set": update_payload})
                        print(f"[QG_SERVICE INFO] Updated existing similar question for theme '{theme_name_cand}'.")
            else: 
                # ... (logic for new theme - seems okay) ...
                print(f"[QG_SERVICE INFO] Theme '{theme_name_cand}' is NEW to DB. Inserting as v1.")
                db_question_data["version"] = 1
                db_question_data["generated_at"] = current_time_utc
                new_q_to_insert = ESGQuestion(**db_question_data)
                try: await new_q_to_insert.insert()
                except Exception as e_ins_new_t: print(f"[QG_SERVICE ERROR] DB Insert (new theme) for Q '{theme_name_cand}': {e_ins_new_t}")

        # --- Deactivate themes logic ---
        # *** KEY CHANGE: Ensure current_run_generated_theme_names_set uses "theme_name" from the dicts in themes_info_for_question_generation ***
        # themes_info_for_question_generation is the list of dicts from identify_consolidated_themes_from_kg
        # Each dict should have "theme_name" as the primary key.

        # This set comprehension MUST use a key that is guaranteed to exist in all dicts within themes_info_for_question_generation
        # Based on _process_single_cluster_with_llm, this should be "theme_name" (which is theme_name_en)
        current_run_generated_theme_names_set = {
            theme_d.get("theme_name") 
            for theme_d in themes_info_for_question_generation 
            if isinstance(theme_d, dict) and theme_d.get("theme_name") # Robust check
        }

        should_deactivate_obsolete_themes = process_scope in [
            PROCESS_SCOPE_KG_INITIAL_FULL_SCAN_FROM_CONTENT, 
            PROCESS_SCOPE_KG_UPDATED_FULL_SCAN_FROM_CONTENT, 
            PROCESS_SCOPE_KG_FULL_SCAN_DB_EMPTY_FROM_CONTENT, 
            PROCESS_SCOPE_KG_SCHEDULED_REFRESH_FROM_CONTENT
        ]

        if should_deactivate_obsolete_themes:
            for active_theme_name_loop, active_q_doc_loop in active_themes_map.items():
                if active_theme_name_loop not in current_run_generated_theme_names_set and active_q_doc_loop.is_active:
                    print(f"[QG_SERVICE INFO] Theme '{active_theme_name_loop}' (v{active_q_doc_loop.version}) no longer identified from KG. Deactivating.")
                    await ESGQuestion.find_one(ESGQuestion.id == active_q_doc_loop.id).update(
                        {"$set": {"is_active": False, "updated_at": current_time_utc}}
                    )

        final_active_db_questions: List[ESGQuestion] = await ESGQuestion.find(ESGQuestion.is_active == True).to_list()
        final_pydantic_questions_for_api: List[GeneratedQuestion] = [
            GeneratedQuestion(
                question_text_en=q.question_text_en,
                question_text_th=q.question_text_th,
                category=q.category,
                theme=q.theme, # This 'theme' field in GeneratedQuestion matches ESGQuestion.theme
                 # If GeneratedQuestion has additional_info and you want to populate it from ESGQuestion:
                additional_info={"detailed_source_info_for_subquestions": q.detailed_source_info_for_subquestions} if q.detailed_source_info_for_subquestions else None

            ) for q in final_active_db_questions
        ]
            
        print(f"[QG_SERVICE LOG] Evolution complete. Total {len(final_pydantic_questions_for_api)} bilingual questions are effectively active in DB.")
        return final_pydantic_questions_for_api