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


    async def identify_consolidated_themes_from_kg(
        self,
        existing_active_theme_names: Optional[List[str]] = None, # Can be used to guide LLM or for diffing
        min_cluster_size_hdbscan: int = 7, # Parameter for HDBSCAN
        min_samples_hdbscan: Optional[int] = None # Parameter for HDBSCAN
    ) -> List[Dict[str, Any]]:
        print("[QG_SERVICE LOG] Starting Consolidated Theme Identification from KG content...")
        
        all_chunks_data = await self.neo4j_service.get_all_standard_chunks_for_theme_generation()
        if not all_chunks_data or len(all_chunks_data) < min_cluster_size_hdbscan : # Need enough chunks
            print(f"[QG_SERVICE WARNING] Not enough StandardChunks ({len(all_chunks_data)}) available from Neo4j to identify themes (min required: {min_cluster_size_hdbscan}).")
            return []

        print(f"[QG_SERVICE INFO] Retrieved {len(all_chunks_data)} chunks for theme identification.")

        embeddings_array = np.array([chunk['embedding'] for chunk in all_chunks_data if chunk.get('embedding')])
        if embeddings_array.ndim == 1: # Handle case where only one embedding might be returned (though unlikely with check above)
            if len(all_chunks_data) >=1 : # If only one chunk, it's its own theme
                 print("[QG_SERVICE WARNING] Only one chunk with embedding found. Treating as a single theme cluster.")
                 # Simulate a single cluster for this one chunk
                 cluster_labels = np.array([0])
                 # Ensure embeddings_array is 2D for hdbscan if it proceeds (though it won't with 1 sample)
                 if embeddings_array.size > 0 and embeddings_array.ndim == 1:
                     embeddings_array = embeddings_array.reshape(1, -1)

            else: # No embeddings
                print("[QG_SERVICE WARNING] No embeddings found in retrieved chunks.")
                return []
        elif embeddings_array.shape[0] < min_cluster_size_hdbscan:
             print(f"[QG_SERVICE WARNING] Number of chunks with embeddings ({embeddings_array.shape[0]}) is less than min_cluster_size_hdbscan ({min_cluster_size_hdbscan}). Cannot perform robust clustering. Treating each as a potential small theme or skipping.")
             # For now, let's skip if too few for clustering. Or handle as individual items if desired.
             # Option: Process them individually by LLM without clustering.
             # For simplicity in this example, returning empty. This needs refinement.
             return []


        print(f"[QG_SERVICE INFO] Performing HDBSCAN clustering on {embeddings_array.shape[0]} chunk embeddings...")
        loop = asyncio.get_running_loop()
        try:
            clusterer = HDBSCAN(
                min_cluster_size=min_cluster_size_hdbscan, 
                min_samples=min_samples_hdbscan, # If None, defaults to min_cluster_size
                metric='cosine', # Cosine distance is often good for text embeddings
                # allow_single_cluster=True # Add if you want to allow a single large cluster
            )
            # HDBSCAN fit_predict is synchronous
            cluster_labels = await loop.run_in_executor(None, clusterer.fit_predict, embeddings_array)
            num_identified_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
            num_outliers = np.sum(cluster_labels == -1)
            print(f"[QG_SERVICE INFO] HDBSCAN found {num_identified_clusters} clusters and {num_outliers} outliers.")

        except Exception as e_cluster:
            print(f"[QG_SERVICE ERROR] Clustering error: {e_cluster}")
            traceback.print_exc()
            return []

        identified_themes_from_kg: List[Dict[str, Any]] = []
        
        # Group chunks by cluster label
        clustered_chunks: Dict[int, List[Dict[str, Any]]] = {}
        for i, label in enumerate(cluster_labels):
            if label != -1: # Exclude outliers for now
                clustered_chunks.setdefault(label, []).append(all_chunks_data[i])
        
        # Process each cluster with LLM
        llm_tasks = []
        for cluster_id, chunks_in_this_cluster in clustered_chunks.items():
            if not chunks_in_this_cluster: continue

            # Create context for LLM from chunks in this cluster
            context_for_llm = ""
            char_count = 0
            max_chars_for_context = 1000000 # Adjust based on LLM context window and token limits
            num_chunks_for_context = 0
            MAX_CHUNKS_SAMPLE = 7 # Take up to 7 chunks for context

            temp_context_parts = []
            for chunk_data in chunks_in_this_cluster:
                if num_chunks_for_context < MAX_CHUNKS_SAMPLE:
                    chunk_text_sample = chunk_data['text'][:int(max_chars_for_context / MAX_CHUNKS_SAMPLE)] # Distribute char limit
                    if char_count + len(chunk_text_sample) <= max_chars_for_context:
                        source_info = f"(From Doc: {chunk_data.get('source_document_doc_id', 'Unknown')}, Page: {chunk_data.get('page_number', 'N/A')}, Orig.Section: {chunk_data.get('original_section_id', 'N/A')})"
                        temp_context_parts.append(f"SAMPLE CHUNK {source_info}:\n{chunk_text_sample}\n---\n")
                        char_count += len(chunk_text_sample) + len(source_info) + 20 # Approx
                        num_chunks_for_context +=1
                    else:
                        break # Stop if adding next chunk exceeds max_chars
                else:
                    break # Stop if max_chunks_sample reached
            context_for_llm = "".join(temp_context_parts)
            
            if not context_for_llm and chunks_in_this_cluster: # Fallback if all chunks were too long / yielded empty
                 first_chunk_text = chunks_in_this_cluster[0]['text']
                 context_for_llm = first_chunk_text[:max_chars_for_context]


            source_doc_names_in_cluster = sorted(list(set(c['source_document_doc_id'] for c in chunks_in_this_cluster if c.get('source_document_doc_id'))))

            prompt_template_str = """
            You are an expert ESG analyst. A group of semantically similar text chunks has been extracted from various ESG standard documents. Your task is to define a consolidated ESG reporting theme based on these chunks, tailored for the **industrial packaging sector**.

            Representative text chunks from this group:
            ---REPRESENTATIVE CHUNKS START---
            {representative_chunk_content}
            ---REPRESENTATIVE CHUNKS END---

            These chunks originate from standards like: {source_standards_list_str}

            Based on this information:
            1.  Propose a concise and descriptive **Theme Name** (in English) for this group.
            2.  Provide a brief **Theme Description** (in English, 1-2 sentences).
            3.  Determine the most relevant **ESG Dimension** ('E', 'S', or 'G').
            4.  Suggest 3-5 relevant **Keywords** (in English, comma-separated).

            Output ONLY a single, valid JSON object with keys: "theme_name_en", "description_en", "dimension", "keywords_en".
            Example:
            {{"theme_name_en": "Sustainable Water Management", "description_en": "Covers responsible water use, treatment, and risk mitigation in packaging facilities.", "dimension": "E", "keywords_en": "water consumption, wastewater, water recycling, water stewardship"}}
            """ # Simplified JSON output request for easier parsing
            
            prompt = PromptTemplate.from_template(prompt_template_str)
            formatted_prompt = prompt.format(
                representative_chunk_content=context_for_llm if context_for_llm else "Generic ESG context.",
                source_standards_list_str=", ".join(source_doc_names_in_cluster[:3]) if source_doc_names_in_cluster else "various ESG standards"
            )
            
            # Add task to call LLM for this cluster
            # We also pass along constituent_chunk_ids and source_standards for this cluster
            # to be added to the final theme dict after LLM response.
            llm_tasks.append(
                self._process_single_cluster_with_llm(
                    formatted_prompt,
                    [c['chunk_id'] for c in chunks_in_this_cluster],
                    source_doc_names_in_cluster,
                    cluster_id # For logging
                )
            )

        # Execute all LLM calls concurrently
        if llm_tasks:
            llm_results_with_exceptions = await asyncio.gather(*llm_tasks, return_exceptions=True)
            for result_item in llm_results_with_exceptions:
                if isinstance(result_item, dict) and result_item.get("theme_name_en"): # Check if it's a valid theme dict
                    identified_themes_from_kg.append(result_item)
                elif isinstance(result_item, Exception):
                    print(f"[QG_SERVICE ERROR] LLM processing for a cluster failed: {result_item}")
                    traceback.print_exc()
        
        # TODO: Handle outliers (chunks with label -1) if desired.
        # For now, they are ignored. Could try to assign to nearest cluster or process separately.

        print(f"[QG_SERVICE LOG] Identified {len(identified_themes_from_kg)} consolidated themes from KG content after LLM processing.")
        return identified_themes_from_kg

    async def _process_single_cluster_with_llm(
        self, 
        formatted_prompt: str, 
        chunk_ids_in_cluster: List[str], 
        source_docs_in_cluster: List[str],
        cluster_id_for_log: int
    ) -> Optional[Dict[str, Any]]:
        """Helper to process one cluster with LLM and structure the output."""
        try:
            # print(f"[QG_SERVICE DEBUG / Cluster {cluster_id_for_log}] Sending prompt to LLM (first 200 chars): {formatted_prompt[:200]}...")
            response = await self.qg_llm.ainvoke(formatted_prompt)
            llm_output = response.content.strip()
            
            # print(f"[QG_SERVICE DEBUG / Cluster {cluster_id_for_log}] LLM Raw Output: {llm_output}")
            json_str = ""
            match = re.search(r"```json\s*(\{.*?\})\s*```", llm_output, re.DOTALL)
            if match: json_str = match.group(1)
            else:
                first_brace = llm_output.find('{'); last_brace = llm_output.rfind('}')
                if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
                    json_str = llm_output[first_brace : last_brace+1]
                else: # If it's already a valid JSON string without ```json
                    if llm_output.strip().startswith("{") and llm_output.strip().endswith("}"):
                        json_str = llm_output.strip()
                    else:
                        print(f"[QG_SERVICE ERROR / Cluster {cluster_id_for_log}] LLM output not valid JSON: {llm_output}")
                        return None
            
            llm_theme_data = json.loads(json_str)
            
            theme_name_en = llm_theme_data.get("theme_name_en")
            if not theme_name_en: 
                print(f"[QG_SERVICE WARNING / Cluster {cluster_id_for_log}] LLM did not provide 'theme_name_en'. Skipping this cluster.")
                return None

            theme_name_th = await self.translate_text_to_thai(theme_name_en)
            description_en = llm_theme_data.get("description_en", "")
            # description_th = await self.translate_text_to_thai(description_en) # Optional to translate desc

            dimension = llm_theme_data.get("dimension", "G").upper()
            if dimension not in ['E', 'S', 'G']: dimension = "G" # Default to G

            return {
                "theme_name": theme_name_en, # Use EN as primary key for now
                "theme_name_en": theme_name_en,
                "theme_name_th": theme_name_th,
                "description_en": description_en,
                # "description_th": description_th,
                "dimension": dimension,
                "keywords": llm_theme_data.get("keywords_en", ""), # String of comma-separated keywords
                "constituent_chunk_ids": chunk_ids_in_cluster,
                "source_standards": source_docs_in_cluster
            }
        except json.JSONDecodeError as e_json:
            print(f"[QG_SERVICE ERROR / Cluster {cluster_id_for_log}] Failed to parse JSON from LLM: {e_json}. Cleaned: {json_str}. Original: {llm_output}")
            return None
        except Exception as e_llm:
            print(f"[QG_SERVICE ERROR / Cluster {cluster_id_for_log}] LLM call or processing error: {e_llm}")
            traceback.print_exc()
            return None

    # generate_question_for_theme is renamed to generate_question_for_consolidated_theme
    async def generate_question_for_consolidated_theme(
        self, 
        theme_info: Dict[str, Any] # Consolidated Theme object
    ) -> Optional[GeneratedQuestion]:
        # ... (Implementation from previous message, ensure it uses theme_info correctly for context)
        theme_name = theme_info.get("theme_name") # This is theme_name_en
        theme_description = theme_info.get("description_en", "")
        theme_keywords = theme_info.get("keywords", "") 
        constituent_chunk_ids = theme_info.get("constituent_chunk_ids", [])
        source_standards_involved = theme_info.get("source_standards", [])
        chunk_texts_list = []
        retrieved_chunks_count = 0
        valuable_chunks_count = 0
        MIN_VALUABLE_CHUNK_LENGTH = 30
        LOW_VALUE_SECTION_TYPES = ["header", "footer", "title", "page_number_text_unstructured"] # Add more if needed from Unstructured output
        if not theme_name or not constituent_chunk_ids:
            print(f"[QG_SERVICE ERROR /gen_q_consolidated] Insufficient theme_info for theme: '{theme_name}'. Missing name or chunk_ids.")
            return None

        if constituent_chunk_ids:
            query = """
            UNWIND $chunk_ids AS target_chunk_id
            MATCH (sc:StandardChunk {chunk_id: target_chunk_id})
            // ใช้ COALESCE เพื่อความปลอดภัยในการ ORDER BY แม้ว่า sc.doc_id ควรจะมีค่าเสมอ
            WITH sc, 
                 COALESCE(sc.doc_id, "zz_unknown_source_for_ordering") AS source_doc_for_ordering, 
                 COALESCE(sc.sequence_in_document, 99999) AS seq_for_ordering
            RETURN sc.text as text, 
                   sc.original_section_id as original_section, 
                   sc.doc_id as source_doc, 
                   sc.page_number as page_number,
                   sc.sequence_in_document as seq
            ORDER BY source_doc_for_ordering, seq_for_ordering
            """
            loop = asyncio.get_running_loop()
            try:
                # แก้ไขการเรียก run_in_executor ตามที่เคยทำไปแล้ว
                query_callable = functools.partial(
                    self.neo4j_service.graph.query, 
                    query, 
                    params={'chunk_ids': constituent_chunk_ids} # <--- *** แก้ไขตรงนี้เป็น 'params' ***
                )
                chunk_results = await loop.run_in_executor(None, query_callable)

                if chunk_results:
                    retrieved_chunks_count = len(chunk_results)
                    for res_idx, res in enumerate(chunk_results):
                        source_doc_val = res.get('source_doc', 'Unknown Document')
                        original_section_val = res.get('original_section', 'N/A') # This is often the element type from Unstructured
                        page_number_val = str(res.get('page_number', 'N/A'))
                        text_val = res.get('text', '').strip() # .strip() to remove leading/trailing whitespace

                        # --- Filtering Logic ---
                        if not text_val or len(text_val) < MIN_VALUABLE_CHUNK_LENGTH:
                            # print(f"[QG_SERVICE DEBUG] Skipping chunk for context (too short): '{text_val[:50]}...' from {source_doc_val}")
                            continue

                        # Optional: More advanced filtering based on original_section_val (element type)
                        # For example, if 'original_section_val' is 'Header' and text_val is very short or common.
                        # This requires knowing what UnstructuredLoader typically puts in 'original_section_id'.
                        # Let's assume original_section_val might be something like "Header", "Footer", "NarrativeText"
                        current_section_type_lower = str(original_section_val).lower()
                        if any(low_val_type in current_section_type_lower for low_val_type in LOW_VALUE_SECTION_TYPES):
                            # If it's a low-value type, check if the content is generic or very short
                            # This is a heuristic and might need tuning
                            if len(text_val.split()) < 5 or text_val.lower() == "chain": # Example: less than 5 words or just "chain"
                                # print(f"[QG_SERVICE DEBUG] Skipping chunk for context (low value type and content): '{text_val[:50]}...' from {source_doc_val}")
                                continue
                        # --- End Filtering Logic ---

                        source_str = f"Doc='{source_doc_val}', Sec/Page='{original_section_val}/{page_number_val}'"
                        chunk_texts_list.append(f"CONTEXT_CHUNK {valuable_chunks_count+1} (Source: {source_str}):\n{text_val}\n---\n")
                        valuable_chunks_count += 1
                    
                    print(f"[QG_SERVICE INFO] Retrieved {retrieved_chunks_count} raw chunks, selected {valuable_chunks_count} valuable chunks for theme '{theme_name}' context.")
                else:
                     print(f"[QG_SERVICE WARNING] No chunks retrieved from DB for ids: {constituent_chunk_ids} for theme '{theme_name}'")
            except Exception as e_ctx:
                print(f"[QG_SERVICE ERROR] Error retrieving or filtering chunk context for theme '{theme_name}': {e_ctx}")
                traceback.print_exc() # Good to have traceback here
        
        MAX_CONTEXT_LENGTH = 1000000 # Gemini 1.5 Flash has large context, but keep it reasonable for performance and cost. Adjust.
        context_from_chunks = "".join(chunk_texts_list) # Already has newlines and "---"
        if len(context_from_chunks) > MAX_CONTEXT_LENGTH:
            print(f"[QG_SERVICE WARNING] Context for theme '{theme_name}' is too long ({len(context_from_chunks)} chars). Truncating to {MAX_CONTEXT_LENGTH}.")
            context_from_chunks = context_from_chunks[:MAX_CONTEXT_LENGTH] + "\n... [CONTEXT TRUNCATED]"
        
        if not context_from_chunks.strip(): # Check if it became empty after potential processing
            context_from_chunks = "No specific content chunks available for this theme. Focus on the theme name and description."

        prompt_template_str = """
        You are an expert ESG consultant assisting an **industrial packaging factory** in preparing its **Sustainability Report**.
        Your task is to formulate a set of 2-5 specific, actionable, and data-driven sub-questions for the given consolidated ESG theme. These sub-questions are crucial for gathering the precise information needed for comprehensive sustainability reporting.
        
        For EACH sub-question:
        - It should be a direct question that elicits factual data, policies, processes, or performance metrics.
        - If evident from the provided context chunks (each prefixed with its source), specify its most direct source standard and clause/section using the format (Source: DocumentName - SectionInfo). If the source is not clear for a sub-question, omit this source annotation.

        Consolidated ESG Theme: "{theme_name}"
        Theme Description: {theme_description}
        Keywords for this theme: {theme_keywords}
        Relevant Source Standards involved: {source_standards_list_str}

        Context from Standard Documents (each prefixed with its source):
        ---CONTEXT START---
        {context_from_chunks}
        ---CONTEXT END---

        Instructions for Output:
        Return ONLY a single, valid JSON object with these exact keys:
        - "question_text_en": A string containing ONLY the sub-questions, each numbered and on a new line (e.g., "1. Sub-question 1 text... (Source: DocName - SectionX)\\n2. Sub-question 2 text..."). DO NOT include a primary question.
        - "category": The overall ESG category for this theme ('E', 'S', or 'G').
        - "theme": Must be exactly "{theme_name}".
        - "detailed_source_info_for_subquestions": A brief textual summary of the source standards and clauses explicitly identified for the sub-questions (e.g., "Sub-Q1 based on GRI X.Y, Sub-Q2 on ISO Z:A.B"). This is for internal documentation.

        Example for "question_text_en" (containing only sub-questions):
        "1. What were the total direct (Scope 1) greenhouse gas (GHG) emissions in the reporting period, in metric tons of CO2 equivalent? (Source: GRI 305-1 - Direct GHG Emissions)\\n2. What methodology was used to calculate these Scope 1 GHG emissions? (Source: GRI 305-1 - Calculation Methodology)\\n3. What were the total indirect (Scope 2) GHG emissions from purchased electricity, heat, or steam in the reporting period? (Source: GRI 305-2 - Indirect GHG Emissions)"
        
        If context is sparse, formulate broader sub-questions suitable for initial data gathering for sustainability reporting based on the theme name, description, and keywords.
        Focus on questions that help disclose performance and management approaches.
        """
        prompt = PromptTemplate.from_template(prompt_template_str)
        
        formatted_prompt = prompt.format(
            theme_name=theme_name,
            theme_description=theme_description if theme_description else "Not specified.",
            theme_keywords=theme_keywords if theme_keywords else "Not specified.",
            source_standards_list_str=", ".join(source_standards_involved) if source_standards_involved else "Various ESG standards.",
            context_from_chunks=context_from_chunks
        )
        
        llm_output = "" 
        json_str = "" 
        try:
            print(f"[QG_SERVICE LOG] Generating question for consolidated theme: {theme_name} (using new prompt)")
            response = await self.qg_llm.ainvoke(formatted_prompt)
            llm_output = response.content.strip()
            
            match = re.search(r"```json\s*(\{[\s\S]*?\})\s*```", llm_output, re.DOTALL) # Allow multiline JSON
            if match: json_str = match.group(1)
            else:
                first_brace = llm_output.find('{'); last_brace = llm_output.rfind('}')
                if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
                    json_str = llm_output[first_brace : last_brace+1]
                else:
                    print(f"[QG_SERVICE ERROR /gen_q_consolidated] LLM output not valid JSON for '{theme_name}': {llm_output}")
                    return None
            
            question_data = json.loads(json_str)
            
            # Validate required fields from LLM output
            if not all(k in question_data for k in ["question_text_en", "category", "theme"]):
                print(f"[QG_SERVICE ERROR /gen_q_consolidated] LLM JSON output missing required keys for theme '{theme_name}'. Got: {question_data.keys()}")
                return None

            # Ensure the theme name from input is used, not whatever LLM might hallucinate
            final_theme_name = theme_name 
            final_category = question_data.get("category", "G").upper()
            if final_category not in ['E', 'S', 'G']: final_category = "G"
            
            detailed_source_info = question_data.get("detailed_source_info_for_subquestions")

            generated_q_obj = GeneratedQuestion(
                 question_text_en=question_data.get("question_text_en", "Error: Question text missing from LLM"),
                 category=final_category, # final_category ถูก define ไว้แล้ว
                 theme=final_theme_name,  # final_theme_name ถูก define ไว้แล้ว
                 additional_info={"detailed_source_info_for_subquestions": detailed_source_info} if detailed_source_info else None
            )
            # Store detailed_source_info_for_subquestions in model_extra for later use in evolve_and_store_questions
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
        active_themes_map: Dict[str, ESGQuestion] = {q.theme: q for q in active_questions_in_db} # Keyed by consolidated theme name (EN)
        print(f"[QG_SERVICE INFO] Found {len(active_questions_in_db)} active questions in DB, covering {len(active_themes_map)} unique themes.")

        process_scope: str = PROCESS_SCOPE_NO_ACTION_DEFAULT
        # This list will hold dicts from `identify_consolidated_themes_from_kg`
        themes_info_for_question_generation: List[Dict[str, Any]] = [] 

        # --- Determine Process Scope and Identify Themes from KG Content ---
        if initial_neo4j_empty_check_result:
            process_scope = PROCESS_SCOPE_KG_INITIAL_FULL_SCAN_FROM_CONTENT
            print(f"[QG_SERVICE INFO] KG is empty/near empty. Scope: {process_scope}. Identifying all themes from KG content.")
            if active_questions_in_db: # Should ideally be empty if KG is empty, but as a safeguard
                print(f"[QG_SERVICE INFO] Deactivating ALL {len(active_questions_in_db)} existing questions as KG is considered empty for new generation.")
                for q_to_deactivate in active_questions_in_db:
                    await ESGQuestion.find_one(ESGQuestion.id == q_to_deactivate.id).update(
                        {"$set": {"is_active": False, "updated_at": current_time_utc}}
                    )
                active_questions_in_db = [] # Clear local cache
                active_themes_map.clear()
            themes_info_for_question_generation = await self.identify_consolidated_themes_from_kg()

        elif uploaded_file_content_bytes: # PDF new upload, KG not initially empty
            process_scope = PROCESS_SCOPE_KG_UPDATED_FULL_SCAN_FROM_CONTENT
            print(f"[QG_SERVICE INFO] PDF uploaded, KG not initially empty. Scope: {process_scope}. Re-identifying all themes from current KG content.")
            themes_info_for_question_generation = await self.identify_consolidated_themes_from_kg(
                existing_active_theme_names=list(active_themes_map.keys())
            )
        elif not uploaded_file_content_bytes: # Scheduled run (no new PDF)
            if not active_questions_in_db and not initial_neo4j_empty_check_result:
                process_scope = PROCESS_SCOPE_KG_FULL_SCAN_DB_EMPTY_FROM_CONTENT
                print(f"[QG_SERVICE INFO] Scheduled run. KG has content, but Question DB is empty. Scope: {process_scope}.")
                themes_info_for_question_generation = await self.identify_consolidated_themes_from_kg()
            elif active_questions_in_db: # Regular scheduled refresh
                process_scope = PROCESS_SCOPE_KG_SCHEDULED_REFRESH_FROM_CONTENT
                print(f"[QG_SERVICE INFO] Scheduled run, active questions exist. Scope: {process_scope}. Re-identifying themes.")
                themes_info_for_question_generation = await self.identify_consolidated_themes_from_kg(
                    existing_active_theme_names=list(active_themes_map.keys())
                )
            else: # Scheduled, KG empty, DB empty - already covered by initial_neo4j_empty_check_result
                 process_scope = PROCESS_SCOPE_NO_ACTION_KG_AND_DB_EMPTY_ON_SCHEDULED_RUN
                 print(f"[QG_SERVICE INFO] Scheduled run. KG and Question DB are both empty. Scope: {process_scope}")
        else:
            process_scope = PROCESS_SCOPE_NO_ACTION_UNEXPECTED_STATE
            print(f"[QG_SERVICE WARNING] Reached unexpected state for process scope determination. No action taken.")

        print(f"[QG_SERVICE INFO] Final Determined process_scope: {process_scope}")
        
        if not themes_info_for_question_generation:
            print(f"[QG_SERVICE WARNING] No themes were identified from KG content for scope '{process_scope}'.")
            # If not an initial scan of an empty KG, and active questions exist, return them as fallback.
            if process_scope != PROCESS_SCOPE_KG_INITIAL_FULL_SCAN_FROM_CONTENT and active_questions_in_db :
                 print("[QG_SERVICE WARNING] Returning existing active questions as fallback.")
                 return [GeneratedQuestion(**q.model_dump(exclude_none=True, exclude={'id','_id', 'revision_id', 'created_at', 'generated_at'})) for q in active_questions_in_db]
            return []

        # --- Generate Candidate Questions for identified themes ---
        print(f"[QG_SERVICE INFO] Generating English questions for {len(themes_info_for_question_generation)} consolidated themes.")
        english_question_tasks = [
            self.generate_question_for_consolidated_theme(theme_info)
            for theme_info in themes_info_for_question_generation
        ]
        generated_english_q_results_with_exc = await asyncio.gather(*english_question_tasks, return_exceptions=True)

        # Link successful GeneratedQuestion objects back to their original full theme_info dict
        # This map uses the object ID of the GeneratedQuestion as a key.
        successful_english_q_objs_map_to_theme_info: Dict[int, Dict[str, Any]] = {}
        
        candidate_bilingual_pydantic_questions: List[GeneratedQuestion] = [] # Holds GeneratedQuestion for API response

        # Filter successful English questions and prepare for translation
        temp_english_questions_for_translation: List[GeneratedQuestion] = []
        for i, result_item in enumerate(generated_english_q_results_with_exc):
            original_theme_info = themes_info_for_question_generation[i]
            if isinstance(result_item, GeneratedQuestion): # Successfully generated English question object
                temp_english_questions_for_translation.append(result_item)
                successful_english_q_objs_map_to_theme_info[id(result_item)] = original_theme_info
            elif isinstance(result_item, Exception):
                theme_name_for_err = original_theme_info.get("theme_name", "Unknown Theme")
                print(f"[QG_SERVICE ERROR] English question generation task failed for theme '{theme_name_for_err}': {result_item}")
                traceback.print_exc()
        
        if temp_english_questions_for_translation:
            print(f"[QG_SERVICE INFO] Translating {len(temp_english_questions_for_translation)} English questions to Thai...")
            translation_tasks = [self.translate_text_to_thai(q.question_text_en) for q in temp_english_questions_for_translation]
            translated_th_texts_with_exc = await asyncio.gather(*translation_tasks, return_exceptions=True)

            for i, en_question_obj in enumerate(temp_english_questions_for_translation):
                th_text_result = translated_th_texts_with_exc[i]
                question_text_th = th_text_result if isinstance(th_text_result, str) else None
                if isinstance(th_text_result, Exception):
                    print(f"[QG_SERVICE ERROR] Translation failed for EN q '{en_question_obj.question_text_en[:50]}...': {th_text_result}")
                
                # Create the final Pydantic GeneratedQuestion object for this iteration
                pydantic_q = GeneratedQuestion(
                    question_text_en=en_question_obj.question_text_en,
                    question_text_th=question_text_th,
                    category=en_question_obj.category,
                    theme=en_question_obj.theme
                )
                # Carry over model_extra (e.g., detailed_source_info_for_subquestions)
                if hasattr(en_question_obj, 'model_extra') and en_question_obj.model_extra:
                    pydantic_q.model_extra = en_question_obj.model_extra
                candidate_bilingual_pydantic_questions.append(pydantic_q)
        
        print(f"[QG_SERVICE INFO] Prepared {len(candidate_bilingual_pydantic_questions)} candidate bilingual questions for DB evolution.")

        # --- DB Operations: Evolve questions in MongoDB ---
        # `all_historical_questions_by_theme` should be fetched once (already done above)
        # --- BEGIN MODIFICATION: Move this block UP ---
        # Fetch all historical questions ONCE for efficient lookup BEFORE the loop
        all_db_questions_docs: List[ESGQuestion] = await ESGQuestion.find_all().to_list()
        all_historical_questions_by_theme: Dict[str, List[ESGQuestion]] = {}
        for q_doc_hist in all_db_questions_docs:
            all_historical_questions_by_theme.setdefault(q_doc_hist.theme, []).append(q_doc_hist)
        
        # Sort historical questions by version descending for each theme
        for theme_name_hist_sort in all_historical_questions_by_theme:
            all_historical_questions_by_theme[theme_name_hist_sort].sort(key=lambda q: q.version, reverse=True)
        
        print(f"[QG_SERVICE INFO] Fetched and prepared {len(all_db_questions_docs)} historical questions from DB, grouped into {len(all_historical_questions_by_theme)} themes.")

        for pydantic_candidate_q in candidate_bilingual_pydantic_questions:
            theme_name_cand = pydantic_candidate_q.theme # This is the consolidated theme name EN

            # Find the original theme_info that generated this pydantic_candidate_q
            # This is a bit indirect. We need to find which object in `temp_english_questions_for_translation`
            # corresponds to `pydantic_candidate_q` to then use its ID for the map.
            original_theme_info_for_db_op: Optional[Dict[str, Any]] = None
            for eng_q_obj_ref in temp_english_questions_for_translation:
                if eng_q_obj_ref.theme == pydantic_candidate_q.theme and \
                   eng_q_obj_ref.question_text_en == pydantic_candidate_q.question_text_en:
                    original_theme_info_for_db_op = successful_english_q_objs_map_to_theme_info.get(id(eng_q_obj_ref))
                    break
            
            if not original_theme_info_for_db_op:
                print(f"[QG_SERVICE WARNING] Critical: Could not map Pydantic candidate Q for theme '{theme_name_cand}' back to its original theme_info. Skipping DB ops.")
                continue

            # ตอนนี้ all_historical_questions_by_theme ถูก define แล้ว
            historical_q_versions_for_theme = all_historical_questions_by_theme.get(theme_name_cand, [])
            latest_db_q_for_theme = historical_q_versions_for_theme[0] if historical_q_versions_for_theme else None
            
            detailed_source_info_from_additional = None
        # Use the new field name here:
            if pydantic_candidate_q.additional_info: 
                detailed_source_info_from_additional = pydantic_candidate_q.additional_info.get("detailed_source_info_for_subquestions")
            # Prepare data for ESGQuestion model from pydantic_candidate_q and original_theme_info_for_db_op
            db_question_data = {
                "question_text_en": pydantic_candidate_q.question_text_en,
                "question_text_th": pydantic_candidate_q.question_text_th,
                "category": pydantic_candidate_q.category,
                "theme": theme_name_cand,
                "keywords": original_theme_info_for_db_op.get("keywords", ""),
                "theme_description_en": original_theme_info_for_db_op.get("description_en"),
                "theme_description_th": original_theme_info_for_db_op.get("theme_name_th"), # This was translated theme name, TODO: translate description_en
                "dimension": original_theme_info_for_db_op.get("dimension"),
                "source_document_references": original_theme_info_for_db_op.get("source_standards", []),
                "constituent_chunk_ids": original_theme_info_for_db_op.get("constituent_chunk_ids", []),
                "detailed_source_info_for_subquestions": detailed_source_info_from_additional,
                "updated_at": current_time_utc,
                "is_active": True
            }

            if latest_db_q_for_theme: # Theme already exists in DB
                is_content_similar = await self.are_questions_substantially_similar(
                    pydantic_candidate_q.question_text_en, latest_db_q_for_theme.question_text_en
                )
                if not is_content_similar: # Content changed significantly
                    print(f"[QG_SERVICE INFO] Question for theme '{theme_name_cand}' content CHANGED. Deactivating old v{latest_db_q_for_theme.version}, inserting new.")
                    if latest_db_q_for_theme.is_active:
                        await ESGQuestion.find_one(ESGQuestion.id == latest_db_q_for_theme.id).update(
                            {"$set": {"is_active": False, "updated_at": current_time_utc}}
                        )
                    db_question_data["version"] = latest_db_q_for_theme.version + 1
                    db_question_data["generated_at"] = current_time_utc
                    new_q_to_insert = ESGQuestion(**db_question_data)
                    try: await new_q_to_insert.insert()
                    except Exception as e_ins_v: print(f"[QG_SERVICE ERROR] DB Insert (new ver) for Q '{theme_name_cand}': {e_ins_v}")
                else: # Content is similar, update details if needed, ensure active
                    print(f"[QG_SERVICE INFO] Question for theme '{theme_name_cand}' content SIMILAR to v{latest_db_q_for_theme.version}. Updating details/activating.")
                    # Check if any other field (keywords, descriptions, sources etc.) changed
                    fields_to_check_for_update = [
                        "keywords", "theme_description_en", "theme_description_th", "dimension", 
                        "source_document_references", "constituent_chunk_ids", 
                        "detailed_source_info_for_subquestions", "question_text_th", "category"
                    ]
                    update_payload = {"is_active": True, "updated_at": current_time_utc}
                    needs_db_update = not latest_db_q_for_theme.is_active
                    for field_key in fields_to_check_for_update:
                        new_value = db_question_data.get(field_key)
                        old_value = getattr(latest_db_q_for_theme, field_key, None)
                        # Handle list comparison carefully if they are not hashable or order matters
                        if isinstance(new_value, list) and isinstance(old_value, list):
                            if sorted(new_value) != sorted(old_value): # Simple list content check
                                update_payload[field_key] = new_value
                                needs_db_update = True
                        elif new_value != old_value:
                            update_payload[field_key] = new_value
                            needs_db_update = True
                    
                    if needs_db_update:
                        await ESGQuestion.find_one(ESGQuestion.id == latest_db_q_for_theme.id).update({"$set": update_payload})
                        print(f"[QG_SERVICE INFO] Updated existing similar question for theme '{theme_name_cand}'.")
            else: # Theme is completely new
                print(f"[QG_SERVICE INFO] Theme '{theme_name_cand}' is NEW to DB. Inserting as v1.")
                db_question_data["version"] = 1
                db_question_data["generated_at"] = current_time_utc
                new_q_to_insert = ESGQuestion(**db_question_data)
                try: await new_q_to_insert.insert()
                except Exception as e_ins_new_t: print(f"[QG_SERVICE ERROR] DB Insert (new theme) for Q '{theme_name_cand}': {e_ins_new_t}")

    # --- Deactivate themes that were active but not generated in this run (if full scan) ---
        current_run_generated_theme_names_set = {theme_info["theme_name"] for theme_info in themes_info_for_question_generation}
        should_deactivate_obsolete_themes = process_scope in [
            PROCESS_SCOPE_KG_INITIAL_FULL_SCAN_FROM_CONTENT, 
            PROCESS_SCOPE_KG_UPDATED_FULL_SCAN_FROM_CONTENT, 
            PROCESS_SCOPE_KG_FULL_SCAN_DB_EMPTY_FROM_CONTENT, 
            PROCESS_SCOPE_KG_SCHEDULED_REFRESH_FROM_CONTENT
        ]

        if should_deactivate_obsolete_themes:
            for active_theme_name, active_q_doc in active_themes_map.items(): # active_themes_map from earlier in the function
                if active_theme_name not in current_run_generated_theme_names_set and active_q_doc.is_active:
                    print(f"[QG_SERVICE INFO] Theme '{active_theme_name}' (v{active_q_doc.version}) no longer identified from KG content during a full scan. Deactivating.")
                    # This await is correctly placed if evolve_and_store_questions is an async def method
                    await ESGQuestion.find_one(ESGQuestion.id == active_q_doc.id).update( # This is likely your line 690
                        {"$set": {"is_active": False, "updated_at": current_time_utc}}
                    )

        # --- Prepare final list for API response (all currently active questions) ---
        final_active_db_questions: List[ESGQuestion] = await ESGQuestion.find(ESGQuestion.is_active == True).to_list()
        final_pydantic_questions_for_api: List[GeneratedQuestion] = [
            GeneratedQuestion(
                question_text_en=q.question_text_en,
                question_text_th=q.question_text_th,
                category=q.category,
                theme=q.theme
            ) for q in final_active_db_questions
        ]
            
        print(f"[QG_SERVICE LOG] Evolution complete. Total {len(final_pydantic_questions_for_api)} bilingual questions are effectively active in DB.")
        return final_pydantic_questions_for_api