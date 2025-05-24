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
from datetime import datetime, timezone

from functools import partial
from unstructured.partition.auto import partition as unstructured_partition

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from app.models.esg_question_model import ESGQuestion
from app.services.neo4j_service import Neo4jService

load_dotenv()

PROCESS_SCOPE_PDF_SPECIFIC_IMPACT = "pdf_specific_impact"
PROCESS_SCOPE_KG_FULL_SCAN_GENERAL_PDF = "kg_full_scan_general_pdf"
PROCESS_SCOPE_KG_FULL_SCAN_SCHEDULED_INITIAL_OR_EMPTY_KG = "kg_full_scan_scheduled_initial_or_empty_kg"
PROCESS_SCOPE_NO_OP_ANALYSIS_FAILED_OR_NO_TARGETS = "no_op_analysis_failed_or_no_targets"

class GeneratedQuestion(BaseModel):
    question_text_en: str = PydanticField(..., description="The text of the generated ESG question.")
    question_text_th: Optional[str] = PydanticField(None, description="The Thai text of the generated ESG question.")
    category: str = PydanticField(..., description="The ESG category (E, S, or G).")
    theme: str = PydanticField(..., description="The ESG theme or area of inquiry this question relates to.")

class QuestionGenerationService:
    def __init__(self,
                 neo4j_service: Neo4jService,
                 similarity_embedding_model: Embeddings
                ):
        self.neo4j_service = neo4j_service
        self.similarity_llm_embedding = similarity_embedding_model
        self.qg_llm = ChatGoogleGenerativeAI(
            model=os.getenv("QUESTION_GENERATION_MODEL", "gemini-1.5-flash-preview-0514"),
            temperature=0.7,
            max_retries=3,
        )
        self.translation_llm = ChatGoogleGenerativeAI(
            model=os.getenv("TRANSLATION_MODEL", "gemini-1.5-flash-preview-0514"),
            temperature=0.0,
            max_retries=3,
        )
        self.graph_schema_content: Optional[str] = None
        self._load_graph_schema()
        if self.similarity_llm_embedding:
            print("[QG_SERVICE LOG] Similarity embedding model received and configured.")
        else:
            print("[QG_SERVICE WARNING] Similarity embedding model was NOT provided. Similarity checks might fallback or be less accurate.")

    def _load_graph_schema(self):
        try:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(os.path.dirname(current_dir))
            schema_file_path = os.path.join(project_root, "neo4j_graph_schema.txt")
            if os.path.exists(schema_file_path):
                with open(schema_file_path, 'r', encoding='utf-8') as f:
                    self.graph_schema_content = f.read()
                print(f"[QG_SERVICE LOG] Successfully loaded Neo4j graph schema from: {schema_file_path}")
            else:
                self.graph_schema_content = "Graph schema is not available."
                print(f"[QG_SERVICE ERROR] Graph schema file not found at: {schema_file_path}")
        except Exception as e:
            self.graph_schema_content = "Error loading graph schema."
            print(f"[QG_SERVICE ERROR] Error loading graph schema: {e}")

    async def translate_text_to_thai(self, text_to_translate: str) -> Optional[str]:
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

    async def analyze_pdf_impact(self, file_content_bytes: bytes, existing_theme_details_input: List[Dict[str, str]]) -> Dict[str, Any]:
        default_error_response = {"impacted_theme_names": [], "new_topic_description": None, "is_general_update": True, "brief_summary": "Error in analysis process."}
        if not file_content_bytes:
            return {"impacted_theme_names": [], "new_topic_description": None, "is_general_update": False, "brief_summary": "No content to analyze."}

        # For the prompt, we only need theme names from the input details
        simple_existing_theme_names = [item['theme_name'] for item in existing_theme_details_input]
        existing_themes_str_for_prompt = "\n".join([f"- {th_name}" for th_name in simple_existing_theme_names]) if simple_existing_theme_names else "No existing themes provided."
        
        pdf_text_content = ""
        temp_file_path = None
        try:
            print(f"[QG_SERVICE LOG /analyze_pdf_impact] Starting PDF content extraction with unstructured...")
            loop = asyncio.get_running_loop()
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmpfile:
                tmpfile.write(file_content_bytes)
                temp_file_path = tmpfile.name
            
            unstructured_partition_callable = partial(unstructured_partition, filename=temp_file_path, strategy="auto")
            elements = await loop.run_in_executor(None, unstructured_partition_callable)
            pdf_text_content = "\n\n".join([el.text for el in elements if hasattr(el, 'text') and el.text])

            if not pdf_text_content.strip():
                print(f"[QG_SERVICE WARNING /analyze_pdf_impact] PDF content seems empty after unstructured processing.")
                return {"impacted_theme_names": [], "new_topic_description": None, "is_general_update": True, "brief_summary": "Processed PDF content is empty."}
            print(f"[QG_SERVICE LOG /analyze_pdf_impact] Extracted PDF content length: {len(pdf_text_content)} characters.")
            pdf_text_for_llm = pdf_text_content
        except Exception as e:
            print(f"[QG_SERVICE ERROR /analyze_pdf_impact] Error during PDF content extraction: {e}")
            return {**default_error_response, "brief_summary": "Error processing PDF content."}
        finally:
            if temp_file_path and os.path.exists(temp_file_path):
                try: os.remove(temp_file_path)
                except Exception as e_remove: print(f"[QG_SERVICE WARNING /analyze_pdf_impact] Could not remove temp file: {e_remove}")

        prompt_template_str = """
        You are an ESG analyst reviewing an uploaded document.
        The full content of the document is:
        ---DOCUMENT CONTENT START---
        {pdf_content_full}
        ---DOCUMENT CONTENT END---

        Here is a list of existing ESG Question Themes:
        ---EXISTING THEMES START---
        {existing_themes_list_str} 
        ---EXISTING THEMES END---

        Task:
        1. Analyze the document content.
        2. Identify which of the "EXISTING THEMES" are most directly impacted or updated. List their exact names.
        3. If the document introduces a significant new ESG topic not covered by existing themes, describe it.
        4. Determine if this document is a general ESG update (e.g., a full sustainability report).

        Output your analysis in STRICT JSON format with ONLY these keys:
        - "impacted_theme_names": List of exact theme names from EXISTING THEMES affected. Empty if none.
        - "new_topic_description": String describing new topic areas, or null.
        - "is_general_update": Boolean (true if a broad update, false if specific).
        - "brief_summary": One-sentence summary of the document's main ESG relevance.
        Do not add explanations outside the JSON object.

        Example Output 1 (Specific Update):
        {{"impacted_theme_names": ["Greenhouse Gas Emissions Reporting", "Energy Management and Efficiency"], "new_topic_description": null, "is_general_update": false, "brief_summary": "The document details updated CO2 emission factors and new energy saving targets."}}
        Example Output 2 (New Topic):
        {{"impacted_theme_names": [], "new_topic_description": "The document introduces regulations about 'Circular Economy in Packaging Materials', which is not fully covered.", "is_general_update": false, "brief_summary": "The document outlines new specific regulations for packaging recycling."}}
        Example Output 3 (General Update):
        {{"impacted_theme_names": ["Greenhouse Gas Emissions Reporting", "Waste Management and Circular Economy", "Occupational Health and Safety Performance"], "new_topic_description": null, "is_general_update": true, "brief_summary": "The document is a comprehensive annual sustainability report covering multiple ESG aspects."}}
        """
        prompt = PromptTemplate.from_template(prompt_template_str)
        formatted_prompt = prompt.format(pdf_content_full=pdf_text_for_llm, existing_themes_list_str=existing_themes_str_for_prompt)

        try:
            print(f"[QG_SERVICE LOG /analyze_pdf_impact] Sending PDF content (length: {len(pdf_text_for_llm)}) to LLM for impact analysis.")
            response = await self.qg_llm.ainvoke(formatted_prompt)
            llm_output = response.content.strip()
            print(f"[QG_SERVICE DEBUG /analyze_pdf_impact] LLM Raw Output: {llm_output}")
            json_str = ""
            match = re.search(r"```json\s*(\{.*?\})\s*```", llm_output, re.DOTALL)
            if match: json_str = match.group(1)
            else:
                first_brace = llm_output.find('{'); last_brace = llm_output.rfind('}')
                if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
                    json_str = llm_output[first_brace : last_brace+1]
                else:
                    if llm_output.strip().startswith("{") and llm_output.strip().endswith("}"): json_str = llm_output.strip()
                    else:
                        print(f"[QG_SERVICE ERROR /analyze_pdf_impact] LLM output not valid JSON: {llm_output}")
                        return {**default_error_response, "brief_summary": "LLM output format error."}
            
            analysis_result = json.loads(json_str)
            return {
                "impacted_theme_names": analysis_result.get("impacted_theme_names", []),
                "new_topic_description": analysis_result.get("new_topic_description"),
                "is_general_update": analysis_result.get("is_general_update", False),
                "brief_summary": analysis_result.get("brief_summary", "Summary not provided.")
            }
        except json.JSONDecodeError as e:
            print(f"[QG_SERVICE ERROR /analyze_pdf_impact] Failed to parse JSON: {e}. Cleaned: {json_str}. Original: {llm_output}")
            return {**default_error_response, "brief_summary": "Error in LLM output parsing."}
        except Exception as e:
            print(f"[QG_SERVICE ERROR /analyze_pdf_impact] LLM call error: {e}")
            return {**default_error_response, "brief_summary": "Error during LLM analysis."}

    async def identify_esg_themes_from_graph(self,
                                            num_themes: int = 20,
                                            target_themes_input: Optional[List[Union[str, Dict[str, str]]]] = None,
                                            exclude_existing_theme_names: Optional[List[str]] = None # New parameter
                                            ) -> List[Dict[str, str]]:
        if not self.graph_schema_content or "not available" in self.graph_schema_content or "Error loading" in self.graph_schema_content:
            return []

        final_themes_to_return: List[Dict[str, str]] = []
        themes_needing_keywords_from_llm: List[str] = []

        if target_themes_input:
            # ... (existing logic for target_themes_input remains the same) ...
            print(f"[QG_SERVICE LOG] Processing target_themes_input ({len(target_themes_input)} items) for keyword generation.")
            for item in target_themes_input:
                if isinstance(item, str):
                    themes_needing_keywords_from_llm.append(item)
                elif isinstance(item, dict):
                    theme_name = item.get("theme_name")
                    keywords = item.get("keywords")
                    if theme_name and keywords:
                        final_themes_to_return.append({"theme_name": theme_name, "keywords": keywords})
                    elif theme_name:
                        themes_needing_keywords_from_llm.append(theme_name)
            if not themes_needing_keywords_from_llm and final_themes_to_return: # All targets had keywords
                return final_themes_to_return
            if not themes_needing_keywords_from_llm and not final_themes_to_return: # No valid targets
                 return []


            prompt_template_str = """
            You are an expert ESG analyst. For each of the following ESG themes, provide 3-5 relevant keywords or concepts
            (comma-separated in English) that might be found or represented in the ESG Knowledge Graph
            whose schema is provided below. These keywords will be used to retrieve context for generating questions.

            ESG Knowledge Graph Schema:
            ---SCHEMA START---
            {graph_schema}
            ---SCHEMA END---

            Generate keywords for these specific themes:
            {target_themes_list_str}

            Output Format for EACH theme:
            Theme Name: [The Exact Theme Name Provided]
            Keywords: keyword1, keyword2, keyword3
            """
            target_themes_names_for_prompt = "\n".join([f"- {name}" for name in themes_needing_keywords_from_llm])
            formatted_prompt = PromptTemplate.from_template(prompt_template_str).format(
                graph_schema=self.graph_schema_content,
                target_themes_list_str=target_themes_names_for_prompt
            )
            mode_description = f"keywords for {len(themes_needing_keywords_from_llm)} target themes"
            process_llm_output_for_targeted_keywords = True
        else: # Full theme generation (discover new themes, potentially excluding existing ones)
            exclude_instruction = ""
            if exclude_existing_theme_names:
                exclude_instruction = (
                    "Identify key reporting themes that are DISTINCT and DIFFERENT from the following existing themes:\n"
                    + "\n".join([f"- {name}" for name in exclude_existing_theme_names])
                    + "\nFocus on identifying themes not well covered by the list above.\n"
                )
            
            prompt_template_str = """
            You are an expert ESG (Environmental, Social, and Governance) analyst identifying key reporting themes
            for industrial packaging factories based on their knowledge graph structure.

            ESG Knowledge Graph Schema:
            ---SCHEMA START---
            {graph_schema}
            ---SCHEMA END---

            {exclude_instruction}
            Strictly using ONLY the provided knowledge graph schema, identify {num_themes} key reporting themes.
            For each, provide: a. A concise theme name (English). b. 3-5 relevant keywords (comma-separated English).

            Output Format: List of themes, each with "Theme Name:" and "Keywords:" on separate lines. Example:
            Theme Name: Carbon Emission Reduction Strategies
            Keywords: carbon footprint, emission targets, renewable energy

            List {num_themes} such themes.
            """
            formatted_prompt = PromptTemplate.from_template(prompt_template_str).format(
                graph_schema=self.graph_schema_content,
                num_themes=num_themes,
                exclude_instruction=exclude_instruction
            )
            mode_description = f"{num_themes} ESG themes from schema (exclude known: {bool(exclude_existing_theme_names)})"
            process_llm_output_for_targeted_keywords = False
        # ... (rest of the parsing logic for identify_esg_themes_from_graph - should be fine) ...
        print(f"[QG_SERVICE LOG] Identifying {mode_description} using LLM...")
        llm_output = "" 
        try:
            response = await self.qg_llm.ainvoke(formatted_prompt)
            llm_output = response.content.strip()
            generated_items_from_llm: List[Dict[str, str]] = []
            current_item_dict = {}
            for line in llm_output.splitlines():
                line_stripped = line.strip()
                if line_stripped.startswith("Theme Name:"):
                    if current_item_dict.get("theme_name") and current_item_dict.get("keywords"):
                        generated_items_from_llm.append(current_item_dict)
                    current_item_dict = {"theme_name": line_stripped.replace("Theme Name:", "").strip()}
                elif line_stripped.startswith("Keywords:") and "theme_name" in current_item_dict:
                    current_item_dict["keywords"] = line_stripped.replace("Keywords:", "").strip()
            if current_item_dict.get("theme_name") and current_item_dict.get("keywords"):
                generated_items_from_llm.append(current_item_dict)

            if process_llm_output_for_targeted_keywords:
                generated_keywords_map = {item["theme_name"].strip(): item["keywords"].strip() for item in generated_items_from_llm if item.get("theme_name") and item.get("keywords")}
                for theme_name_requested in themes_needing_keywords_from_llm:
                    matched_theme_name = next((k for k in generated_keywords_map.keys() if k.lower() == theme_name_requested.lower()), None)
                    if matched_theme_name: 
                        final_themes_to_return.append({
                            "theme_name": theme_name_requested, 
                            "keywords": generated_keywords_map[matched_theme_name]
                        })
                    else:
                        print(f"[QG_SERVICE WARNING] LLM did not generate keywords for targeted theme: {theme_name_requested}. LLM generated themes: {list(generated_keywords_map.keys())}")
                valid_themes = final_themes_to_return
            else: 
                valid_themes = [t for t in generated_items_from_llm if t.get("theme_name") and t.get("keywords")]
                if exclude_existing_theme_names: # Filter out themes that are too similar to excluded ones
                    valid_themes = [
                        vt for vt in valid_themes 
                        if not any(exc_name.lower() == vt['theme_name'].lower() for exc_name in exclude_existing_theme_names)
                    ]


            print(f"[QG_SERVICE LOG] Successfully processed {len(valid_themes)} themes with keywords for {mode_description}.")
            return valid_themes
        except Exception as e:
            print(f"[QG_SERVICE ERROR] Error during LLM call or parsing for {mode_description}: {e}. LLM output snippet: {llm_output[:500]}")
            if process_llm_output_for_targeted_keywords:
                return final_themes_to_return 
            return []


    async def generate_question_for_theme(self, theme_name: str, keywords: str) -> Optional[GeneratedQuestion]:
        # ... (context retrieval ... )
        cypher_context_str = "No specific graph structure snippets retrieved."
        vector_context_str = "No specific textual information retrieved."
        try:
            retrieved_data = await self.neo4j_service.get_output(query=f"{theme_name} {keywords}", k=3)
            if hasattr(retrieved_data, 'cypher_answer') and retrieved_data.cypher_answer:
                cypher_context_str = retrieved_data.cypher_answer
            if hasattr(retrieved_data, 'relate_documents') and retrieved_data.relate_documents:
                vector_context_str = "\n".join([doc.page_content for doc in retrieved_data.relate_documents if doc.page_content])
        except Exception as e:
            print(f"[QG_SERVICE ERROR] Error retrieving context for theme '{theme_name}': {e}.")

        # MODIFIED PROMPT FOR SUB-QUESTIONS
        prompt_template_str = """
        You are an expert ESG consultant for industrial packaging factories. For the theme "{theme_name}" with keywords "{keywords}" and context:
        Context (Graph): {cypher_context}
        Context (Vector): {vector_context}

        Task: 
        1. Formulate ONE primary, overarching question for the theme "{theme_name}".
        2. Formulate 2-4 specific sub-questions that delve into key aspects or components of this primary question. These sub-questions should guide a user to provide comprehensive information related to the theme.
        3. Indicate the overall ESG category (E, S, or G) for this theme.

        Output ONLY a single JSON object with the following structure:
        {{
          "question_text_en": "Primary Question Text...\\n\\nSub-questions:\\n1. Sub-question 1 text...\\n2. Sub-question 2 text...\\n3. Sub-question 3 text... (and so on if more)",
          "category": "E",
          "theme": "{theme_name}"
        }}
        Ensure the sub-questions are clearly enumerated within the "question_text_en" field, separated by newlines.
        "category" MUST be 'E', 'S', or 'G'. "theme" MUST be "{theme_name}".
        If context is minimal, formulate broader questions based on theme and keywords.
        """
        prompt = PromptTemplate.from_template(prompt_template_str)
        formatted_prompt = prompt.format(
            theme_name=theme_name, keywords=keywords,
            cypher_context=cypher_context_str, vector_context=vector_context_str
        )
        # ... (LLM call and JSON parsing - existing logic should be fine, but ensure it handles potentially longer question_text_en)
        llm_output = "" 
        json_str = "" 
        try:
            response = await self.qg_llm.ainvoke(formatted_prompt)
            llm_output = response.content.strip()
            
            match = re.search(r"```json\s*(\{.*?\})\s*```", llm_output, re.DOTALL)
            if match: json_str = match.group(1)
            else:
                first_brace = llm_output.find('{'); last_brace = llm_output.rfind('}')
                if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
                    json_str = llm_output[first_brace : last_brace+1]
                else:
                    print(f"[QG_SERVICE ERROR /gen_q_theme] LLM output not JSON for '{theme_name}': {llm_output}")
                    return None
            
            question_data = json.loads(json_str)
            if question_data.get("theme", theme_name) != theme_name:
                question_data["theme"] = theme_name

            generated_q = GeneratedQuestion(**question_data)
            if generated_q.category.upper() not in ['E', 'S', 'G']:
                generated_q.category = "G"
            return generated_q
        except json.JSONDecodeError as e:
            print(f"[QG_SERVICE ERROR /gen_q_theme] JSON parse error for '{theme_name}': {e}. Str: {json_str}. Original: {llm_output}")
            return None
        except Exception as e:
            print(f"[QG_SERVICE ERROR /gen_q_theme] Question formulation error for '{theme_name}': {e}")
            return None


    async def are_questions_substantially_similar(self, text1: str, text2: str, threshold: float = 0.85) -> bool:
        # ... (existing implementation is good) ...
        if not text1 or not text2: return False
        text1_lower = text1.strip().lower()
        text2_lower = text2.strip().lower()
        if text1_lower == text2_lower: return True
        
        if not self.similarity_llm_embedding:
            print("[QG_SERVICE WARNING] Similarity model N/A. Basic string compare for similarity.")
            return text1_lower == text2_lower
        
        try:
            loop = asyncio.get_running_loop()
            embeddings = await loop.run_in_executor(None, self.similarity_llm_embedding.embed_documents, [text1_lower, text2_lower])
            if len(embeddings) < 2 or not embeddings[0] or not embeddings[1]:
                print("[QG_SERVICE ERROR] Failed to generate embeddings for similarity check.")
                return False
            
            embedding1 = np.array(embeddings[0]).reshape(1, -1)
            embedding2 = np.array(embeddings[1]).reshape(1, -1)
            sim_score = cosine_similarity(embedding1, embedding2)[0][0]
            is_similar = sim_score >= threshold
            print(f"[QG_SERVICE SIMILARITY] Score for '{text1[:30]}...' vs '{text2[:30]}...': {sim_score:.4f} (Th: {threshold}, Similar: {is_similar})")
            return is_similar
        except Exception as e:
            print(f"[QG_SERVICE ERROR] Similarity check error: {e}. Defaulting to False (not similar).")
            return False


    async def evolve_and_store_questions(self, uploaded_file_content_bytes: Optional[bytes] = None) -> List[GeneratedQuestion]:
        print(f"[QG_SERVICE LOG] Orchestrating Question AI Evolution. File uploaded: {'Yes' if uploaded_file_content_bytes else 'No'}")
        current_time_utc = datetime.now(timezone.utc)
        
        initial_neo4j_empty_check_result = await self.neo4j_service.is_graph_substantially_empty()
        print(f"[QG_SERVICE INFO] Neo4j graph substantially empty check (initial): {initial_neo4j_empty_check_result}")

        process_scope: str = ""
        target_themes_for_theme_id_phase: Optional[List[Union[str, Dict[str, str]]]] = None # For PDF_SPECIFIC_IMPACT
        impact_analysis_result_holder: Optional[Dict[str, Any]] = None # To store result if PDF is analyzed

        active_questions_in_db: List[ESGQuestion] = await ESGQuestion.find(ESGQuestion.is_active == True).to_list()
        active_themes_map: Dict[str, ESGQuestion] = {q.theme: q for q in active_questions_in_db}
        print(f"[QG_SERVICE INFO] Found {len(active_questions_in_db)} active questions in DB, covering {len(active_themes_map)} unique themes.")

        if not uploaded_file_content_bytes:
            process_scope = PROCESS_SCOPE_KG_FULL_SCAN_SCHEDULED_INITIAL_OR_EMPTY_KG
        elif not active_questions_in_db:
            process_scope = PROCESS_SCOPE_KG_FULL_SCAN_SCHEDULED_INITIAL_OR_EMPTY_KG
        elif initial_neo4j_empty_check_result:
            process_scope = PROCESS_SCOPE_KG_FULL_SCAN_SCHEDULED_INITIAL_OR_EMPTY_KG
            if active_questions_in_db:
                print(f"[QG_SERVICE INFO] Deactivating ALL {len(active_questions_in_db)} existing questions from MongoDB as KG was initially empty.")
                for q_to_deactivate in active_questions_in_db:
                     await ESGQuestion.find_one(ESGQuestion.id == q_to_deactivate.id).update(
                        {"$set": {"is_active": False, "updated_at": current_time_utc}})
                active_questions_in_db = []
                active_themes_map.clear()
        else: # PDF uploaded, active questions exist, KG was NOT initially empty
            print("[QG_SERVICE INFO] PDF uploaded, active questions exist, KG not initially empty. Analyzing PDF impact...")
            existing_theme_details_for_analysis: List[Dict[str, str]] = [
                {"theme_name": q.theme, "keywords": q.keywords if hasattr(q, 'keywords') and q.keywords else "", "current_question_en": q.question_text_en}
                for q in active_questions_in_db
            ]
            impact_analysis_result = await self.analyze_pdf_impact(uploaded_file_content_bytes, existing_theme_details_for_analysis)
            impact_analysis_result_holder = impact_analysis_result # Store for later use
            print(f"[QG_SERVICE INFO] PDF Impact Analysis Result: {impact_analysis_result}")

            if impact_analysis_result.get("is_general_update") or \
               (not impact_analysis_result.get("impacted_theme_names") and not impact_analysis_result.get("new_topic_description")):
                process_scope = PROCESS_SCOPE_KG_FULL_SCAN_GENERAL_PDF
            else:
                target_themes_for_theme_id_phase = []
                impacted_theme_names_from_pdf = list(set(impact_analysis_result.get("impacted_theme_names", [])))
                for theme_name in impacted_theme_names_from_pdf:
                    active_q_for_theme = active_themes_map.get(theme_name)
                    if active_q_for_theme and hasattr(active_q_for_theme, 'keywords') and active_q_for_theme.keywords:
                        target_themes_for_theme_id_phase.append({"theme_name": theme_name, "keywords": active_q_for_theme.keywords})
                    else:
                        target_themes_for_theme_id_phase.append(theme_name)
                
                if impact_analysis_result.get("new_topic_description"):
                    new_topic_theme_name = f"New Topic: {impact_analysis_result['new_topic_description'][:100]}"
                    target_themes_for_theme_id_phase.append(new_topic_theme_name)
                
                if target_themes_for_theme_id_phase:
                    process_scope = PROCESS_SCOPE_PDF_SPECIFIC_IMPACT
                else: 
                    process_scope = PROCESS_SCOPE_NO_OP_ANALYSIS_FAILED_OR_NO_TARGETS
                    return [GeneratedQuestion(**q.model_dump(exclude_none=True, exclude={'id','_id', 'revision_id'})) for q in active_questions_in_db]
        
        print(f"[QG_SERVICE INFO] Determined process_scope: {process_scope}")

        themes_to_process_info_with_keywords: List[Dict[str, str]] = []
        if process_scope == PROCESS_SCOPE_KG_FULL_SCAN_SCHEDULED_INITIAL_OR_EMPTY_KG:
            default_num_themes = 20 # Initial set, or full refresh from empty
            print(f"[QG_SERVICE INFO] Identifying up to {default_num_themes} themes from KG (Initial/Empty KG Scan).")
            themes_to_process_info_with_keywords = await self.identify_esg_themes_from_graph(
                num_themes=default_num_themes, target_themes_input=None)
        
        elif process_scope == PROCESS_SCOPE_KG_FULL_SCAN_GENERAL_PDF:
            print(f"[QG_SERVICE INFO] Scope '{process_scope}'. Refreshing existing themes and discovering truly new ones.")
            combined_targets_for_general_pdf_scan: List[Union[str, Dict[str,str]]] = []
            if active_themes_map:
                print(f"[QG_SERVICE INFO] Preparing {len(active_themes_map)} existing active themes for keyword refresh.")
                for theme_name, q_doc in active_themes_map.items():
                    combined_targets_for_general_pdf_scan.append({
                        "theme_name": theme_name, 
                        "keywords": q_doc.keywords if hasattr(q_doc, 'keywords') and q_doc.keywords else ""
                    })
            
            # Add new topic from PDF impact analysis, if any, to be processed for keywords
            if impact_analysis_result_holder and impact_analysis_result_holder.get("new_topic_description"):
                new_topic_theme_name = f"New Topic: {impact_analysis_result_holder['new_topic_description'][:100]}"
                # Avoid duplicate if it somehow got named like an existing theme
                if not any(isinstance(t, str) and t.lower() == new_topic_theme_name.lower() for t in combined_targets_for_general_pdf_scan) and \
                   not any(isinstance(t, dict) and t.get("theme_name", "").lower() == new_topic_theme_name.lower() for t in combined_targets_for_general_pdf_scan):
                    print(f"[QG_SERVICE INFO] Adding potential new topic '{new_topic_theme_name}' from PDF to general scan targets.")
                    combined_targets_for_general_pdf_scan.append(new_topic_theme_name)

            if combined_targets_for_general_pdf_scan:
                themes_to_process_info_with_keywords = await self.identify_esg_themes_from_graph(
                    target_themes_input=combined_targets_for_general_pdf_scan
                )
            else: # No active themes and no new topic from PDF, but KG is not empty.
                  # Fallback to identifying a default set of themes like an initial scan.
                default_num_themes = 5 # Discover a smaller number of themes if all else fails.
                print(f"[QG_SERVICE WARNING] General PDF scan with no active themes and no new PDF topic. Discovering {default_num_themes} themes from KG.")
                themes_to_process_info_with_keywords = await self.identify_esg_themes_from_graph(
                    num_themes=default_num_themes, 
                    target_themes_input=None,
                    exclude_existing_theme_names=list(active_themes_map.keys()) # Try to get genuinely new ones
                )
        
        elif process_scope == PROCESS_SCOPE_PDF_SPECIFIC_IMPACT:
            print(f"[QG_SERVICE INFO] Scope '{process_scope}'. Identifying/enriching targeted themes: {target_themes_for_theme_id_phase}")
            themes_to_process_info_with_keywords = await self.identify_esg_themes_from_graph(
                target_themes_input=target_themes_for_theme_id_phase)
        
        if not themes_to_process_info_with_keywords:
            print(f"[QG_SERVICE WARNING] No themes identified to process for scope '{process_scope}'.")
            if process_scope != PROCESS_SCOPE_KG_FULL_SCAN_SCHEDULED_INITIAL_OR_EMPTY_KG or \
               (process_scope == PROCESS_SCOPE_KG_FULL_SCAN_SCHEDULED_INITIAL_OR_EMPTY_KG and not initial_neo4j_empty_check_result and active_questions_in_db):
                 print("[QG_SERVICE WARNING] Scan on non-empty graph or specific PDF impact yielded no themes. Returning existing active questions as fallback.")
                 return [GeneratedQuestion(**q.model_dump(exclude_none=True, exclude={'id','_id', 'revision_id'})) for q in active_questions_in_db]
            return []

        # --- Generate Candidate Questions & DB Ops (largely same as before) ---
        # ... (The rest of your code for generating questions, translating, and DB operations)
        # ... The crucial change was in how themes_to_process_info_with_keywords is populated for PROCESS_SCOPE_KG_FULL_SCAN_GENERAL_PDF
        # ... The final loop "Handle previously active themes that were NOT processed in this run" remains important
        #     and its logic to PRESERVE themes for PROCESS_SCOPE_KG_FULL_SCAN_GENERAL_PDF is correct.

        print(f"[QG_SERVICE INFO] Generating English questions for {len(themes_to_process_info_with_keywords)} themes identified for scope '{process_scope}'...")
        english_question_tasks = [
            self.generate_question_for_theme(theme_info["theme_name"], theme_info["keywords"])
            for theme_info in themes_to_process_info_with_keywords if theme_info.get("theme_name") and theme_info.get("keywords")
        ]
        newly_generated_english_results = await asyncio.gather(*english_question_tasks, return_exceptions=True)

        successful_new_english_questions: List[GeneratedQuestion] = []
        successful_indices_for_keywords: List[int] = [] 
        for result_idx, result_item in enumerate(newly_generated_english_results):
            if isinstance(result_item, GeneratedQuestion):
                successful_new_english_questions.append(result_item)
                successful_indices_for_keywords.append(result_idx) 
            elif isinstance(result_item, Exception):
                if result_idx < len(themes_to_process_info_with_keywords): # Bounds check
                    theme_name_for_error = themes_to_process_info_with_keywords[result_idx].get("theme_name", "Unknown Theme")
                    print(f"[QG_SERVICE ERROR] English question generation task failed for theme '{theme_name_for_error}': {result_item}")
                else:
                    print(f"[QG_SERVICE ERROR] English question generation task failed for an out-of-bounds index: {result_item}")

        candidate_bilingual_questions_pydantic: List[GeneratedQuestion] = []
        candidate_question_keywords_map: Dict[str, str] = {} 

        if successful_new_english_questions:
            print(f"[QG_SERVICE INFO] Translating {len(successful_new_english_questions)} English questions to Thai...")
            translation_tasks = [self.translate_text_to_thai(q.question_text_en) for q in successful_new_english_questions]
            translated_th_results = await asyncio.gather(*translation_tasks, return_exceptions=True)

            for i, en_question_obj in enumerate(successful_new_english_questions):
                th_text_result = translated_th_results[i]
                question_text_th = th_text_result if isinstance(th_text_result, str) else None
                if isinstance(th_text_result, Exception):
                    print(f"[QG_SERVICE ERROR] Translation failed for EN q '{en_question_obj.question_text_en[:30]}...': {th_text_result}")
                
                pydantic_q = GeneratedQuestion(
                    question_text_en=en_question_obj.question_text_en,
                    question_text_th=question_text_th,
                    category=en_question_obj.category,
                    theme=en_question_obj.theme 
                )
                candidate_bilingual_questions_pydantic.append(pydantic_q)
                
                original_theme_index = successful_indices_for_keywords[i]
                if original_theme_index < len(themes_to_process_info_with_keywords): # Bounds check
                    original_theme_info = themes_to_process_info_with_keywords[original_theme_index]
                    candidate_question_keywords_map[pydantic_q.theme] = original_theme_info.get('keywords', '')
                else:
                    print(f"[QG_SERVICE WARNING] Could not map keywords for theme '{pydantic_q.theme}' due to index out of bounds.")
        print(f"[QG_SERVICE INFO] Prepared {len(candidate_bilingual_questions_pydantic)} candidate bilingual questions for DB evolution.")

        all_final_questions_to_return_for_ui: List[GeneratedQuestion] = []
        
        all_db_questions_docs = await ESGQuestion.find_all().to_list()
        all_historical_questions_by_theme: Dict[str, List[ESGQuestion]] = {}
        for q_doc in all_db_questions_docs:
            all_historical_questions_by_theme.setdefault(q_doc.theme, []).append(q_doc)
        for theme_name_hist in all_historical_questions_by_theme:
            all_historical_questions_by_theme[theme_name_hist].sort(key=lambda q: q.version, reverse=True)

        for candidate_q_pydantic in candidate_bilingual_questions_pydantic:
            theme_name = candidate_q_pydantic.theme
            keywords_for_candidate_q = candidate_question_keywords_map.get(theme_name, '')
            historical_versions_for_theme = all_historical_questions_by_theme.get(theme_name, [])
            latest_historical_q_doc = historical_versions_for_theme[0] if historical_versions_for_theme else None

            if latest_historical_q_doc:
                is_similar = await self.are_questions_substantially_similar(
                    candidate_q_pydantic.question_text_en, latest_historical_q_doc.question_text_en)
                if not is_similar:
                    print(f"[QG_SERVICE INFO] Q for theme '{theme_name}' CHANGED. Deactivating old (v{latest_historical_q_doc.version}), inserting new.")
                    if latest_historical_q_doc.is_active:
                         await ESGQuestion.find_one(ESGQuestion.id == latest_historical_q_doc.id).update(
                             {"$set": {"is_active": False, "updated_at": current_time_utc}})
                    new_version_num = latest_historical_q_doc.version + 1
                    esg_new_q_doc = ESGQuestion(
                        **candidate_q_pydantic.model_dump(), keywords=keywords_for_candidate_q,
                        is_active=True, version=new_version_num,
                        generated_at=current_time_utc, updated_at=current_time_utc)
                    try: 
                        await esg_new_q_doc.insert()
                        all_final_questions_to_return_for_ui.append(candidate_q_pydantic)
                    except Exception as e: print(f"[QG_SERVICE ERROR] DB Insert (new ver) for Q '{theme_name}': {e}")
                else:
                    print(f"[QG_SERVICE INFO] Q for theme '{theme_name}' SIMILAR to latest historical (v{latest_historical_q_doc.version}). Ensuring it's active & updating.")
                    update_payload = {"updated_at": current_time_utc, "is_active": True}
                    if hasattr(latest_historical_q_doc, 'keywords') and latest_historical_q_doc.keywords != keywords_for_candidate_q:
                        update_payload["keywords"] = keywords_for_candidate_q
                    await ESGQuestion.find_one(ESGQuestion.id == latest_historical_q_doc.id).update({"$set": update_payload})
                    updated_doc_for_pydantic = latest_historical_q_doc.model_copy(update=update_payload)
                    all_final_questions_to_return_for_ui.append(
                        GeneratedQuestion(**updated_doc_for_pydantic.model_dump(exclude={'id', '_id', 'revision_id'})))
            else:
                print(f"[QG_SERVICE INFO] Theme '{theme_name}' is NEW. Inserting as v1.")
                esg_new_q_doc = ESGQuestion(
                    **candidate_q_pydantic.model_dump(), keywords=keywords_for_candidate_q,
                    is_active=True, version=1,
                    generated_at=current_time_utc, updated_at=current_time_utc)
                try: 
                    await esg_new_q_doc.insert()
                    all_final_questions_to_return_for_ui.append(candidate_q_pydantic)
                except Exception as e: print(f"[QG_SERVICE ERROR] DB Insert (new theme) for Q '{theme_name}': {e}")

        themes_handled_in_current_run = {q.theme for q in all_final_questions_to_return_for_ui}
        
        for theme_name_previously_active, previously_active_q_doc in active_themes_map.items():
            if theme_name_previously_active not in themes_handled_in_current_run:
                should_deactivate_due_to_empty_kg = (
                    process_scope == PROCESS_SCOPE_KG_FULL_SCAN_SCHEDULED_INITIAL_OR_EMPTY_KG and
                    initial_neo4j_empty_check_result 
                )
                if should_deactivate_due_to_empty_kg:
                    if previously_active_q_doc.is_active: 
                        print(f"[QG_SERVICE INFO] Theme '{theme_name_previously_active}' (v{previously_active_q_doc.version}) is OBSOLETE due to initial empty KG state. Deactivating.")
                        await ESGQuestion.find_one(ESGQuestion.id == previously_active_q_doc.id).update(
                            {"$set": {"is_active": False, "updated_at": current_time_utc}})
                else:
                    print(f"[QG_SERVICE INFO] Theme '{theme_name_previously_active}' (v{previously_active_q_doc.version}) was not processed for update in this run (scope: {process_scope}). PRESERVING as active.")
                    all_final_questions_to_return_for_ui.append(
                        GeneratedQuestion(**previously_active_q_doc.model_dump(exclude={'id', '_id', 'revision_id'})))
        
        seen_questions_tuples = set()
        deduplicated_final_list = []
        for q_to_return in all_final_questions_to_return_for_ui:
            q_tuple = (q_to_return.theme, q_to_return.question_text_en)
            if q_tuple not in seen_questions_tuples:
                deduplicated_final_list.append(q_to_return)
                seen_questions_tuples.add(q_tuple)
        
        print(f"[QG_SERVICE LOG] Evolution complete. Total {len(deduplicated_final_list)} bilingual questions are effectively active.")
        return deduplicated_final_list