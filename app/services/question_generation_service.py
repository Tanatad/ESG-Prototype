import asyncio
import os
from typing import List, Dict, Any, Optional, Union # Added Union
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field # สำหรับ Validate JSON output จาก LLM
import json # สำหรับ parse JSON string จาก LLM
from app.models.esg_question_model import ESGQuestion # Assuming this can be imported for storage
from app.services.neo4j_service import Neo4jService # สมมติว่า Neo4jService สามารถ import ได้
from app.services.rate_limit import llm_rate_limiter # ใช้ rate limiter เดียวกัน
# from datetime import datetime # Uncomment if you use it for generated_at

load_dotenv()

# Pydantic model สำหรับโครงสร้างคำถามที่เราต้องการ
class GeneratedQuestion(BaseModel):
    question_text_en: str = Field(..., description="The text of the generated ESG question.")
    question_text_th: Optional[str] = Field(None, description="The Thai text of the generated ESG question.")
    category: str = Field(..., description="The ESG category (E, S, or G).")
    theme: str = Field(..., description="The ESG theme or area of inquiry this question relates to.")

class QuestionGenerationService:
    def __init__(self, neo4j_service: Neo4jService):
        self.neo4j_service = neo4j_service
        self.qg_llm = ChatGoogleGenerativeAI(
            model=os.getenv("QUESTION_GENERATION_MODEL", "gemini-1.5-flash-preview-0514"), # Updated model name based on common Gemini versions
            temperature=0.8,
            max_retries=5,
            # rate_limiter=llm_rate_limiter # Assuming llm_rate_limiter is correctly defined elsewhere
        )
        self.translation_llm = ChatGoogleGenerativeAI(
            model=os.getenv("TRANSLATION_MODEL", "gemini-1.5-flash-preview-0514"), # Updated model name
            temperature=0.0,
            max_retries=5,
            # rate_limiter=llm_rate_limiter # Assuming llm_rate_limiter is correctly defined elsewhere
        )
        self.graph_schema_content: Optional[str] = None
        self._load_graph_schema()

    def _load_graph_schema(self):
        """
        Loads the graph schema from the neo4j_graph_schema.txt file.
        Make sure this file is accessible by the service.
        """
        try:
            schema_file_path = os.path.join(os.path.dirname(__file__), "..", "..", "neo4j_graph_schema.txt")
            if os.path.exists(schema_file_path):
                with open(schema_file_path, 'r', encoding='utf-8') as f:
                    self.graph_schema_content = f.read()
                print("[QG_SERVICE LOG] Successfully loaded Neo4j graph schema.")
            else:
                print(f"[QG_SERVICE ERROR] Graph schema file not found at: {schema_file_path}")
                self.graph_schema_content = "Graph schema is not available."
        except Exception as e:
            print(f"[QG_SERVICE ERROR] Error loading graph schema: {e}")
            self.graph_schema_content = "Error loading graph schema."

    async def translate_text_to_thai(self, text_to_translate: str) -> Optional[str]:
        if not text_to_translate:
            return None
        
        prompt_template = PromptTemplate.from_template(
            "Translate the following English text to Thai. Provide only the Thai translation.\n\nEnglish Text: \"{english_text}\"\n\nThai Translation:"
        )
        formatted_prompt = prompt_template.format(english_text=text_to_translate)
        
        try:
            print(f"[QG_SERVICE LOG] Translating to Thai: '{text_to_translate[:50]}...'")
            response = await self.translation_llm.ainvoke(formatted_prompt)
            translated_text = response.content.strip()
            print(f"[QG_SERVICE LOG] Translated to Thai: '{translated_text[:50]}...'")
            return translated_text
        except Exception as e:
            print(f"[QG_SERVICE ERROR] Error during translation to Thai: {e}")
            return None

    async def identify_esg_themes_from_graph(self, num_themes: int = 15) -> List[Dict[str, str]]:
        if not self.graph_schema_content or "not available" in self.graph_schema_content or "Error loading" in self.graph_schema_content:
            print("[QG_SERVICE ERROR] Cannot identify themes without a valid graph schema.")
            return []

        prompt_template = PromptTemplate.from_template(
            """
            You are an expert ESG (Environmental, Social, and Governance) analyst tasked with identifying key reporting themes
            for industrial factories based on their knowledge graph structure.
            Your goal is to help generate relevant questions for their ESG sustainability reports.

            Here is the schema of their ESG Knowledge Graph:
            ---SCHEMA START---
            {graph_schema}
            ---SCHEMA END---

            Considering this schema and the general mandatory ESG reporting requirements for industrial factories,
            please identify {num_themes} distinct and crucial ESG reporting themes or areas of inquiry.
            For each theme, provide a concise name for the theme and a few relevant keywords or concepts
            (3-5 comma-separated keywords) that might be found or represented in the graph related to this theme.

            Example Output Format for a single theme:
            Theme Name: Carbon Emission Reduction Strategies
            Keywords: carbon footprint, emission targets, renewable energy, energy efficiency, GHG reporting

            Provide your output as a list of themes, with each theme starting with "Theme Name:" and "Keywords:" on separate lines.
            List {num_themes} such themes. Focus on themes directly applicable to industrial factory operations and mandatory reporting.
            """
        )
        
        formatted_prompt = prompt_template.format(graph_schema=self.graph_schema_content, num_themes=num_themes)
        
        print(f"[QG_SERVICE LOG] Identifying {num_themes} ESG themes using LLM...")
        try:
            response = await self.qg_llm.ainvoke(formatted_prompt)
            llm_output = response.content
            print(f"[QG_SERVICE LOG] LLM output for themes:\n{llm_output}")
            
            themes = []
            current_theme = {}
            for line in llm_output.splitlines():
                if line.startswith("Theme Name:"):
                    if current_theme.get("theme_name") and current_theme.get("keywords"): # Save previous valid theme
                        themes.append(current_theme)
                    current_theme = {"theme_name": line.replace("Theme Name:", "").strip()}
                elif line.startswith("Keywords:") and "theme_name" in current_theme: # Ensure theme_name was set
                    current_theme["keywords"] = line.replace("Keywords:", "").strip()
            if current_theme.get("theme_name") and current_theme.get("keywords"): # Add the last valid theme
                themes.append(current_theme)
            
            valid_themes = [t for t in themes if "theme_name" in t and "keywords" in t and t["theme_name"] and t["keywords"]]
            print(f"[QG_SERVICE LOG] Successfully parsed {len(valid_themes)} themes.")
            return valid_themes
        except Exception as e:
            print(f"[QG_SERVICE ERROR] Error during LLM call for theme identification: {e}")
            return []

    async def generate_question_for_theme(self, theme_name: str, keywords: str) -> Optional[GeneratedQuestion]:
        print(f"[QG_SERVICE LOG] Generating question for theme: '{theme_name}' with keywords: '{keywords}'")
        
        cypher_context_str = "No specific graph structure snippets retrieved."
        vector_context_str = "No specific textual information retrieved."

        try:
            retrieved_data = await self.neo4j_service.get_output(query=f"{theme_name} {keywords}", k=3)
            
            if retrieved_data.cypher_answer:
                 cypher_context_str = retrieved_data.cypher_answer
            
            if retrieved_data.relate_documents:
                vector_context_str = "\n".join([doc.page_content for doc in retrieved_data.relate_documents])
            
            if cypher_context_str == "No specific graph structure snippets retrieved." and vector_context_str == "No specific textual information retrieved.":
                print(f"[QG_SERVICE WARNING] No context retrieved for theme: {theme_name}. Using theme and keywords only.")

        except Exception as e:
            print(f"[QG_SERVICE ERROR] Error retrieving context for theme '{theme_name}': {e}. Proceeding without specific context.")
        
        prompt_template = PromptTemplate.from_template(
            """
            You are an expert ESG (Environmental, Social, and Governance) consultant creating targeted questions for sustainability reports
            of industrial factories.

            Theme/Area of Inquiry: {theme_name}

            Retrieved Context from ESG Knowledge Graph:
            1. Graph Structure Snippets (Cypher Context - describes entities and relationships):
            {cypher_context}

            2. Related Textual Information (Vector Context - provides textual details):
            {vector_context}

            Task:
            Based on the theme and the provided context (both graph structure and textual information),
            formulate one precise and actionable question that a factory should answer for its ESG report.
            - The question should be directly relevant to the theme and the factory's operations.
            - If the context suggests a potential gap, deficiency, or lack of information regarding the theme,
              the question should be phrased to inquire about this gap or the factory's status on the matter.
            - If context is minimal or absent, formulate a general but insightful question based on the theme and keywords.
            - The question should encourage a specific and informative answer, not just a yes/no.
            - Indicate the primary ESG category for this question (E for Environmental, S for Social, G for Governance).

            Output ONLY a single JSON object with the following keys: "question_text_en", "category", "theme".
            Example JSON output:
            {{
                "question_text_en": "Describe the specific measures your factory has implemented in the past year to reduce fresh water consumption and what were the results?",
                "category": "E",
                "theme": "Water Conservation"
            }}
            
            Ensure the category is one of 'E', 'S', or 'G'.
            Do not add any explanations or conversational text outside the JSON object.
            The "theme" in the output JSON should be the original theme: {theme_name}
            """
        )

        formatted_prompt = prompt_template.format(
            theme_name=theme_name,
            cypher_context=cypher_context_str,
            vector_context=vector_context_str
        )

        try:
            print(f"[QG_SERVICE LOG] Generating question text with LLM for theme: {theme_name}")
            response = await self.qg_llm.ainvoke(formatted_prompt)
            llm_output = response.content.strip()
            print(f"[QG_SERVICE LOG] LLM output for question:\n{llm_output}")

            if llm_output.startswith("```json"):
                llm_output = llm_output.replace("```json", "").replace("```", "").strip()
            
            question_data = json.loads(llm_output)
            
            generated_q = GeneratedQuestion(**question_data)
            if generated_q.category.upper() not in ['E', 'S', 'G']:
                print(f"[QG_SERVICE WARNING] LLM returned invalid category: {generated_q.category}. Defaulting to 'G' or consider re-prompting.")
                # Potentially default or raise an error if category is critical and invalid
                # For now, Pydantic might catch it if an Enum was used, or we handle it post-validation.
                # If not using Enum, you might want to enforce category: e.g., generated_q.category = "G" (default)
            
            # Ensure the theme from the LLM output matches the input theme, or use the input theme.
            if generated_q.theme != theme_name:
                print(f"[QG_SERVICE WARNING] LLM changed the theme from '{theme_name}' to '{generated_q.theme}'. Using original theme '{theme_name}'.")
                generated_q.theme = theme_name

            print(f"[QG_SERVICE LOG] Successfully generated question for theme '{theme_name}': {generated_q.question_text_en}")
            return generated_q
        except json.JSONDecodeError as e:
            print(f"[QG_SERVICE ERROR] Failed to parse JSON output from LLM for theme '{theme_name}': {e}. Output was: {llm_output}")
            return None
        except Exception as e:
            print(f"[QG_SERVICE ERROR] Error during LLM call for question formulation for theme '{theme_name}': {e}")
            return None

    async def generate_and_store_questions(self) -> List[GeneratedQuestion]:
        print("[QG_SERVICE LOG] Starting Question AI generation process...")
        all_bilingual_questions: List[GeneratedQuestion] = []

        identified_themes = await self.identify_esg_themes_from_graph(num_themes=20)

        if not identified_themes:
            print("[QG_SERVICE WARNING] No ESG themes identified. Cannot generate questions.")
            return []

        english_question_tasks = []
        for theme_info in identified_themes:
            if "theme_name" in theme_info and "keywords" in theme_info:
                english_question_tasks.append(
                    self.generate_question_for_theme(theme_info["theme_name"], theme_info["keywords"])
                )
        
        # Results can be GeneratedQuestion, None, or Exception
        english_question_results: List[Union[GeneratedQuestion, None, Exception]] = await asyncio.gather(
            *english_question_tasks, return_exceptions=True
        )

        successful_english_questions: List[GeneratedQuestion] = []
        translation_input_texts: List[str] = []

        for result in english_question_results:
            if isinstance(result, GeneratedQuestion):
                successful_english_questions.append(result)
                translation_input_texts.append(result.question_text_en)
            elif isinstance(result, Exception):
                print(f"[QG_SERVICE ERROR] An error occurred during EN question generation for a theme: {result}")
            # If result is None (due to JSON parsing error or other handled issue in generate_question_for_theme), log it or skip silently
            elif result is None:
                print(f"[QG_SERVICE INFO] Skipping translation for a theme as no valid EN question object was generated.")


        if not successful_english_questions:
            print("[QG_SERVICE WARNING] No English questions were successfully generated for translation.")
        else:
            translation_tasks_for_successful_ones = [
                self.translate_text_to_thai(text) for text in translation_input_texts
            ]
            # Results can be str, None, or Exception
            translated_th_results: List[Union[str, None, Exception]] = await asyncio.gather(
                *translation_tasks_for_successful_ones, return_exceptions=True
            )

            for i, en_question_obj in enumerate(successful_english_questions):
                th_text_result = translated_th_results[i]
                question_text_th: Optional[str] = None

                if isinstance(th_text_result, str):
                    question_text_th = th_text_result
                elif th_text_result is None: # translate_text_to_thai returned None explicitly
                    print(f"[QG_SERVICE INFO] Translation returned None for EN question: '{en_question_obj.question_text_en[:50]}...'")
                elif isinstance(th_text_result, Exception):
                    print(f"[QG_SERVICE ERROR] Error translating EN question '{en_question_obj.question_text_en[:50]}...': {th_text_result}")
                
                # Ensure all leading spaces are standard spaces (ASCII 32)
                # Corrected instantiation with standard spaces:
                bilingual_q = GeneratedQuestion(
                    question_text_en=en_question_obj.question_text_en,
                    question_text_th=question_text_th, # Will be None if translation failed or returned None
                    category=en_question_obj.category,
                    theme=en_question_obj.theme
                )
                all_bilingual_questions.append(bilingual_q)
        
        # Corrected print statement with standard spaces
        print(f"[QG_SERVICE LOG] Generated a total of {len(all_bilingual_questions)} bilingual questions.")
        
        if all_bilingual_questions:
            print(f"[QG_SERVICE LOG] Storing {len(all_bilingual_questions)} questions to MongoDB (simulation)...")
            for q_idx, q_obj in enumerate(all_bilingual_questions):
                # Simulate storing by printing
                # print(f"  Storing Q{q_idx+1}: EN='{q_obj.question_text_en[:50]}...', TH='{str(q_obj.question_text_th)[:50] if q_obj.question_text_th else 'N/A'}' Cat='{q_obj.category}', Theme='{q_obj.theme}'")
                # Actual MongoDB storage logic using Beanie:
                # try:
                #     esg_q_doc = ESGQuestion(
                #         question_text_en=q_obj.question_text_en,
                #         question_text_th=q_obj.question_text_th,
                #         category=q_obj.category,
                #         theme=q_obj.theme,
                #         # generated_at=datetime.utcnow() # Add a timestamp if ESGQuestion model has it
                #     )
                #     await esg_q_doc.insert()
                # except Exception as e:
                #     print(f"[QG_SERVICE ERROR] Failed to store question {q_idx+1} in MongoDB: {e}")
                pass # Placeholder for actual storage
            print("[QG_SERVICE LOG] Questions stored successfully (simulation).")
        else:
            print("[QG_SERVICE WARNING] No questions were generated to store.")

        return all_bilingual_questions

# Example of how this service might be triggered
async def main_trigger_question_generation(neo4j_service_instance: Neo4jService):
    qg_service = QuestionGenerationService(neo4j_service=neo4j_service_instance)
    await qg_service.generate_and_store_questions()