import os
import uuid
import logging
import json
import re
import asyncio
from typing import List, IO, Dict, Any, Optional
from langchain_core.documents import Document
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain_cohere import CohereRerank

from app.models.esg_question_model import ESGQuestion
from app.services.pinecone_service import PineconeService
from app.services.persistence.mongodb import MongoDBSaver
from app.services.neo4j_service import Neo4jService

class ReportGenerationService:
    def __init__(self,
                 mongodb_service: MongoDBSaver,
                 pinecone_service: PineconeService,
                 neo4j_service: Neo4jService,
                 llm: ChatGoogleGenerativeAI):
        self.logger = logging.getLogger(__name__)
        self.mongodb_service = mongodb_service
        self.pinecone_service = pinecone_service
        self.neo4j_service = neo4j_service
        self.llm = llm # For English content generation
        self.reranker = CohereRerank(
            cohere_api_key=os.getenv("COHERE_API_KEY"),
            model="rerank-english-v3.0",
            top_n=20
        )
        self.translation_llm = ChatGoogleGenerativeAI(
            model=os.getenv("TRANSLATION_MODEL", "gemini-1.5-flash-preview-0514"),
            temperature=0.7
        )
        self.logger.info("ReportGenerationService initialized with English-First Multi-Agent Architecture.")

    # --- Translation Helpers (Unchanged) ---
    async def _translate_chunks_to_english(self, documents: List[Document]) -> List[Document]:
        self.logger.info(f"Normalizing {len(documents)} chunks to English...")
        def contains_thai(text: str) -> bool:
            return bool(re.search("[\u0E00-\u0E7F]", text))
        prompt = PromptTemplate.from_template(
            "You are a professional translator. Translate the following Thai text accurately to English. Return ONLY the translated English text.\nThai Text:\n{text}\n\nEnglish Translation:"
        )
        chain = prompt | self.translation_llm
        tasks = []
        for doc in documents:
            if contains_thai(doc.page_content):
                tasks.append(chain.ainvoke({"text": doc.page_content}))
            else:
                future = asyncio.Future()
                future.set_result(type('obj', (object,), {'content': doc.page_content})())
                tasks.append(future)
        translated_responses = await asyncio.gather(*tasks, return_exceptions=True)
        translated_docs = []
        for i, response in enumerate(translated_responses):
            if isinstance(response, Exception):
                self.logger.error(f"Translation failed for chunk {i}, using original. Error: {response}")
                translated_docs.append(documents[i])
            else:
                translated_doc = Document(page_content=response.content.strip(), metadata=documents[i].metadata)
                translated_docs.append(translated_doc)
        return translated_docs

    async def _translate_report_to_thai(self, english_report: str, company_name: str) -> str:
        if not english_report or not isinstance(english_report, str): return ""
        self.logger.info(f"Translating the final report for {company_name} to Thai...")
        prompt = PromptTemplate.from_template(
            """
            You are an expert technical translator specializing in translating structured Markdown documents from English to Thai for corporate sustainability reports.
            **Critical Instruction:**
            Your task is to translate the English text content into professional, high-quality Thai.
            You MUST preserve the original Markdown formatting EXACTLY. This includes all headings (e.g., #, ##, ###), horizontal rules (---), bullet points (* or -), bold text (**text**), and all newlines.
            DO NOT alter the structure of the document. DO NOT add any conversational text or explanations. Your entire output must be only the translated Markdown document.

            **English Markdown Document to Translate:**
            ---
            {english_text}
            ---

            **Thai Markdown Translation:**
            """
        )
        chain = prompt | self.translation_llm
        try:
            response = await chain.ainvoke({"english_text": english_report})
            return response.content.strip()
        except Exception as e:
            self.logger.error(f"Final report translation failed: {e}")
            return english_report

    # --- Agent Implementations (Prompts are now consistently in English) ---

    async def _strictly_validate_context(self, sub_questions: List[str], context_chunks: List[str]) -> Dict:
        if not context_chunks or not sub_questions:
            return {"status": "insufficient", "reason": "No context or no sub-questions were provided for validation."}
        context_string = "\n\n---\n\n".join(context_chunks)
        sub_questions_string = "\n".join(f"- {sq}" for sq in sub_questions)
        validator_prompt = PromptTemplate.from_template(
            """
            You are a meticulous ESG Auditor. Your task is to determine if the provided 'Context' contains enough specific information to answer the **majority** of the 'Sub-Questions' listed below.
            **Context from Company Documents:**
            ---
            {context}
            ---
            **Sub-Questions to be Answered:**
            ---
            {sub_questions}
            ---
            **Your Assessment:**
            Review the context and the sub-questions carefully. Can you find explicit facts, figures, or policy details in the context to answer at least 80% of the sub-questions?
            Respond with ONLY a single, valid JSON object with two keys:
            1. "status": Must be either "sufficient" or "insufficient".
            2. "reason": A brief, one-sentence explanation. If insufficient, specify which type of information is missing (e.g., "The context lacks quantitative data," or "No policy details were found.").
            """
        )
        validator_chain = validator_prompt | self.llm
        try:
            response = await validator_chain.ainvoke({"sub_questions": sub_questions_string, "context": context_string})
            json_str = response.content[response.content.find('{'):response.content.rfind('}')+1]
            return json.loads(json_str)
        except Exception as e:
            self.logger.error(f"Strict validator agent failed: {e}")
            return {"status": "insufficient", "reason": "Could not determine sufficiency due to an internal error."}

    async def _generate_answer_for_sub_question(self, sub_question: str, context_chunks: List[str]) -> str:
        if not context_chunks:
            return "No relevant context found for this specific sub-question."
        context_string = "\n\n---\n\n".join(context_chunks)
        answer_prompt = PromptTemplate.from_template(
            """
            Based ONLY on the provided 'Context', provide a direct and concise answer to the following 'Sub-Question'.
            If the context does not contain the answer, state 'Insufficient information in the context to answer this sub-question.'.
            Context:
            ---
            {context}
            ---
            Sub-Question: {sub_question}
            Answer:
            """
        )
        answer_chain = answer_prompt | self.llm
        response = await answer_chain.ainvoke({"sub_question": sub_question, "context": context_string})
        return response.content.strip()

    async def _synthesize_disclosure_from_sub_answers(self, main_question: str, sub_q_and_a: List[Dict]) -> str:
        if not sub_q_and_a:
            return "Although the document contains relevant information, detailed aspects could not be automatically generated."
        synthesis_context = ""
        for item in sub_q_and_a:
            synthesis_context += f"- Sub-Question: {item['sub_question']}\n- Answer Found: {item['sub_answer']}\n\n"
        synthesis_prompt = PromptTemplate.from_template(
            """
            You are a professional sustainability report writer. Your task is to write a formal disclosure statement for an ESG report.
            You will be given a Main Question and a series of detailed Sub-Questions with their corresponding answers, which were extracted from the company's documents.
            Synthesize all the information from the 'Sub-Question & Answer Pairs' into a single, cohesive, and well-written paragraph or a few paragraphs.
            Ensure the final text flows naturally and directly addresses the 'Main Question'.
            Main Question to Address:
            {main_question}
            Sub-Question & Answer Pairs:
            ---
            {synthesis_context}
            ---
            Final Disclosure Statement:
            """
        )
        synthesis_chain = synthesis_prompt | self.llm
        response = await synthesis_chain.ainvoke({"main_question": main_question, "synthesis_context": synthesis_context})
        return response.content.strip()

    def _parse_sub_questions(self, sub_questions_text: str) -> List[str]:
        if not sub_questions_text: return []
        return [q.strip() for q in re.split(r'\n', sub_questions_text) if q.strip()]

    # --- Agent 4: Section Writer ---
    async def _generate_report_section(self, category_name: str, section_data: List[Dict]) -> str:
        self.logger.info(f"Generating detailed report section for: {category_name}")
        if not section_data:
            return f"Specific details on {category_name} policies and performance were not identified in the provided documents."

        structured_context = ""
        for item in section_data:
            question_data = item.get("question_data", {})
            final_disclosure = item.get("final_disclosure", "No disclosure generated.")
            structured_context += f"### Topic: {question_data.get('theme')}\n\n**Disclosure Summary:**\n{final_disclosure}\n\n---\n"
        
        prompt = PromptTemplate.from_template(
            """
            You are an expert ESG report writer. Your task is to write a detailed, comprehensive, and well-structured narrative for the '{category_name}' section of a sustainability report.
            Use the provided 'Disclosure Summaries' below, which are already structured with subheadings (### Topic: ...).
            Your job is to synthesize the information under each subheading into a flowing, professional narrative. **You MUST keep the original `### Topic: ...` subheadings.**
            Elaborate on the points and connect them logically. Where quantitative data is present, summarize it into Markdown tables for clarity. Do not just list the statements. The final text should be well-written and ready for a corporate report.

            **Disclosure Summaries for {category_name}:**
            ---
            {structured_context}
            ---

            **Detailed {category_name} Section Narrative (in English, preserving all ### subheadings):**
            """
        )
        chain = prompt | self.llm
        response = await chain.ainvoke({"category_name": category_name, "structured_context": structured_context})
        return response.content.strip()
    
    # --- Agent 5: Executive Editor ---
    async def _finalize_report(self, company_name: str, content_g: str, content_e: str, content_s: str, insufficient_items: List[Dict]) -> str:
        self.logger.info(f"Assembling the final report for {company_name}...")
        appendix_section = ""
        if insufficient_items:
            missing_topics_list = "\n".join(f"- **{item['question_data']['theme']} ({item['question_data']['category']}):** {item.get('final_disclosure', '')}" for item in insufficient_items)
            appendix_section = f"""
## Appendix: Topics Requiring More Information
Based on the analysis, additional information is required for the following topics to create a comprehensive report. It is recommended to gather specific data and policies related to these areas:
{missing_topics_list}
"""
        finalizer_prompt = PromptTemplate.from_template(
            """
            You are an automated ESG report generation engine. Your sole task is to generate a complete Markdown document based on the provided pre-written sections and data. Do not write any conversational text. Your entire output MUST be the Markdown code for the report.

            **Instructions:**
            1.  Use the provided company name '{company_name}'.
            2.  Write a professional 2-paragraph "Message from the CEO" summarizing the company's commitment to sustainability.
            3.  Write a brief "About This Report" section explaining the report's scope and methodology (e.g., based on GRI standards, covering the 2024 period).
            4.  Assemble the pre-written sections for Governance, Environmental, and Social exactly as provided.
            5.  Write a brief, forward-looking Conclusion.
            6.  If the Appendix section is provided, include it at the end.

            ---
            **REPORT TEMPLATE AND CONTENT**

            # Sustainability Report
            ## {company_name}
            *(Reporting Period: 2024)*

            ---

            ### Message from the CEO
            (Your written 2-paragraph message goes here. Be professional, strategic, and inspiring.)

            ### About This Report
            (Your written brief explanation goes here, mentioning the use of international standards for guidance.)

            ---

            ## 2. Governance (G)
            {content_g}

            ---

            ## 3. Environmental (E)
            {content_e}

            ---

            ## 4. Social (S)
            {content_s}

            ---

            ## 5. Conclusion
            (Your written forward-looking conclusion goes here.)

            {appendix_section}
            """
        )
        chain = finalizer_prompt | self.llm
        
        english_report_response = await chain.ainvoke({
            "company_name": company_name,
            "content_g": content_g,
            "content_e": content_e,
            "content_s": content_s,
            "appendix_section": appendix_section
        })
        english_report_content = english_report_response.content.strip()
        
        return await self._translate_report_to_thai(english_report_content, company_name)

    # --- Main Orchestrator ---
    async def generate_sustainability_report(self, files: List[IO[bytes]], file_names: List[str]) -> Dict[str, Any]:
        session_index_name = f"user-report-{uuid.uuid4().hex[:12]}"
        self.logger.info(f"Starting report generation with English-First Multi-Agent workflow for: {', '.join(file_names)}")

        try:
            # Step 1: Ingest, translate to English, and index documents
            self.logger.info("Phase 1: Ingesting and processing documents...")
            initial_docs = await self.neo4j_service.read_PDFs_and_create_documents_azure(files, file_names)
            text_chunks = await self.neo4j_service.split_documents_into_chunks(initial_docs)
            english_chunks = await self._translate_chunks_to_english(text_chunks)
            self.logger.info(f"Uploading {len(english_chunks)} English chunks to Pinecone...")
            self.pinecone_service.upsert_documents(session_index_name, english_chunks)

            # Step 2: Fetch questions and generate raw data
            self.logger.info("Phase 2: Generating raw disclosure data in English...")
            question_ai_set = await ESGQuestion.find(ESGQuestion.is_active == True).to_list()
            if not question_ai_set:
                raise ValueError("Question AI set is empty.")

            raw_report_data = []
            for q_data in question_ai_set:
                main_question_text = q_data.main_question_text_en
                if not main_question_text: continue
                sub_questions_list = []
                if q_data.sub_questions_sets:
                    sub_questions_text = q_data.sub_questions_sets[0].sub_question_text_en
                    sub_questions_list = self._parse_sub_questions(sub_questions_text)
                
                if not sub_questions_list:
                    answer_status, final_answer, detailed_q_and_a, top_context = "insufficient", "The standard question is missing sub-questions.", [], []
                else:
                    initial_context = self.pinecone_service.query(session_index_name, main_question_text, top_k=50)
                    docs_for_rerank = [Document(page_content=text) for text in initial_context]
                    reranked_docs = self.reranker.compress_documents(documents=docs_for_rerank, query=main_question_text)
                    top_context = [doc.page_content for doc in reranked_docs]
                    
                    validation_result = await self._strictly_validate_context(sub_questions_list, top_context)
                    answer_status = validation_result.get("status", "insufficient")
                    
                    final_answer, detailed_q_and_a = "", []
                    if answer_status == "sufficient":
                        for sub_q in sub_questions_list:
                            sub_q_context = self.pinecone_service.query(session_index_name, sub_q, top_k=5)
                            sub_q_answer = await self._generate_answer_for_sub_question(sub_q, sub_q_context)
                            detailed_q_and_a.append({"sub_question": sub_q, "sub_answer": sub_q_answer})
                        final_answer = await self._synthesize_disclosure_from_sub_answers(main_question_text, detailed_q_and_a)
                    else:
                        final_answer = f"INSUFFICIENT DATA: {validation_result.get('reason')}"

                raw_report_data.append({
                    "question_data": q_data.model_dump(by_alias=True),
                    "status": answer_status,
                    "final_disclosure": final_answer,
                    "detailed_q_and_a": detailed_q_and_a,
                    "retrieved_context": top_context
                })
            
            # Step 3: Generate detailed report sections in English
            self.logger.info("Phase 3: Generating detailed English report sections...")
            successful_items = [item for item in raw_report_data if item.get("status") == "sufficient"]
            g_data = [item for item in successful_items if item['question_data']['category'] == 'G']
            e_data = [item for item in successful_items if item['question_data']['category'] == 'E']
            s_data = [item for item in successful_items if item['question_data']['category'] == 'S']

            governance_section_text_en = await self._generate_report_section("Governance", g_data)
            environmental_section_text_en = await self._generate_report_section("Environmental", e_data)
            social_section_text_en = await self._generate_report_section("Social", s_data)

            # Step 4: Assemble and translate the final report
            self.logger.info("Phase 4: Assembling and translating final report...")
            company_name = file_names[0].split('_')[0].replace('-', ' ') if file_names else "The Company"
            insufficient_items = [item for item in raw_report_data if item.get("status") == "insufficient"]
            
            final_markdown_report_th = await self._finalize_report(
                company_name=company_name,
                content_g=governance_section_text_en,
                content_e=environmental_section_text_en,
                content_s=social_section_text_en,
                insufficient_items=insufficient_items
            )

            return {
                "raw_data": raw_report_data,
                "markdown_report": final_markdown_report_th
            }

        finally:
            self.logger.info(f"Cleaning up temporary index: {session_index_name}")
            self.pinecone_service.delete_index(session_index_name)