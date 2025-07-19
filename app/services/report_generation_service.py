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
from langchain_core.messages.base import BaseMessage
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
import tempfile
from langchain_community.document_loaders import AzureAIDocumentIntelligenceLoader
from azure.core.credentials import AzureKeyCredential
from azure.ai.documentintelligence import DocumentIntelligenceClient
# --- แก้ไข Imports ---
from app.models.esg_question_model import ESGQuestion
from app.services.vector_store_service import VectorStoreService
from app.services.persistence.mongodb import MongoDBSaver

def _extract_content_from_response(response: Any) -> str:
    """
    A robust helper to extract content from an LLM response,
    correctly handling both a single message object and a list containing a message.
    """
    message_to_process = None
    if isinstance(response, list) and response:
        message_to_process = response[0]
    elif isinstance(response, BaseMessage):
        message_to_process = response
    
    if message_to_process:
        return getattr(message_to_process, 'content', "").strip()
    return ""

class ReportGenerationService:
    def __init__(self,
                 mongodb_service: MongoDBSaver,
                 vector_store_service: VectorStoreService, # <-- ใช้ Service ใหม่
                 llm: ChatGoogleGenerativeAI):
        self.logger = logging.getLogger(__name__)
        self.mongodb_service = mongodb_service
        self.vector_store_service = vector_store_service # <-- ใช้ Service ใหม่
        self.llm = llm
        self.reranker = CohereRerank(
            cohere_api_key=os.getenv("COHERE_API_KEY"),
            model="rerank-v4.0",
            top_n=25
        )
        self.translation_llm = ChatGoogleGenerativeAI(
            model=os.getenv("TRANSLATION_MODEL", "gemini-2.5-flash-preview-05-20"),
            temperature=0.7
        )
        self.logger.info("ReportGenerationService initialized with VectorStoreService.")
        self.azure_di_endpoint = os.getenv("AZURE_DOCUMENTINTELLIGENCE_ENDPOINT") # ใช้ชื่อ ENV ที่ตรงกับของคุณ
        self.azure_di_key = os.getenv("AZURE_DOCUMENTINTELLIGENCE_KEY") # ใช้ชื่อ ENV ที่ตรงกับของคุณ

        if not all([self.azure_di_endpoint, self.azure_di_key]):
            self.document_intelligence_client = None
            self.logger.warning("Azure Document Intelligence credentials not set. PDF processing will fail.")
        else:
            try:
                self.document_intelligence_client = DocumentIntelligenceClient(
                    endpoint=self.azure_di_endpoint,
                    credential=AzureKeyCredential(self.azure_di_key)
                )
                self.logger.info("Azure Document Intelligence client initialized successfully.")
            except Exception as e:
                self.document_intelligence_client = None
                self.logger.error(f"Failed to initialize Azure Document Intelligence client: {e}")
    # --- Document Processing Helper ---
    async def _read_pages_with_azure(self, files: List[IO[bytes]], file_names: List[str]) -> List[Document]:
        """
        อ่าน PDF แต่ละหน้าด้วย Azure DI และสร้าง Document object สำหรับแต่ละหน้า
        """
        if not self.document_intelligence_client:
            self.logger.error("Azure DI client not initialized. Cannot read PDFs.")
            return []

        all_page_documents: List[Document] = []
        loop = asyncio.get_running_loop()

        for i, file_stream in enumerate(files):
            file_name = file_names[i]
            self.logger.info(f"Processing '{file_name}' with Azure Document Intelligence...")
            
            try:
                file_stream.seek(0)
                file_bytes = file_stream.read()

                poller = self.document_intelligence_client.begin_analyze_document(
                    model_id="prebuilt-layout",
                    analyze_request=file_bytes,
                    content_type="application/octet-stream"
                )
                
                # เรียกแบบ blocking ใน thread แยก เพื่อไม่ให้กระทบ event loop หลัก
                result = await loop.run_in_executor(None, poller.result)

                if result and result.pages:
                    for page in result.pages:
                        page_text = "\n".join([line.content for line in page.lines if line.content])
                        if page_text.strip():
                            metadata = {
                                "source_file_name": file_name,
                                "page_number": page.page_number
                            }
                            all_page_documents.append(Document(page_content=page_text, metadata=metadata))
                    self.logger.info(f"Successfully extracted {len(result.pages)} pages from '{file_name}'.")
                else:
                    self.logger.warning(f"Azure DI processed '{file_name}' but found no pages.")

            except Exception as e:
                self.logger.error(f"Error processing '{file_name}' with Azure DI: {e}", exc_info=True)

        return all_page_documents

    # --- Translation Helpers ---
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
                content = _extract_content_from_response(response)
                translated_doc = Document(page_content=content, metadata=documents[i].metadata)
                translated_docs.append(translated_doc)
        return translated_docs

    async def _translate_report_to_thai(self, english_report: str, company_name: str) -> str:
        if not english_report or not isinstance(english_report, str): return ""
        self.logger.info(f"Translating the final report for {company_name} to Thai...")
        prompt = PromptTemplate.from_template(
            """
            You are an expert technical translator specializing in structured Markdown. Translate the English text to professional Thai, PRESERVING ALL MARKDOWN FORMATTING EXACTLY (headings, lists, bold, tables, etc.). Your entire output must be only the translated Markdown.

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
            return _extract_content_from_response(response)
        except Exception as e:
            self.logger.error(f"Final report translation failed: {e}")
            return english_report

    # --- Agent 1: Strict Validator ---
    async def _strictly_validate_context(self, sub_questions: List[str], context_chunks: List[str]) -> Dict:
        if not context_chunks or not sub_questions:
            return {"status": "insufficient", "reason": "No context or no sub-questions were provided for validation."}
        context_string = "\n\n---\n\n".join(context_chunks)
        sub_questions_string = "\n".join(f"- {sq}" for sq in sub_questions)
        validator_prompt = PromptTemplate.from_template(
            """
            You are a meticulous ESG Auditor. Your task is to determine if the provided 'Context' contains enough specific, factual information (like numbers, names, specific policy details) to answer the **majority** of the 'Sub-Questions'. General, high-level statements are NOT sufficient.

            **Context from Company Documents:**
            {context}

            **Sub-Questions to be Answered:**
            {sub_questions}

            **Your Assessment:**
            Review the context and sub-questions carefully. Respond with ONLY a single, valid JSON object with two keys:
            1. "status": Must be "sufficient" or "insufficient".
            2. "reason": A brief, one-sentence explanation for your decision.
            """
        )
        validator_chain = validator_prompt | self.llm
        try:
            response = await validator_chain.ainvoke({"sub_questions": sub_questions_string, "context": context_string})
            content = _extract_content_from_response(response)
            json_str = content[content.find('{'):content.rfind('}')+1]
            return json.loads(json_str)
        except Exception as e:
            self.logger.error(f"Strict validator agent failed: {e}")
            return {"status": "insufficient", "reason": "Could not determine sufficiency due to an internal error."}

    # --- Agent 2: Fact Extractor ---
    async def _extract_facts_for_sub_question(self, sub_question: str, context: List[str]) -> List[str]:
        prompt_template = PromptTemplate.from_template(
            """
            You are a meticulous data extractor. Your task is to extract raw, objective facts from the "Source Documents" that DIRECTLY answer the "Question".

            **INSTRUCTIONS:**
            1. **Extract, DO NOT Analyze:** Do not interpret or form opinions. Extract only factual statements, data points, or direct quotes.
            2. **Direct Relevance Only:** Only extract information that directly answers the Question.
            3. **Output Format:** Present the extracted facts as a clear, bulleted list.
            4. **No Information Found:** If you find NO relevant facts, respond with the single phrase: "NO RELEVANT INFORMATION FOUND".

            **Question:**
            {sub_question}

            **Source Documents:**
            ---
            {context_str}
            ---

            **Extracted Facts:**
            """
        )
        chain = prompt_template | self.llm
        try:
            context_str = "\n---\n".join(context)
            response = await chain.ainvoke({"sub_question": sub_question, "context_str": context_str}, config={"configurable": {"temperature": 0.0}})
            content = _extract_content_from_response(response)
            if "NO RELEVANT INFORMATION FOUND" in content: return []
            return [fact.strip() for fact in content.split('\n') if fact.strip() and fact.startswith(('*', '-'))]
        except Exception as e:
            self.logger.error(f"Fact Extractor agent failed for sub-question '{sub_question}': {e}")
            return []

    async def _analyze_facts_for_disclosure(self, main_question: str, facts_by_sub_q: List[Dict]) -> str:
        """
        ปรับปรุงใหม่: วิเคราะห์ชุดข้อเท็จจริงดิบที่สกัดมาได้ เพื่อสร้างเป็น Disclosure ที่สมดุล
        """
        
        # จัดรูปแบบ Fact ที่ได้มาให้อ่านง่าย
        formatted_facts = ""
        for item in facts_by_sub_q:
            formatted_facts += f"\n**Sub-Question:** {item['sub_question']}\n"
            if item['facts']:
                for fact in item['facts']:
                    formatted_facts += f"- {fact}\n"
            else:
                formatted_facts += "- No specific facts were extracted for this sub-question.\n"

        # --- START: PROMPT ที่ปรับปรุงใหม่ ---
        prompt_text = f"""
        You are an expert ESG analyst. Your task is to write a balanced, neutral, and comprehensive disclosure statement for a sustainability report.

        **Primary Question to Answer:**
        {main_question}

        **Available Factual Evidence:**
        You must base your entire analysis STRICTLY on the following "Factual Evidence" extracted from the company's documents. Do not invent information.
        ---
        {formatted_facts}
        ---

        **Analysis and Writing Instructions:**
        1.  **Synthesize the Facts:** Review all the provided facts to form a holistic understanding.
        2.  **Identify Strengths and Weaknesses:** Based *only* on the evidence, what does the company do well regarding the primary question? What information is missing or appears to be a weakness?
        3.  **Write a Neutral Disclosure:** Draft a formal disclosure paragraph.
            - Start with a direct answer to the primary question if possible.
            - Present both the positive aspects (what the evidence shows they are doing) and the gaps (what the evidence does not show).
            - Maintain a professional, objective, and non-promotional ("no-hype") tone.
            - If the evidence is insufficient to answer the question, state that clearly. For example: "Based on the provided documents, there is insufficient information to determine the company's detailed policy on XYZ."

        **Final Disclosure Statement:**
        """
        # --- END: PROMPT ที่ปรับปรุงใหม่ ---

        # --- START OF FIX: แก้ไขการเรียกใช้ LLM ให้ถูกต้อง ---
        try:
            prompt_template = PromptTemplate.from_template(prompt_text)
            chain = prompt_template | self.llm

            response = await chain.ainvoke(
                {"main_question": main_question, "formatted_facts": formatted_facts},
                config={"configurable": {"temperature": 0.3}} # ใช้ temp ที่สูงขึ้นเล็กน้อยเพื่อการเขียนที่ลื่นไหล
            )
            return _extract_content_from_response(response)
            
        except Exception as e:
            self.logger.error(f"Disclosure Analyst agent failed for main question '{main_question}': {e}")
            return "Failed to generate disclosure due to an internal error."

    async def _generate_report_section(self, category_name: str, section_data: List[Dict]) -> str:
        self.logger.info(f"Generating detailed report section for: {category_name}")
        if not section_data:
            return f"Specific details on {category_name} policies and performance were not identified in the provided documents."
        structured_context = ""
        for item in section_data:
            question_data = item.get("question_data", {})
            final_disclosure = item.get("final_disclosure", "No disclosure generated.")
            retrieved_context = "\n".join(item.get("retrieved_context", []))
            structured_context += (
                f"### Topic: {question_data.get('theme')}\n\n"
                f"**Narrative Disclosure:**\n{final_disclosure}\n\n"
                f"**Supporting Raw Data from Document:**\n{retrieved_context}\n\n---\n"
            )
        prompt = PromptTemplate.from_template(
            """
            You are an expert ESG report writer and data analyst. Your task is to write a detailed, comprehensive, and well-structured narrative for the '{category_name}' section.
            **CRITICAL INSTRUCTIONS:**
            1. For each "Topic", use the "Narrative Disclosure" to write the main text.
            2. After writing the narrative for a topic, review the "Supporting Raw Data". **If you find specific, quantitative performance data (e.g., numbers, percentages, metrics from a specific year), you MUST create a summary in a well-formatted, clear Markdown Table.** This is not optional.
            3. You MUST keep the original `### Topic: ...` subheadings.
            4. Your entire output must be a professional, well-structured text with narratives and tables where appropriate.
            **Content and Data for {category_name}:**
            ---
            {structured_context}
            ---
            **Detailed {category_name} Section Narrative (in English, preserving subheadings and adding tables for all quantitative data):**
            """
        )
        chain = prompt | self.llm
        response = await chain.ainvoke({"category_name": category_name, "structured_context": structured_context})
        return _extract_content_from_response(response)

    async def _finalize_report(self, company_name: str, g_data: List[Dict], e_data: List[Dict], s_data: List[Dict]) -> str:
        self.logger.info(f"Assembling the final report for {company_name}...")

        def format_section(title: str, data: List[Dict]) -> str:
            if not data:
                # สร้างข้อความที่เป็นกลางเมื่อไม่มีข้อมูลสำหรับหมวดหมู่นั้นๆ
                return f"## {title}\n\nBased on the provided documents, specific details regarding the company's {title.lower()} policies and performance were not identified. For a complete analysis, please provide relevant documentation covering this area.\n"
            
            content = f"## {title}\n\n"
            # จัดกลุ่มหัวข้อที่มีธีมคล้ายกัน (ตัวอย่างการจัดกลุ่มแบบง่าย)
            for item in data:
                theme = item.get("question_data", {}).get("theme", "Uncategorized Topic")
                disclosure = item.get("final_disclosure", "No specific disclosure could be generated.")
                content += f"### {theme}\n{disclosure}\n\n"
            return content

        content_g = format_section("Governance (G)", g_data)
        content_e = format_section("Environmental (E)", e_data)
        content_s = format_section("Social (S)", s_data)

        finalizer_prompt = PromptTemplate.from_template(
            """
            You are an automated ESG report generation engine. Your sole task is to assemble a complete, professional, and coherent Markdown document based on the provided pre-written sections.

            **Instructions:**
            1.  Use the provided company name '{company_name}'.
            2.  Write a professional 2-paragraph "Message from the CEO" summarizing the company's commitment to sustainability, using the provided content as inspiration.
            3.  Write a brief "About This Report" section explaining the report's scope and methodology.
            4.  Assemble the pre-written sections for Governance, Environmental, and Social exactly as provided. Ensure seamless transitions between them.
            5.  Write a brief, forward-looking Conclusion summarizing the company's future commitments.
            6.  **DO NOT** create an Appendix or a section for missing information. The final output must be a clean, submittable report.

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
            {content_g}
            ---
            {content_e}
            ---
            {content_s}
            ---

            ## Conclusion
            (Your written forward-looking conclusion goes here.)
            """
        )
        chain = finalizer_prompt | self.llm
        response = await chain.ainvoke({
            "company_name": company_name,
            "content_g": content_g,
            "content_e": content_e,
            "content_s": content_s
        })
        return _extract_content_from_response(response)

    def _parse_sub_questions(self, sub_questions_text: str) -> List[str]:
        if not sub_questions_text: return []
        return [q.strip() for q in re.split(r'\n', sub_questions_text) if q.strip()]

# --- (ใหม่) Agent 6: Suggestions Generator ---
    async def _generate_suggestion_file(self, company_name: str, insufficient_items: List[Dict]) -> str:
        self.logger.info(f"Generating suggestion file for {company_name}...")
        if not insufficient_items:
            return "# Suggestions for Report Improvement\n\nExcellent! All topics were analyzed successfully with sufficient data. No further suggestions at this time."

        missing_topics_list = ""
        for item in insufficient_items:
            theme = item['question_data'].get('theme', 'Unknown Topic')
            category = item['question_data'].get('category', 'N/A')
            reason = item.get('final_disclosure', 'Reason not specified.')
            reason_clean = reason.replace("INSUFFICIENT DATA:", "").strip()
            missing_topics_list += f"- **Topic:** {theme} ({category})\n  - **Status:** Insufficient Data\n  - **Reason/Details:** {reason_clean}\n\n"

        # --- START OF FIX: เพิ่มคำสั่งให้ AI ไม่ต้องสร้างหัวข้อหลัก ---
        suggestion_prompt = PromptTemplate.from_template(
            """
            You are an expert ESG consultant. Your task is to create the BODY of a helpful and easy-to-understand suggestion document for '{company_name}'.

            **Analysis Results (Topics with Insufficient Data):**
            {missing_topics_list}

            **Your Task & Formatting Rules:**
            1.  Review the list of topics where data was insufficient.
            2.  For each topic, write a clear and actionable recommendation for the user using full, easy-to-understand sentences.
            3.  **IMPORTANT:** Start your response *directly* with the first recommendation section (e.g., `### Recommendations for...`). **DO NOT** add a main title (like '# Suggestions for...'), a preamble, or an introductory paragraph. The main title will be added automatically by the system.
            4.  Structure the entire output in clean Markdown.

            **Example of a Good Recommendation Structure:**
            ### Recommendations for Sustainable Materials & Resource Management
            * **What's Missing:** The analysis could not find a formal policy or a description of the governance structure...
            * **What to Provide Next Time:** To fix this, please provide an official policy document...

            ---
            **Suggestion Document BODY (Markdown format):**
            """
        )
        # --- END OF FIX ---

        chain = suggestion_prompt | self.llm
        response = await chain.ainvoke({
            "company_name": company_name,
            "missing_topics_list": missing_topics_list
        })

        # โค้ดส่วนนี้จะทำหน้าที่สร้างหัวข้อหลักเพียงที่เดียว
        return f"# ข้อเสนอแนะสำหรับการปรับปรุงรายงาน ESG ของบริษัท {company_name}\n\n" + _extract_content_from_response(response)

    # --- Main Orchestrator ---
    async def generate_sustainability_report(self, files: List[IO[bytes]], file_names: List[str], company_name: str) -> Dict[str, Any]:
        session_index_name = f"user-report-{uuid.uuid4().hex[:12]}"
        self.logger.info(f"Starting report generation for {company_name} with {len(file_names)} documents.")
        try:
            self.logger.info("Phase 1: Reading, splitting, and ingesting documents...")
            # --- START OF FIX: ปรับขั้นตอนการทำงาน ---
            # 1. อ่าน PDF ทุกหน้าด้วย Azure ก่อน
            documents_from_pages = await self._read_pages_with_azure(files, file_names)
            if not documents_from_pages:
                raise ValueError("No content could be extracted from the provided files using Azure DI.")
            
            # 2. นำหน้าที่อ่านได้มาตัดเป็น Chunks เล็กๆ
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            text_chunks = text_splitter.split_documents(documents_from_pages)
            if not text_chunks:
                raise ValueError("No chunks were created after splitting the document pages.")
            
            english_chunks = await self._translate_chunks_to_english(text_chunks)
            self.vector_store_service.initialize_vector_index(session_index_name, english_chunks)

            self.logger.info("Phase 2: Fetching Question AI set...")
            question_ai_set = await ESGQuestion.find(ESGQuestion.is_active == True).to_list()
            if not question_ai_set: raise ValueError("Question AI set is empty.")

            self.logger.info(f"Phase 3: Analyzing document against {len(question_ai_set)} questions...")
            raw_report_data = []
            for q_data in question_ai_set:
                main_question_text = q_data.main_question_text_en
                if not main_question_text: continue
                
                sub_questions_list = self._parse_sub_questions(q_data.sub_questions_sets[0].sub_question_text_en) if q_data.sub_questions_sets else []

                if not sub_questions_list:
                    answer_status, final_disclosure, detailed_facts, top_context = "insufficient", "The standard question is missing sub-questions.", [], []
                else:
                    comprehensive_query = main_question_text + " " + " ".join(sub_questions_list)
                    retrieved_content = self.vector_store_service.vector_search(session_index_name, comprehensive_query, top_k=100)
                    
                    valid_content_for_rerank = [text for text in retrieved_content if text and text.strip()]

                    if not valid_content_for_rerank:
                        self.logger.warning(f"No valid (non-empty) context found for question: '{main_question_text[:50]}...'")
                        top_context = []
                    else:
                        docs_for_rerank = [Document(page_content=text) for text in valid_content_for_rerank]
                        reranked_docs = self.reranker.compress_documents(documents=docs_for_rerank, query=main_question_text)
                        top_context = [doc.page_content for doc in reranked_docs]
                
                    detailed_facts = []
                    # Logic การตัดสินใจ
                    if top_context:
                        for sub_q in sub_questions_list:
                            extracted_facts = await self._extract_facts_for_sub_question(sub_q, top_context)
                            if extracted_facts:
                                detailed_facts.append({"sub_question": sub_q, "facts": extracted_facts})
                    
                    if detailed_facts:
                        answer_status = "sufficient"
                        final_disclosure = await self._analyze_facts_for_disclosure(main_question_text, detailed_facts)
                    else:
                        answer_status = "insufficient"
                        final_disclosure = "INSUFFICIENT DATA: Although some relevant context was found, no specific, verifiable facts could be extracted to directly answer the questions."

                raw_report_data.append({
                    "question_data": q_data.model_dump(by_alias=True),
                    "status": answer_status,
                    "final_disclosure": final_disclosure,
                    "detailed_facts": detailed_facts,
                    "retrieved_context": top_context[:5]
                })

            self.logger.info("Phase 4 & 5: Generating final reports and suggestions...")
            successful_items = [item for item in raw_report_data if item.get("status") == "sufficient"]
            insufficient_items = [item for item in raw_report_data if item.get("status") == "insufficient"]
            g_data = [item for item in successful_items if item['question_data']['category'] == 'G']
            e_data = [item for item in successful_items if item['question_data']['category'] == 'E']
            s_data = [item for item in successful_items if item['question_data']['category'] == 'S']
            
            final_markdown_report_en = await self._finalize_report(company_name, g_data, e_data, s_data)
            final_markdown_report_th = await self._translate_report_to_thai(final_markdown_report_en, company_name)
            
            suggestion_file_en = await self._generate_suggestion_file(company_name, insufficient_items)
            suggestion_file_th = await self._translate_report_to_thai(suggestion_file_en, company_name)

            return {
                "raw_data": raw_report_data,
                "markdown_report": final_markdown_report_th,
                "suggestion_report": suggestion_file_th,
                "processed_file_count": len(file_names)
            }
        finally:
            self.logger.info(f"Cleaning up vector data for index: {session_index_name}")
            self.vector_store_service.delete_index(session_index_name)