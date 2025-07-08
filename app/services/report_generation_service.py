import os
import uuid
import logging
import json
from typing import List, IO, Dict
from langchain_core.documents import Document
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate

# --- FIX: Import the Beanie Document model directly ---
from app.models.esg_question_model import ESGQuestion
# ----------------------------------------------------
from langchain_cohere import CohereRerank
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
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
        self.mongodb_service = mongodb_service # Keep for other potential uses
        self.pinecone_service = pinecone_service
        self.neo4j_service = neo4j_service
        self.llm = llm
        self.reranker = CohereRerank(
            cohere_api_key=os.getenv("COHERE_API_KEY"),
            model="rerank-english-v3.0",
            top_n=20
        )
        
        self.answer_generation_prompt = PromptTemplate.from_template(
            """
            Based ONLY on the following context retrieved from the user's document, answer the question.
            If the context is insufficient or does not contain the answer, state 'The provided document does not contain sufficient information to answer this question.'.
            Do not use any external knowledge.

            Context:
            ---
            {context}
            ---
            Question: {question}
            Answer:
            """
        )
        self.answer_generation_chain = self.answer_generation_prompt | self.llm
        self.logger.info("ReportGenerationService initialized.")

    async def _generate_answer_for_question(self, question: str, context_chunks: List[str]) -> str:
        """Uses an LLM to generate an answer for a single question based on context."""
        if not context_chunks:
            return "No relevant information found in the document for this question."
        
        context_string = "\n\n---\n\n".join(context_chunks)
        
        response = await self.answer_generation_chain.ainvoke({
            "context": context_string,
            "question": question
        })
        
        return response.content.strip()

    async def generate_sustainability_report(self, files: List[IO[bytes]], file_names: List[str]) -> List[Dict]:
        """
        Main method to orchestrate the entire report generation process.
        """
        session_index_name = f"user-report-{uuid.uuid4().hex[:12]}"
        self.logger.info(f"Starting report generation. Using temporary Pinecone index: {session_index_name}")

        try:
            # Process PDFs and upsert to Pinecone
            self.logger.info(f"Processing {len(files)} PDF(s)...")
            initial_docs = await self.neo4j_service.read_PDFs_and_create_documents_azure(files, file_names)
            text_chunks = await self.neo4j_service.split_documents_into_chunks(initial_docs)
            self.logger.info(f"Uploading {len(text_chunks)} chunks to Pinecone index '{session_index_name}'...")
            self.pinecone_service.upsert_documents(session_index_name, text_chunks)
            
            # Fetch Question AI set
            self.logger.info("Fetching Question AI set from MongoDB using Beanie...")
            question_ai_set = await ESGQuestion.find(ESGQuestion.is_active == True).to_list()
            if not question_ai_set:
                raise ValueError("Question AI set is empty in the database. Cannot generate report.")
                
            # Iterate through questions
            self.logger.info(f"Generating answers for {len(question_ai_set)} questions...")
            report_data = []
            for q_data in question_ai_set:
                question_text = q_data.main_question_text_en
                if not question_text:
                    continue

                # Step 1: Retrieve and Rerank
                initial_context = self.pinecone_service.query(session_index_name, question_text, top_k=50)
                docs_for_rerank = [Document(page_content=text) for text in initial_context]
                reranked_docs = self.reranker.compress_documents(documents=docs_for_rerank, query=question_text)
                top_20_chunks = [doc.page_content for doc in reranked_docs]
                
                # Step 2: Validate Context
                validation_result = await self._validate_context(question_text, top_20_chunks)
                
                # Step 3: Generate Disclosure or Suggestion
                answer_status = validation_result.get("status", "insufficient")
                if answer_status == "sufficient":
                    final_answer = await self._generate_disclosure_statement(question_text, top_20_chunks)
                else:
                    final_answer = f"INSUFFICIENT DATA: {validation_result.get('reason')}"

                # Step 4: Store results
                report_data.append({
                    "question_data": q_data.model_dump(by_alias=True),
                    "status": answer_status,
                    "answer_or_suggestion": final_answer,
                    "retrieved_context": top_20_chunks
                })

            self.logger.info("Report data generation complete.")
            return report_data

        finally:
            # Clean up
            self.logger.info(f"Cleaning up temporary index: {session_index_name}")
            self.pinecone_service.delete_index(session_index_name)

    async def _validate_context(self, question: str, context_chunks: List[str]) -> Dict:
        """
        Uses an LLM to validate if the context is sufficient to answer the question.
        Returns a dictionary like: {"status": "sufficient" | "insufficient", "reason": "..."}
        """
        if not context_chunks:
            return {"status": "insufficient", "reason": "No relevant context was found in the document."}

        context_string = "\n\n---\n\n".join(context_chunks)
        
        validator_prompt = PromptTemplate.from_template(
            """
            You are an expert ESG compliance auditor. Your task is to determine if the given 'Context' contains enough information to fully answer the 'Question'.

            Question: {question}

            Context:
            ---
            {context}
            ---

            Based ONLY on the provided context, is there sufficient, specific, and clear information to answer the question thoroughly?
            Respond with ONLY a single, valid JSON object with two keys:
            1. "status": either "sufficient" or "insufficient".
            2. "reason": a brief, one-sentence explanation for your decision.
            """
        )
        validator_chain = validator_prompt | self.llm
        
        try:
            response = await validator_chain.ainvoke({"question": question, "context": context_string})
            # A more robust way to handle potential non-JSON responses from LLM
            json_str = response.content[response.content.find('{'):response.content.rfind('}')+1]
            result = json.loads(json_str)
            return result
        except Exception:
            return {"status": "insufficient", "reason": "Could not determine sufficiency from the context."}
    
    async def _generate_disclosure_statement(self, question: str, context_chunks: List[str]) -> str:
        """
        Uses an LLM to draft a formal disclosure statement for an ESG report.
        """
        context_string = "\n\n---\n\n".join(context_chunks)
        
        writer_prompt = PromptTemplate.from_template(
            """
            You are a professional sustainability report writer. Based on the following 'Context' from a company's internal documents, draft a formal, clear, and comprehensive disclosure statement that answers the 'Question'.
            Write in a professional, corporate tone suitable for an official ESG report.

            Question: {question}

            Context:
            ---
            {context}
            ---

            Disclosure Statement:
            """
        )
        writer_chain = writer_prompt | self.llm
        response = await writer_chain.ainvoke({"question": question, "context": context_string})
        return response.content.strip()
