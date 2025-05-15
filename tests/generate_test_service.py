
from tenacity import retry, stop_after_attempt, wait_fixed
from functools import wraps
from azure.core.credentials import AzureKeyCredential
from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.ai.documentintelligence.models import AnalyzeOutputOption, AnalyzeResult, ContentFormat
from azure.core.exceptions import HttpResponseError
import json
from langchain_core.documents import Document
import asyncio
from typing import List,AsyncGenerator
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
import os
from ragas.testset import TestsetGenerator
from app.utils.retry import retry_on_exception
from langchain_text_splitters import RecursiveCharacterTextSplitter 

class GenerateTestService:
    
    def __init__(self):
        self.generator_llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o-mini"))
        self.generator_embeddings = LangchainEmbeddingsWrapper(OpenAIEmbeddings())
    
    def read_single_pdf_to_markdown(self, file_path: str) -> Document:
        endpoint = os.getenv("DOCUMENT_INTELLIGENCE_ENDPOINT")
        key = os.getenv("DOCUMENT_INTELLIGENCE_KEY")
        credential = AzureKeyCredential(key)
        document_intelligence_client = DocumentIntelligenceClient(endpoint, credential)
        with open(file_path, "rb") as f:
                poller = document_intelligence_client.begin_analyze_document(
                    "prebuilt-read",
                    analyze_request=f,
                    content_type="application/octet-stream",
                    output_content_format=ContentFormat.MARKDOWN
                )
        result: AnalyzeResult = poller.result()
        content = result.content
        return Document(page_content=content)
                    
    @retry_on_exception                 
    async def aread_pdf_to_markdown(self, file_path: str) -> Document:
        return await asyncio.to_thread(self.read_single_pdf_to_markdown, file_path)
    
    async def read_folder_to_documents(self, folder_path: str) -> List[Document]:
        documents = []
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                if file.endswith(".pdf"):
                    try:
                        document = await self.aread_pdf_to_markdown(os.path.join(root, file))
                        documents.append(document)
                    except Exception as e:
                        print(f"Failed to process {file}: {e}")
        return documents
    
    async def split_documents_into_chunks(self, documents: List[Document], chunk_size: int = 4096, chunk_overlap: int = 1024):
        splitter = RecursiveCharacterTextSplitter(
                            chunk_size=chunk_size, 
                            chunk_overlap=chunk_overlap
                )
        return await asyncio.to_thread(splitter.split_documents, documents)
    
    async def generate_testset(self, folder_path: str, output_path: str):
        docs = await self.read_folder_to_documents(folder_path)
        split_docs = await self.split_documents_into_chunks(docs, chunk_size=40000, chunk_overlap=0)
        generator = TestsetGenerator(llm=self.generator_llm, embedding_model=self.generator_embeddings)
        dataset = generator.generate_with_langchain_docs(split_docs, testset_size=50)
        print("example",dataset)
        dataset = dataset.to_evaluation_dataset().to_hf_dataset()
        dataset = dataset.rename_columns( column_mapping={"user_input": "question"})
        dataset = dataset.remove_columns("reference_contexts")
        dataset.to_json(output_path, orient="records", lines=True, force_ascii=False)
        return dataset
        
        