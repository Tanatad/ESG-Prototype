
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
import aiofiles
from datasets import load_dataset
from app.utils.retry import retry_on_exception
from app.services.chat_service import ChatService
from app.services.neo4j_service import Neo4jService
from app.services.persistence.mongodb import AsyncMongoDBSaver
from typing import IO, List, Tuple
from datasets import Dataset
from ragas.metrics import LLMContextRecall, Faithfulness, FactualCorrectness, SemanticSimilarity
from ragas import evaluate
from langchain_core.messages import HumanMessage
import uuid
from enum import Enum
from app.services.chat_service import ContextType
class TestPerformanceService:
    
    def __init__(self, chat_service: ChatService, Neo4jService: Neo4jService):
        self.evaluator_llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o-mini"))
        self.evaluator_embeddings = LangchainEmbeddingsWrapper(OpenAIEmbeddings())
        self.chat_service = chat_service
        self.neo4j_service = Neo4jService
    @staticmethod
    async def create():
        """
        Asynchronously initialize TestPerformanceService with dependencies.
        """
        # Initialize Neo4jService
        neo4j_service = Neo4jService()
        neo4j_service.reinit_graph(url=os.getenv("TEST_NEO4J_URI"),
                                        username=os.getenv("TEST_NEO4J_USERNAME"),
                                        password=os.getenv("TEST_NEO4J_PASSWORD"))
        neo4j_service.reinit_vector(
                                        url=os.getenv("TEST_NEO4J_URI"),
                                        username=os.getenv("TEST_NEO4J_USERNAMER"),
                                        password=os.getenv("TEST_NEO4J_PASSWORD"))
        # Initialize AsyncMongoDBSaver for memory
        url = os.getenv("MONGO_URL") or None
        db_name = os.getenv("MONGO_DB_NAME") or None
        memory = await AsyncMongoDBSaver.from_conn_info_async(url, db_name)

        # Initialize ChatService
        chat_service = ChatService(memory, neo4j_service)
        # Return TestPerformanceService instance
        return TestPerformanceService(chat_service, neo4j_service)

    def clear_database(self):
        """
        Clears the database by deleting all nodes and relationships.
        """
        query = "MATCH (n) DETACH DELETE n"
        self.neo4j_service.graph.query(query)
        self.neo4j_service.graph.query("CALL apoc.schema.assert({}, {})")
        
    def read_files_from_path(self, directory_path: str) -> Tuple[List[IO[bytes]],List[str]]:
        file_objects = []
        file_names = []
        for root, dirs, files in os.walk(directory_path):
            for i,file in enumerate(files):
                if file.endswith(".pdf"):
                    file_objects.append(open(os.path.join(root, file), "rb"))
                    file_names.append("files_{i}_" + file)      
        return file_objects, file_names
    
    async def test_flow_all(self, folder_pdf_path: str,
                            test_dataset_path: str,
                            output_folder_jsonl: str, batch = 10,
                            context_types: list[ContextType] = None ):
        
        if not os.path.exists(folder_pdf_path):
            raise FileNotFoundError(f"Directory not found: {folder_pdf_path}")
        if not os.path.exists(test_dataset_path):
            raise FileNotFoundError(f"Dataset file not found: {test_dataset_path}")
        if not os.path.exists(output_folder_jsonl):
            os.makedirs(output_folder_jsonl)
        
        if context_types is None:
            context_types = [
                ContextType.BOTH, 
                ContextType.CYPHER, 
                ContextType.VECTOR, 
                ContextType.EMPTY
            ]

        self.clear_database()
        files, file_names = self.read_files_from_path(folder_pdf_path)
        await self.neo4j_service.flow(files, file_names)
        
        averages = {}

        for context_type in context_types:
            results = await self.test_with_context_type(test_dataset_path,
                                                        context_type.value,
                                                        output_folder_jsonl,
                                                        batch)
            averages[context_type.value] = results

        return averages
        
    async def test_with_context_type(self, test_dataset_path: str, context_type: str,  output_folder_jsonl: str, batch = 10):
        test_dataset = await self.get_agent_response(test_dataset_path, context_type, batch)
        results = await self.test_flow(test_dataset, batch)
        results = results.to_pandas()
        print(f"Test flow {context_type} started...")
        output = os.path.join(output_folder_jsonl, f"output_{context_type}.jsonl")
        results.to_json(output, orient="records", lines=True, force_ascii=False)
        numerical_columns = results.select_dtypes(include=["float64", "int64"])
        averages = numerical_columns.mean() 
        print(f"Average scores for numerical columns with context_type {context_type}: ")
        print(averages)
        return averages
        
    async def get_agent_response(self, test_dataset: str, context_type="both", batch=10) -> Dataset:
        test_dataset = load_dataset("json", data_files={"test": test_dataset}, split="test")
        graph = self.chat_service.graph  
        thread_ids = []  

        for _ in range(len(test_dataset)):
            thread_id = str(uuid.uuid4())
            thread_ids.append({"configurable": {"thread_id": thread_id}})

        # Prepare lists to collect answers and contexts
        answers = []
        retrieved_contexts = []

        for i in range(0, len(test_dataset), batch):
            batch_user_input = [
                test_dataset[j]["question"] 
                for j in range(i, min(i + batch, len(test_dataset)))
            ]

            batch_config = thread_ids[i : i + batch]
            
            batch_answer = await graph.abatch(
                [{"messages": [HumanMessage(content=user_input)], "context_type": ContextType(context_type)} for user_input in batch_user_input],
                config=batch_config
            )
            batch_retriever = await asyncio.gather(
                *[self.neo4j_service.get_output(question, k=10) for question in batch_user_input]
            )
            
            for j, answer in enumerate(batch_answer):
                # Collect the answer content
                answers.append(answer["messages"][-1].content)
                # Collect the retrieved contexts
                cypher_answer = batch_retriever[j].cypher_answer
                related_documents = [doc.page_content for doc in batch_retriever[j].relate_documents]
                context = []
                if context_type == "both":
                    context = ([cypher_answer] if cypher_answer else []) + related_documents
                elif context_type == "cypher":
                    context = [cypher_answer] if cypher_answer else []
                elif context_type == "documents":
                    context = related_documents
                elif context_type == "empty":
                    context = []
                else:
                    raise ValueError(f"Invalid context_type: {context_type}")
                retrieved_contexts.append(context)
                
        test_dataset = test_dataset.add_column('answer', answers)
        test_dataset = test_dataset.add_column('contexts', retrieved_contexts)
                    
        # print(test_dataset[0])
        for thread_id in thread_ids:
            try:
                await self.chat_service.delete_by_thread_id(thread_id["configurable"]["thread_id"])
            except Exception as e:
                print(f"Failed to delete thread ID {thread_id}: {e}")
                
        return test_dataset
    
    async def test_flow(self, test_dataset: Dataset, batch = 10 ):
        print("Dataset features:", test_dataset.features)
        def convert_reference_to_string(example):
            if isinstance(example['reference'], list):  # Check if it's a list
                example['reference'] = " ".join(example['reference'])  # Join the list into a single string
            return example
        
        test_dataset = test_dataset.map(convert_reference_to_string)
        
        metrics = [
            LLMContextRecall(llm=self.evaluator_llm), 
            FactualCorrectness(llm=self.evaluator_llm), 
            Faithfulness(llm=self.evaluator_llm),
            SemanticSimilarity(embeddings=self.evaluator_embeddings)
        ]
        print("Evaluating model performance...")
        results = evaluate(dataset=test_dataset, metrics=metrics)
        print("Results:")
        return results