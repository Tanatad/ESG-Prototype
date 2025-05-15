from langsmith import Client
import matplotlib.pyplot as plt
from app.services.neo4j_service import Neo4jService
from langchain.schema import Document
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import tiktoken
from dotenv import load_dotenv
from typing import List
import time
import asyncio 


import os


load_dotenv()

class ScaleTestService:
    
    def __init__(self):
        self.neo4j_service = Neo4jService()
        self.neo4j_service.reinit_graph(url=os.getenv("TEST_NEO4J_URI"),
                                        username=os.getenv("TEST_NEO4J_USERNAME"),
                                        password=os.getenv("TEST_NEO4J_PASSWORD"))
        self.neo4j_service.reinit_vector(
                                        url=os.getenv("TEST_NEO4J_URI"),
                                        username=os.getenv("TEST_NEO4J_USERNAMER"),
                                        password=os.getenv("TEST_NEO4J_PASSWORD"))
    
    async def generate_docs_and_graph_docs(self, folder_path: str, target_chunks: int = 10, chunk_size: int = 4096, chunk_overlap: int = 1024):
        def merge_documents(docs: List[Document]) -> Document:
            combined_text = "\n\n".join([doc.page_content for doc in docs])
            merged_doc = Document(
                metadata={'source': 'merged_documents'},
                page_content=combined_text
            )
            return merged_doc
        def get_chunks_token(new_chunks: dict):
            encoding = tiktoken.encoding_for_model(os.getenv("NEO4J_GPT_MODEL","gpt-4o-mini"))
            chunk_token = {}
            for k, v in new_chunks.items():
                tokens = 0
                for doc in v:
                    tokens += len(encoding.encode(doc.page_content))
                chunk_token[k] = tokens
            return chunk_token
        
        file_list = os.listdir(folder_path)
        file_list = [os.path.join(folder_path, file) for file in file_list]
        docs = self.neo4j_service.read_PDFs_and_create_documents(
            [open(file, "rb") for file in file_list],
            file_list
        )
        merge_doc = merge_documents(docs)
        split_doc = await self.neo4j_service.split_documents_into_chunks([merge_doc], chunk_size, chunk_overlap)
        translate_split_doc = await self.neo4j_service.translate_documents_with_openai(split_doc)
        
        number_of_chunks = len(translate_split_doc)

        number_merge_chunks = number_of_chunks // target_chunks if number_of_chunks % target_chunks == 0 else number_of_chunks // target_chunks + 1

        new_chunks = {}
        for i in range(target_chunks):
            start = i * number_merge_chunks
            end = (i + 1) * number_merge_chunks
            new_chunks[f"step_{i}"] = translate_split_doc[start:end]
            
        print(get_chunks_token(new_chunks))
            
        all_graph_doc = {}
        for step, chunk in new_chunks.items():
            list_graph_doc = await self.neo4j_service.convert_documents_to_graph(chunk)
            all_graph_doc[step] = [x.model_dump() for x in list_graph_doc]
            
        new_chunks = {k: [x.model_dump() for x in v] for k, v in new_chunks.items()}
                    
        return new_chunks, all_graph_doc

    async def run_scale_test(self, new_chunks: dict, graph_doc: dict, question_list: List[str], n_relate: int = 10):
        def clear_database():
            """
            Clears the database by deleting all nodes and relationships.
            """
            query = "MATCH (n) DETACH DELETE n"
            self.neo4j_service.graph.query(query)
            self.neo4j_service.graph.query("CALL apoc.schema.assert({}, {})")
            
        async def run_multiple_cypher(query: list):
            results = []
            tasks = []
            for q in query:
                task = asyncio.create_task(self.neo4j_service.get_cypher_answer(q))
                tasks.append(task)
                if len(tasks) == 10:
                    results.extend(await asyncio.gather(*tasks))
                    tasks = []
            if tasks:
                results.extend(await asyncio.gather(*tasks))
            return results
        
        
        clear_database()
        
        encoding = tiktoken.encoding_for_model(os.getenv("NEO4J_GPT_MODEL","gpt-4o-mini"))
        
        relate_latency = []
        relate_token = []
        
        print("Your data will be save in Langsmith name:", os.getenv("LANGCHAIN_PROJECT"))
        for step in graph_doc.keys():
            current_chunk = new_chunks[step]
            current_graph = graph_doc[step]

            # Add Document and Graph to the Neo4j
            self.neo4j_service.add_graph_documents_to_neo4j(current_graph)
            self.neo4j_service.add_description_embeddings_to_nodes()
            await self.neo4j_service.store_vector_embeddings(current_chunk)
                
            # Ask Questions
            await run_multiple_cypher(question_list)
            
            for i in range(len(question_list)):
                start = time.time()
                relate_doc = await self.neo4j_service.get_relate(question_list[i], n_relate)
                end = time.time()
                used_time = end - start
                token = 0
                for doc in relate_doc:
                    token += len(encoding.encode(doc.page_content))
                relate_latency.append(used_time)
                relate_token.append(token)
                
        return relate_latency, relate_token
                
    def load_data_from_langsmith(self):
        client = Client()
        start_time = datetime.now() - timedelta(days=7)

        runs = list(
            client.list_runs(
                project_name=os.getenv("LANGCHAIN_PROJECT"),
                run_type="llm",
                start_time=start_time,
            )
        )
        
        df = pd.DataFrame(
            [
                {
                    "name": run.name,
                    "model": run.extra["invocation_params"][
                        "model"
                    ],  # The parameters used when invoking the model are nested in the extra info
                    **run.inputs,
                    **(run.outputs or {}),
                    "error": run.error,
                    "latency": (run.end_time - run.start_time).total_seconds()
                    if run.end_time
                    else None,  # Pending runs have no end time
                    "prompt_tokens": run.prompt_tokens,
                    "completion_tokens": run.completion_tokens,
                    "total_tokens": run.total_tokens,
                }
                for run in runs
            ],
            index=[run.id for run in runs],
        )
        
        return df
        
    def get_metrics(self, df: pd.DataFrame, max_tokens: int, n_question: int, relate_token: list, relate_latency: list, n_chunks: int = 10):
        df_values = df.values[::-1]
        all_chunk = np.arange(1, n_chunks + 1)

        GraphCypherQa_Token = [sum(x)/n_question for x in df_values[:, -1].reshape(n_chunks, n_question)]
        GraphCypherQa_Latency = [sum(x)/n_question for x in df_values[:, -4].reshape(n_chunks, n_question)]
        
        plt.plot(all_chunk, GraphCypherQa_Token)
        plt.title("Average Cypher Token per Chunk")
        plt.xlabel("#Chunk")
        plt.ylabel("Token")
        plt.show()
        
        plt.plot(all_chunk, GraphCypherQa_Latency, label="GraphCypherQa")
        plt.title("Average Cypher Latency per Chunk")
        plt.xlabel("#Chunk")
        plt.ylabel("Latency (s)")
        plt.show()
        
        relate_token = np.array(relate_token)
        relate_latency = np.array(relate_latency)
        
        GetRelated_Token = [sum(x)/n_question for x in relate_token.reshape(n_chunks, n_question)]
        GetRelated_Latency = [sum(x)/n_question for x in relate_latency.reshape(n_chunks, n_question)]

        plt.plot(all_chunk, GetRelated_Token, label="GetRelated Token")
        plt.title("Average Get_relate Token per Chunk")
        plt.xlabel("Chunk")
        plt.ylabel("Token")
        plt.show()
        
        plt.plot(all_chunk, GetRelated_Latency, label="GetRelated Latency")
        plt.title("Average Get_relate Latency per Chunk")
        plt.xlabel("Chunk")
        plt.ylabel("Latency (s)")
        plt.show()