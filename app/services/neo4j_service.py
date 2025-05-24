import re
import os
import asyncio 
import inspect
import traceback
from typing import IO, List
from dotenv import load_dotenv
from langchain_community.graphs import Neo4jGraph
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain_core.documents import Document
from langchain.chains import GraphCypherQAChain
from langchain_unstructured import UnstructuredLoader
from langchain_community.vectorstores import Neo4jVector
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.graphs.graph_document import GraphDocument
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_cohere import CohereEmbeddings
from app.services.rate_limit import llm_rate_limiter
from pydantic import BaseModel
load_dotenv()

placeholder_data = [
    Document(page_content="""ESG stands for Environment, Social, and Governance. 

    - **Environment** refers to how a company manages its impact on the natural world, including issues like climate change, resource consumption, pollution, and biodiversity.
    
    - **Social** refers to how a company manages relationships with its employees, suppliers, customers, and the communities where it operates. This includes things like employee rights, diversity, and community engagement.
    
    - **Governance** refers to the company's leadership structure, transparency, ethical behavior, and how it aligns with regulations and stakeholder interests. This includes corporate governance practices, anti-corruption measures, and shareholder rights.
    
    ESG factors are used by investors, regulators, and organizations to assess the long-term sustainability of companies and projects, with an emphasis on responsible investing, environmental impact, and social fairness.
    """)
]

class RetrieveAnswer(BaseModel):
    cypher_answer: str
    relate_documents: List[Document]
        
class Neo4jService:
    def __init__(self):
        self.graph_chain = None
        self.store = None
        self.graph = None
        self.environment = os.getenv("ENVIRONMENT","development")
        self.llm = ChatGoogleGenerativeAI(
            temperature=0,
            model=os.getenv("NEO4J_MODEL","gemini-2.0-flash"),
            max_retries=10,
            rate_limiter=llm_rate_limiter
        )
        self.llm_embbedding = CohereEmbeddings(
            model='embed-v4.0',
            cohere_api_key=os.getenv("COHERE_API_KEY"),
            max_retries=10
        )
        self.init_graph()
        self.init_vector()
        
    def init_graph(self, url = None, username = None, password = None):
        if not self.graph:
            self.graph = Neo4jGraph(
                url=url or os.environ["NEO4J_URI"],
                username=username or os.environ["NEO4J_USERNAME"],
                password=password or os.environ["NEO4J_PASSWORD"],
            )
            
    def reinit_graph(self, url = None, username = None, password = None):
        self.graph = Neo4jGraph(
            url=url or os.environ["NEO4J_URI"],
            username=username or os.environ["NEO4J_USERNAME"],
            password=password or os.environ["NEO4J_PASSWORD"]
        )
    
    def init_vector(self):
        index_name = "vector"
        try:
            # Attempt to initialize from an existing index
            store = Neo4jVector.from_existing_index(
                self.llm_embbedding,
                url=os.environ["NEO4J_URI"],
                username=os.environ["NEO4J_USERNAME"],
                password=os.environ["NEO4J_PASSWORD"],
                index_name=index_name,
            )
            self.store = store
            print(f"Vector index '{index_name}' initialized successfully.")
        except Exception as e:
            store = Neo4jVector.from_documents(
                placeholder_data, 
                self.llm_embbedding,
                url=os.environ["NEO4J_URI"],
                username=os.environ["NEO4J_USERNAME"],
                password=os.environ["NEO4J_PASSWORD"],
            )
            self.store = store
            print(f"Vector index '{index_name}' initialized successfully.")
        
            
    def reinit_vector(self, url=None, username=None, password=None):
        index_name = "vector"
        try:
            # Attempt to initialize from the existing index
            store = Neo4jVector.from_existing_index(
                self.llm_embbedding,
                url=url or os.environ["NEO4J_URI"],
                username=username or os.environ["NEO4J_USERNAME"],
                password=password or os.environ["NEO4J_PASSWORD"],
                index_name=index_name,
            )
            self.store = store
            print(f"Reinitialized vector store with index '{index_name}' successfully.")
        except Exception as e:
            store = Neo4jVector.from_documents(
                placeholder_data, 
                self.llm_embbedding,
                url=url or os.environ["NEO4J_URI"],
                username=username or os.environ["NEO4J_USERNAME"],
                password=password or os.environ["NEO4J_PASSWORD"],
            )
            self.store = store
            print(f"Vector index '{index_name}' initialized successfully.")

    def read_PDFs_and_create_documents(self, files: List[IO[bytes]], file_names: List[str], chunking_strategy: str = "basic", max_characters: float = 1e6, include_orig_elements: bool = False):
        # Design decision: one document per file
        docs = []
        
        for i in range(len(file_names)):
            loader = UnstructuredLoader(file=files[i], 
                                    chunking_strategy=chunking_strategy, 
                                    max_characters=max_characters,
                                    include_orig_elements=include_orig_elements,
                                    metadata_filename=file_names[i]
                                    )
            
            docs += loader.load()
            
        return docs
    
    async def split_documents_into_chunks(self, documents: List[Document], chunk_size: int = 2048, chunk_overlap: int = 1024):
        splitter = RecursiveCharacterTextSplitter(
                            chunk_size=chunk_size, 
                            chunk_overlap=chunk_overlap
                )
        return await asyncio.to_thread(splitter.split_documents, documents)
    
    async def translate_documents_with_openai(self, documents: List[Document], batch_size: int = 20):
        def contains_thai(text: str) -> bool:
            # Regular expression to match Thai characters (Unicode range: \u0E00-\u0E7F)
            thai_pattern = re.compile("[\u0E00-\u0E7F]")
            return bool(thai_pattern.search(text))
    
        # Initialize the LLM
        llm = self.llm
        
        # Define the translation prompt template
        prompt = PromptTemplate(
            input_variables=["text"],
            template="""
                You are a translator.
                Translate Thai text into English.
                NO NEED FOR ANYTHING EXCEPT FOR A TRANSLATED TEXT.
                {text}"""
        )
                
        chain = prompt | llm
                
        # Define an async function to run the chain
        async def run_chain(text):
            # No handle error for now
            if not contains_thai(text):
                return text
            response = await chain.ainvoke({"text": text})
            return response.content

        # Define a function to run multiple chains concurrently
        async def run_multiple_chains(texts):
            results = []
            tasks = []
            for i, text in enumerate(texts):
                task = asyncio.create_task(run_chain(text))
                tasks.append(task)
                if len(tasks) == batch_size:
                    results.extend(await asyncio.gather(*tasks))
                    tasks = []  
            if tasks:
                results.extend(await asyncio.gather(*tasks))
            return results

        input_texts = [doc.page_content for doc in documents]
        
        results = await run_multiple_chains(input_texts)
        
        for i, result in enumerate(results):
            if result:
                documents[i].page_content = result
        
        return documents
    
    def __get_auth_env(self):
        if self.environment == "test":
            return os.environ["TEST_NEO4J_URI"], os.environ["TEST_NEO4J_USERNAME"], os.environ["TEST_NEO4J_PASSWORD"]
        return os.environ["NEO4J_URI"], os.environ["NEO4J_USERNAME"], os.environ["NEO4J_PASSWORD"]
    
    def add_description_embeddings_to_nodes(self):
        print("Adding description embeddings to nodes...")
        
        url, username, password = self.__get_auth_env()
        Neo4jVector.from_existing_graph(
            self.llm_embbedding,
            node_label='__Entity__',
            text_node_properties=['id','description'],
            embedding_node_property="description_embedding",
            index_name="description",
            url=url,
            username=username,
            password=password,
        )
        print("Description embeddings added to nodes.")
        
    async def convert_documents_to_graph(self, documents: List[Document]):
        llm = self.llm
        llm_transformer = LLMGraphTransformer(
            llm=llm,
            node_properties=["description"],
            relationship_properties=["description"],
            strict_mode=True
        )
                
        graph_documents = await llm_transformer.aconvert_to_graph_documents(documents)
        return graph_documents
    
    def add_graph_documents_to_neo4j(self, graph_documents: List[GraphDocument]):
        try:
            self.graph.add_graph_documents(
                graph_documents,
                baseEntityLabel=True,
                include_source=True,
            )
        except Exception as e_db:
                print(f"ERROR during Neo4j operation in {self.__class__.__name__}.{inspect.currentframe().f_code.co_name}: {e_db}")
                traceback.print_exc()
                
        
    def init_cypher(self):
        graph_chain = GraphCypherQAChain.from_llm(
            self.llm, 
            graph=self.graph,  
            allow_dangerous_requests=True,
            verbose=False,
            validate_cypher=True
        )
        self.graph_chain = graph_chain
    
    async def store_vector_embeddings(self, documents: List[Document]):
        try:
            await self.store.aadd_documents(documents)
        except Exception as e_db:
            print(f"ERROR during Neo4j operation in {self.__class__.__name__}.{inspect.currentframe().f_code.co_name}: {e_db}")
            traceback.print_exc()
        
            
    async def get_relate(self, query, k):                    
        # if error occurs, return empty list
        results = []
        try:
            docs_with_score = await self.store.asimilarity_search_with_score(query, k=k)
            for doc, score in docs_with_score:
                results.append(doc)
        except:
            pass
        return results
    
    async def get_cypher_answer(self, query):
        # If error occurs, return empty string
        url, username, password = self.__get_auth_env()
        retrieval_query = """
            OPTIONAL MATCH (node)-[rel*1..2]-(neighbor)
            WITH node, score, collect({
                relationship: [r IN rel | type(r)], 
                relationship_properties: [r IN rel | properties(r)], 
                neighbor: neighbor {.*, description_embedding: Null}
            })[0..50] AS neighbors_and_relations
            RETURN coalesce(node.description, "No description available") AS text,
                score,
                node {.*, description_embedding: Null, neighbors_and_relations: neighbors_and_relations} AS metadata
        """
        existing_index_return = Neo4jVector.from_existing_index(
                                    self.llm_embbedding,
                                    url=url,
                                    username=username,
                                    password=password,
                                    index_name="description",
                                    text_node_property="description",
                                    retrieval_query=retrieval_query)
        docs = existing_index_return.similarity_search(query=query, k=3)
          
        return f"{docs}"
    
    async def get_output(self, query, k=5) -> RetrieveAnswer:
        cypher_answer = await self.get_cypher_answer(query)
        relate_documents = await self.get_relate(query, k)
        if cypher_answer == "I don't know the answer.":
            cypher_answer = ""
        return RetrieveAnswer(
            cypher_answer=cypher_answer,
            relate_documents=relate_documents
            )
    
    async def is_graph_substantially_empty(self, threshold: int = 10) -> bool:
        """
        Checks if the Neo4j graph has a very low number of nodes.
        Relies on self.graph (Neo4jGraph instance) being initialized.
        """
        if not self.graph: # self.graph คือ instance ของ Neo4jGraph จาก Langchain
            print("[NEO4J_SERVICE ERROR] Neo4jGraph (self.graph) is not initialized. Cannot check if empty.")
            # ตัดสินใจว่าจะ return True (ถือว่าว่าง) หรือ False (ถือว่าไม่ว่าง) ในกรณีนี้
            # การ return True อาจจะปลอดภัยกว่าถ้า logic การสร้างคำถามครั้งแรกขึ้นกับสิ่งนี้
            return True 

        query = "MATCH (n) RETURN count(n) AS node_count"
        
        try:
            # Neo4jGraph.query() ของ Langchain เป็น synchronous call
            # เมื่อเรียกจาก async method เราต้องใช้ run_in_executor
            loop = asyncio.get_running_loop() # หรือ asyncio.get_event_loop() ใน Python < 3.7
            result_data = await loop.run_in_executor(None, self.graph.query, query)
            # self.graph.query(query) จะ return list of dictionaries, เช่น [{'node_count': 123}]
            
            if result_data and len(result_data) > 0 and "node_count" in result_data[0]:
                node_count = result_data[0]["node_count"]
                print(f"[NEO4J_SERVICE LOG] Current node count in graph: {node_count}")
                return node_count < threshold
            else:
                # กรณีที่ query สำเร็จแต่ไม่ได้ผลลัพธ์ตามที่คาดหวัง
                print(f"[NEO4J_SERVICE WARNING] Could not retrieve valid node count. Query result: {result_data}. Assuming graph is not empty for safety.")
                return False 
        except Exception as e:
            print(f"[NEO4J_SERVICE ERROR] Error checking if graph is empty: {e}")
            # ในกรณีที่เกิด error ขณะ query, การตัดสินใจว่าจะ return True หรือ False ขึ้นอยู่กับว่า
            # การ trigger Full Regeneration โดยไม่ได้ตั้งใจนั้นเสียหายกว่าการไม่ trigger หรือไม่
            # การ return False (ถือว่าไม่ว่าง) อาจจะปลอดภัยกว่าในแง่ของการป้องกันการล้างข้อมูลคำถามโดยไม่จำเป็น
            print("[NEO4J_SERVICE WARNING] Assuming graph is NOT empty due to error, to avoid accidental full question regeneration.")
            return False
        
    async def flow(self, files: List[IO[bytes]], file_names: List[str]):
        loop = asyncio.get_running_loop()

        # Load PDFs and create translate chunk-documents
        # documents = self.read_PDFs_and_create_documents(files, file_names) # Original
        documents = await loop.run_in_executor(None, self.read_PDFs_and_create_documents, files, file_names)

        split_docs = await self.split_documents_into_chunks(documents) # This is already async friendly if splitter.split_documents is designed so
        translated_split_doc = await self.translate_documents_with_openai(split_docs) # This is already async

        # Convert documents to graph and store in Neo4j
        graph_doc = await self.convert_documents_to_graph(translated_split_doc) # This is already async

        # await loop.run_in_executor(None, self.add_graph_documents_to_neo4j, graph_doc) # Wrap blocking call
        # Corrected based on your code structure: add_graph_documents_to_neo4j is synchronous
        await loop.run_in_executor(None, self.add_graph_documents_to_neo4j, graph_doc)


        # await loop.run_in_executor(None, self.add_description_embeddings_to_nodes) # Wrap blocking call
        # Corrected based on your code structure: add_description_embeddings_to_nodes is synchronous
        await loop.run_in_executor(None, self.add_description_embeddings_to_nodes)

        # Store vector embeddings
        await self.store_vector_embeddings(translated_split_doc) # This is already async
           