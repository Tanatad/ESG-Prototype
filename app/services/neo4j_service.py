import re
import os
import asyncio
import inspect
import logging
import traceback
import functools
import neo4j
from typing import IO, List, Optional, Dict, Any, Tuple
from dotenv import load_dotenv
from langchain_community.graphs import Neo4jGraph
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain_core.documents import Document
from langchain.chains import GraphCypherQAChain
from langchain_unstructured import UnstructuredLoader
from langchain_community.vectorstores import Neo4jVector
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.graphs.graph_document import GraphDocument, Node as GraphNode, Relationship as GraphRelationship
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_cohere import CohereEmbeddings
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from pydantic import BaseModel
from fastapi import UploadFile
from azure.core.credentials import AzureKeyCredential
from azure.ai.documentintelligence import DocumentIntelligenceClient
from google.api_core.exceptions import InternalServerError, ServiceUnavailable
load_dotenv()

# ... (ส่วนของ Prompts และ Class RetrieveAnswer ไม่มีการเปลี่ยนแปลง) ...
CUSTOM_GRAPH_EXTRACTION_SYSTEM_PROMPT = (
    "# Knowledge Graph Instructions\n"
    "## 1. Overview\n"
    "You are a top-tier algorithm for extracting information in structured formats to build a knowledge graph from ESG (Environmental, Social, and Governance) documents.\n"
    "Focus on accuracy and capturing as much relevant ESG information as possible. Do not add information not explicitly mentioned.\n"
    "- **Nodes** represent entities, concepts, metrics, standards, disclosures, etc.\n"
    "- **Relationships** represent connections between them.\n"
    "The goal is a clear, simple, and accurate knowledge graph.\n\n"

    "## 2. Extracting Nodes (Entities and Concepts)\n"
    "- **Node ID (`id`)**: Use a human-readable, unique identifier for the node, typically the name of the entity or concept found in the text (e.g., 'GRI 201', 'Climate Change', 'Operating Costs'). Prefer concise identifiers.\n"
    "- **Node Type (`type`)**: Use a consistent and basic type from the provided list of allowed node types. For example, if 'Metric' is an allowed type, use 'Metric' instead of 'FinancialMetric'.\n"
    "- **Node Name Property (`name`)**: If the primary identifier is different from a common name, extract the common name as the 'name' property. Often, `id` and `name` might be the same.\n"
    "- **Node Description Property (`description`) [VERY IMPORTANT]**: For EACH extracted node, you MUST generate a concise but informative 'description' property. This description should summarize the entity's meaning, context, and significance *based only on the provided text segment*. Aim for 1-2 clear sentences that would help differentiate this entity and be useful for semantic search. If the text provides a definition, key characteristics, or its role, include that in the description. For example, for a node ID 'EVG&D', a good description might be 'Direct economic value generated and distributed (EVG&D) by an organization, including revenues, operating costs, employee compensation, and payments to capital providers and governments.'\n\n"

    "## 3. Extracting Relationships\n"
    "- Use general yet descriptive relationship types from the provided list of allowed relationship types (e.g., 'REQUIRES_DISCLOSURE_OF', 'COMPONENT_OF', 'IMPACTS').\n"
    "- **Relationship Description Property (`description`)**: If the relationship itself has specific nuances or context mentioned in the text, capture that in the 'description' property of the relationship.\n\n"

    "## 4. Coreference Resolution\n"
    "Maintain entity consistency. If an entity is mentioned multiple times with different names (e.g., 'Global Reporting Initiative', 'GRI'), use the most complete or official identifier (e.g., 'Global Reporting Initiative') as the Node ID and try to capture aliases if possible (though the primary focus is the main ID and description).\n\n"

    "## 5. Strict Compliance and Output Format\n"
    "Adhere to the output format (Nodes and Relationships) and these rules strictly. Provide JSON output as requested by the system."
)

CUSTOM_GRAPH_EXTRACTION_HUMAN_MESSAGE = (
    "Tip: Make sure to answer in the correct JSON format with 'nodes' and 'relationships' keys. "
    "Do not include any explanations outside the JSON structure. "
    "Focus on extracting entities and their relationships from the following ESG text based on the instructions and schema provided.\n"
    "Input Text: {input}"
)

custom_esg_graph_extraction_prompt = ChatPromptTemplate.from_messages(
    [
        SystemMessagePromptTemplate.from_template(CUSTOM_GRAPH_EXTRACTION_SYSTEM_PROMPT),
        HumanMessagePromptTemplate.from_template(CUSTOM_GRAPH_EXTRACTION_HUMAN_MESSAGE),
    ]
)

class RetrieveAnswer(BaseModel):
    cypher_answer: str
    relate_documents: List[Document]


class Neo4jService:
    def __init__(self):
        self.graph_chain = None
        self.store = None
        self.graph = None
        self.environment = os.getenv("ENVIRONMENT", "development")
        self.logger = logging.getLogger(__name__) # <--- เพิ่ม
        self.llm = ChatGoogleGenerativeAI(
            temperature=0.2,
            model=os.getenv("NEO4J_MODEL", "gemini-2.5-flash-preview-05-20"),
            max_retries=10,
        )
        self.llm_embbedding = CohereEmbeddings(
            model='embed-v4.0',
            cohere_api_key=os.getenv("COHERE_API_KEY"),
            max_retries=10
        )
        self.azure_doc_intelligence_endpoint = os.getenv("AZURE_DOCUMENTINTELLIGENCE_ENDPOINT")
        self.azure_doc_intelligence_key = os.getenv("AZURE_DOCUMENTINTELLIGENCE_KEY")
        
        self.standard_chunk_vector_index_name = "standard_chunk_embedding_index"
        self.standard_chunk_node_label = "StandardChunk"
        self.standard_chunk_embedding_property = "embedding"
        self.standard_chunk_text_property = "text"

        self.entity_description_index_name = "entity_description_embedding_index"
        self.entity_node_label = "__Entity__"
        self.entity_description_embedding_property = "description_embedding"
        
        print("INFO: Using Cohere embedding model 'embed-v4.0' with dimension 1536.")

        # --- แก้ไขลำดับการทำงานให้ถูกต้อง ---
        self.init_graph() # 1. สร้าง self.graph ก่อน
        if self.graph:
            self.setup_neo4j_indexes() # 2. ค่อยสร้าง Index
            self.init_vector() # 3. ค่อยสร้าง Vector Store
            self.init_cypher() # 4. ค่อยสร้าง Chain
        else:
            print("CRITICAL: Neo4j graph could not be initialized. Aborting further setup.")

        # Informational messages for Neo4j connection stability
        print("INFO: Regarding Neo4j connection stability ('defunct connection' errors):")
        print("INFO: Please ensure your Neo4j server has appropriate timeout settings (e.g., bolt keep_alive, transaction timeouts).")
        print("INFO: If using Neo4j Aura, check your tier's connection/query limits.")

        if not self.azure_doc_intelligence_endpoint or not self.azure_doc_intelligence_key:
            print("WARNING: Azure Document Intelligence endpoint or key is not configured. PDF processing via Azure will fail.")
            self.document_intelligence_client = None
        else:
            try:
                self.document_intelligence_client = DocumentIntelligenceClient(
                    endpoint=self.azure_doc_intelligence_endpoint,
                    credential=AzureKeyCredential(self.azure_doc_intelligence_key)
                )
                print("Azure Document Intelligence client initialized.")
            except Exception as e:
                self.document_intelligence_client = None
                print(f"Error initializing Azure Document Intelligence client: {e}")
                traceback.print_exc()

    async def find_semantically_similar_chunks(self, query_text: str, top_k: int = 3) -> List[Dict[str, Any]]:
        if not self.store:
            self.logger.warning("Vector store is not initialized. Cannot perform semantic search.")
            return []
        
        try:
            self.logger.info(f"Performing ASYNC semantic search for: '{query_text}'")
            similar_nodes_with_scores = await self.store.asimilarity_search_with_score(query=query_text, k=top_k)
            
            # This part is crucial: we need to get the entity IDs linked to the chunk
            chunk_ids = [node.metadata.get("chunk_id") for node, score in similar_nodes_with_scores if node.metadata.get("chunk_id")]
            entity_map = await self._get_entities_for_chunks(chunk_ids)

            return [
                {
                    "text": node.page_content,
                    "chunk_id": node.metadata.get("chunk_id", ""),
                    "doc_id": node.metadata.get("doc_id", ""),
                    "score": score,
                    "entity_ids": entity_map.get(node.metadata.get("chunk_id"), []) # Add the entity IDs here
                }
                for node, score in similar_nodes_with_scores
            ]
        except Exception as e:
            self.logger.error(f"Error during async semantic similarity search: {e}", exc_info=True)
            return []

    async def _get_entities_for_chunks(self, chunk_ids: List[str]) -> Dict[str, List[str]]:
        """Given a list of chunk IDs, returns a map of chunk_id -> [entity_id_1, ...]."""
        if not chunk_ids or not self.graph: return {}
        
        query = """
        UNWIND $chunk_ids AS c_id
        MATCH (sc:StandardChunk {chunk_id: c_id})-[:CONTAINS_ENTITY]->(e:__Entity__)
        RETURN c_id, COLLECT(e.id) AS entity_ids
        """
        try:
            results = await asyncio.to_thread(self.graph.query, query, params={'chunk_ids': chunk_ids})
            return {record['c_id']: record['entity_ids'] for record in results}
        except Exception as e:
            self.logger.error(f"Error fetching entities for chunks: {e}")
            return {}

    async def get_chunks_by_standard_code(self, standard_code: str) -> List[Dict[str, Any]]:
        """
        Retrieves all StandardChunk nodes that have a specific `standard_code`.
        This is used for the targeted discovery phase (Phase 2).
        """
        if not self.graph:
            self.logger.error("Neo4j graph not initialized. Cannot get chunks by standard code.")
            return []

        self.logger.info(f"Fetching chunks for specific standard_code: {standard_code}")
        
        query = """
        MATCH (sc:StandardChunk {standard_code: $standard_code})
        // Also collect the IDs of entities contained within each chunk
        OPTIONAL MATCH (sc)-[:CONTAINS_ENTITY]->(e:__Entity__)
        RETURN 
            sc.chunk_id AS chunk_id,
            sc.text AS text,
            sc.doc_id AS doc_id,
            sc.page_number AS page_number,
            collect(e.id) AS entity_ids
        """
        try:
            params = {'standard_code': standard_code}
            results = await asyncio.to_thread(self.graph.query, query, params=params)
            
            if not results:
                self.logger.warning(f"No chunks found with standard_code: {standard_code}")
                return []

            # Format the results into a list of dictionaries
            formatted_results = [
                {
                    "chunk_id": record["chunk_id"],
                    "text": record["text"],
                    "doc_id": record["doc_id"],
                    "page_number": record["page_number"],
                    "entity_ids": record["entity_ids"] # This will be a list of entity IDs
                }
                for record in results
            ]
            self.logger.info(f"Found {len(formatted_results)} chunks for standard_code: {standard_code}")
            return formatted_results

        except Exception as e:
            self.logger.error(f"Error executing get_chunks_by_standard_code for '{standard_code}': {e}", exc_info=True)
            return []

    async def run_document_pipeline(self, file: UploadFile):
        """
        Orchestrates the end-to-end processing of an uploaded file into the Knowledge Graph.
        """
        self.logger.info(f"Starting document pipeline for file: {file.filename}")
        try:
            file_content = await file.read()
            # It's important to create a new IO stream for each use or reset it
            file_stream = IO[bytes](file_content)

            await self.flow(files=[file_stream], file_names=[file.filename])

            self.logger.info(f"Successfully completed document pipeline for file: {file.filename}")

        except Exception as e:
            self.logger.error(f"Error in document pipeline for {file.filename}: {e}", exc_info=True)
            raise
        finally:
            await file.close()

    ### NEW AND CORRECTED FUNCTION 2: Keyword-based chunk search ###
    def find_chunks_by_keywords(self, keywords: List[str], top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Searches for StandardChunk nodes using a secure parameterized query.
        """
        if not keywords:
            return []

        self.logger.info(f"Searching for chunks with keywords using parameterized query: {keywords}")

        params = {}
        where_clauses = []
        for i, keyword in enumerate(keywords):
            param_name = f"keyword_{i}"
            params[param_name] = keyword
            where_clauses.append(f"toLower(chunk.text) CONTAINS toLower(${param_name})")

        query = f"""
        MATCH (chunk:StandardChunk)
        WHERE {' OR '.join(where_clauses)}
        WITH chunk
        LIMIT {top_k}
        RETURN chunk.text AS text, chunk.chunk_id AS chunk_id, chunk.doc_id AS document_id
        """

        try:
            results = self.graph.query(query, parameters=params)
            
            if not results:
                self.logger.info("No chunks found for the given keywords.")
                return []

            return [
                {
                    "text": record["text"],
                    "chunk_id": record["chunk_id"],
                    "document_id": record["document_id"]
                }
                for record in results
            ]
        except Exception as e:
            self.logger.error(f"Error executing parameterized keyword search query in Neo4j: {e}", exc_info=True)
            return []
        
    def init_graph(self, url=None, username=None, password=None):
        # --- START: แก้ไขกลับไปเป็นแบบเดิมที่ถูกต้อง ---
        if not self.graph:
            try:
                self.graph = Neo4jGraph(
                    url=url or os.environ["NEO4J_URI"],
                    username=username or os.environ["NEO4J_USERNAME"],
                    password=password or os.environ["NEO4J_PASSWORD"],
                )
                print("Neo4jGraph initialized successfully.")
            except Exception as e:
                print(f"CRITICAL: Failed to create Neo4jGraph instance: {e}")
                traceback.print_exc()
                self.graph = None
        # --- END: แก้ไข ---

    def reinit_graph(self, url=None, username=None, password=None):
        # --- START: แก้ไข ---
        self.graph = None # Reset before re-initializing
        self.init_graph(url, username, password)
        # --- END: แก้ไข ---

    def setup_neo4j_indexes(self):
        cohere_embedding_dimension = 1536  # For embed-english-v3.0
        # Index for StandardDocument
        try:
            self.graph.query("CREATE CONSTRAINT standard_document_doc_id IF NOT EXISTS FOR (d:StandardDocument) REQUIRE d.doc_id IS UNIQUE")
            print("Constraint/Index for StandardDocument.doc_id ensured.")
        except Exception as e:
            print(f"Error creating constraint/index for StandardDocument: {e}")

        # Index for StandardChunk (chunk_id)
        try:
            self.graph.query("CREATE CONSTRAINT standard_chunk_chunk_id IF NOT EXISTS FOR (sc:StandardChunk) REQUIRE sc.chunk_id IS UNIQUE")
            print("Constraint/Index for StandardChunk.chunk_id ensured.")
        except Exception as e:
            print(f"Error creating constraint/index for StandardChunk: {e}")
        
        # For StandardChunk index
        try:
            self.graph.query(
                f"""CREATE VECTOR INDEX {self.standard_chunk_vector_index_name} IF NOT EXISTS 
                   FOR (sc:{self.standard_chunk_node_label}) ON (sc.{self.standard_chunk_embedding_property}) 
                   OPTIONS {{indexConfig: {{
                    `vector.dimensions`: {cohere_embedding_dimension},
                    `vector.similarity_function`: 'cosine'
                   }}}}"""
            )
            print(f"Vector Index '{self.standard_chunk_vector_index_name}' ensured.")
        except Exception as e:
            print(f"CRITICAL ERROR creating vector index '{self.standard_chunk_vector_index_name}': {e}")
            traceback.print_exc()

        # For __Entity__ description embedding index
        try:
            self.graph.query(
                f"""CREATE VECTOR INDEX {self.entity_description_index_name} IF NOT EXISTS
                   FOR (e:{self.entity_node_label}) ON (e.{self.entity_description_embedding_property})
                   OPTIONS {{indexConfig: {{
                    `vector.dimensions`: {cohere_embedding_dimension},
                    `vector.similarity_function`: 'cosine'
                   }}}}"""
            )
            print(f"Vector Index '{self.entity_description_index_name}' ensured.")
        except Exception as e:
            print(f"CRITICAL ERROR creating vector index '{self.entity_description_index_name}': {e}")
            traceback.print_exc()

    def init_cypher(self):
        if not self.graph:
            print("WARNING: Graph not initialized, skipping Cypher chain setup.")
            return
        self.graph_chain = GraphCypherQAChain.from_llm(
            self.llm, 
            graph=self.graph,  
            allow_dangerous_requests=True,
            verbose=False,
            validate_cypher=True
        )
        
    def init_vector(self, url=None, username=None, password=None):
        db_url = url or os.environ["NEO4J_URI"]
        db_username = username or os.environ["NEO4J_USERNAME"]
        db_password = password or os.environ["NEO4J_PASSWORD"]
        
        try:
            self.store = Neo4jVector.from_existing_index(
                embedding=self.llm_embbedding,
                url=db_url,
                username=db_username,
                password=db_password,
                index_name=self.standard_chunk_vector_index_name,
                node_label=self.standard_chunk_node_label,
                text_node_property=self.standard_chunk_text_property,
                embedding_node_property=self.standard_chunk_embedding_property,
            )
            print(f"Vector store '{self.standard_chunk_vector_index_name}' initialized from existing index.")
        except ValueError as ve: 
            print(f"CRITICAL ValueError during Neo4jVector.from_existing_index for index '{self.standard_chunk_vector_index_name}': {ve}")
            print("This means the vector index WAS NOT FOUND. Ensure `setup_neo4j_indexes()` ran successfully.")
            # --- START: แก้ไข Parameter ที่ผิด ---
            try:
                self.store = Neo4jVector(
                    embedding=self.llm_embbedding,
                    url=db_url,
                    username=db_username,
                    password=db_password,
                    index_name=self.standard_chunk_vector_index_name,
                    node_label=self.standard_chunk_node_label,
                    text_node_property=self.standard_chunk_text_property, # << แก้จาก text_node_properties
                    embedding_node_property=self.standard_chunk_embedding_property
                )
                print(f"Neo4jVector instance created for '{self.standard_chunk_node_label}'. Index should exist.")
            except Exception as e_fallback:
                 print(f"CRITICAL FALLBACK ERROR: Could not create even a basic Neo4jVector instance: {e_fallback}")
                 traceback.print_exc()
                 self.store = None
            # --- END: แก้ไข ---
        except Exception as e:
            print(f"An unexpected error occurred during init_vector: {e}")
            traceback.print_exc()
            self.store = None


    def reinit_vector(self, url=None, username=None, password=None):
        self.init_vector(url, username, password)

    # ... The rest of your file (read_PDFs, flow, etc.) remains unchanged.
    # Make sure to keep all other methods from your original file.
    
    async def read_PDFs_and_create_documents(self, files: List[IO[bytes]], file_names: List[str], 
                                     chunking_strategy: str = "basic", 
                                     ):
        docs = []
        temp_dir = "temp_files_for_unstructured"
        os.makedirs(temp_dir, exist_ok=True)

        for i, file_content_io in enumerate(files):
            file_name = file_names[i]
            temp_file_path = os.path.join(temp_dir, file_name)
            
            with open(temp_file_path, "wb") as temp_file:
                temp_file.write(file_content_io.read())
            
            loader = UnstructuredLoader(
                file_path=temp_file_path,
                mode="elements", 
                strategy=chunking_strategy if chunking_strategy in ["fast", "hi_res", "ocr_only"] else "fast",
            )
            try:
                loaded_docs_for_file = loader.load()
                for ld_doc in loaded_docs_for_file:
                    ld_doc.metadata["source_file_name"] = file_name 
                docs.extend(loaded_docs_for_file)
            except Exception as e_load:
                print(f"Error loading file {file_name} with UnstructuredFileLoader: {e_load}")
                traceback.print_exc()
            finally:
                if os.path.exists(temp_file_path):
                    os.remove(temp_file_path)
        
        if not os.listdir(temp_dir):
            os.rmdir(temp_dir)
            
        return docs

    async def read_PDFs_and_create_documents_azure(
        self, files: List[IO[bytes]], file_names: List[str]
    ) -> List[Document]:
        if not self.document_intelligence_client:
            print("ERROR: Azure Document Intelligence client is not initialized. Cannot process PDFs.")
            return []

        all_lc_documents: List[Document] = []
        loop = asyncio.get_running_loop()

        for i, file_content_io in enumerate(files):
            file_name = file_names[i]
            print(f"Processing PDF with Azure Document Intelligence: {file_name}")

            try:
                file_content_io.seek(0)
                file_bytes = file_content_io.read()

                poller = self.document_intelligence_client.begin_analyze_document(
                    model_id="prebuilt-layout",
                    analyze_request=file_bytes,
                    content_type="application/octet-stream"
                )
                
                blocking_call = functools.partial(poller.result)
                result = await loop.run_in_executor(None, blocking_call)

                if result and hasattr(result, 'pages') and result.pages:
                    for page_idx, page in enumerate(result.pages):
                        page_text_parts = []
                        if page.lines:
                            for line in page.lines:
                                page_text_parts.append(line.content)

                        full_page_text = "\n".join(page_text_parts).strip()

                        if not full_page_text:
                            print(f"  Skipping page {page.page_number} of {file_name} as it contains no extracted text.")
                            continue

                        lc_doc_metadata = {
                            "source_file_name": file_name,
                            "page_number": page.page_number,
                        }

                        page_roles = set()
                        if hasattr(result, 'paragraphs') and result.paragraphs:
                            for para in result.paragraphs:
                                if para.bounding_regions:
                                    for region in para.bounding_regions:
                                        if region.page_number == page.page_number and para.role:
                                            page_roles.add(para.role)

                        if "title" in page_roles:
                            lc_doc_metadata["category"] = "title"
                        elif "sectionHeading" in page_roles:
                            lc_doc_metadata["category"] = "sectionHeading"

                        all_lc_documents.append(
                            Document(page_content=full_page_text, metadata=lc_doc_metadata)
                        )
                    print(f"Successfully processed {len(result.pages)} pages from {file_name} using Azure Document Intelligence.")
                elif result:
                    print(f"Azure Document Intelligence processed {file_name}, but no pages were found in the result.")
                else:
                    print(f"Azure Document Intelligence returned a None result for {file_name}.")

            except Exception as e_azure_ocr:
                print(f"Error processing PDF {file_name} with Azure Document Intelligence: {e_azure_ocr}")
                traceback.print_exc()

        return all_lc_documents

    async def split_documents_into_chunks(self, documents: List[Document], chunk_size: int = 2048, chunk_overlap: int = 1024):
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            is_separator_regex=False,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        return await asyncio.to_thread(splitter.split_documents, documents)

    async def translate_documents_with_gemini(self, documents: List[Document], batch_size: int = 10):
        def contains_thai(text: str) -> bool:
            thai_pattern = re.compile("[\u0E00-\u0E7F]")
            return bool(thai_pattern.search(text))

        llm = self.llm
        prompt = PromptTemplate(
            input_variables=["text"],
            template="You are a professional English translator. Translate the following Thai text accurately and fluently into English. Return ONLY the translated English text, without any additional explanations, conversational phrases, or original Thai text.\nThai Text:\n{text}\n\nEnglish Translation:"
        )
        chain = prompt | llm

        async def run_chain(text_content):
            if not text_content or not contains_thai(text_content):
                return text_content
            try:
                response = await chain.ainvoke({"text": text_content})
                return response.content
            except Exception as e_translate:
                print(f"Error translating text: {e_translate}. Returning original text.")
                return text_content

        tasks = []
        for doc_idx in range(0, len(documents), batch_size):
            batch_docs = documents[doc_idx:doc_idx + batch_size]
            batch_input_texts = [doc.page_content for doc in batch_docs]
            
            current_batch_tasks = [run_chain(text) for text in batch_input_texts]
            tasks.extend(current_batch_tasks)

        all_translated_texts = await asyncio.gather(*tasks, return_exceptions=True)
        
        processed_idx = 0
        for i in range(len(documents)):
            if isinstance(all_translated_texts[processed_idx], Exception):
                print(f"Translation task for document index {i} failed: {all_translated_texts[processed_idx]}")
            elif all_translated_texts[processed_idx]:
                documents[i].page_content = all_translated_texts[processed_idx]
            processed_idx += 1
            
        return documents


    def __get_auth_env(self):
        return os.environ["NEO4J_URI"], os.environ["NEO4J_USERNAME"], os.environ["NEO4J_PASSWORD"]

    def add_description_embeddings_to_nodes(self):
        print(f"Attempting to populate description embeddings for nodes with label '{self.entity_node_label}'...")
        url, username, password = self.__get_auth_env()
        try:
            Neo4jVector.from_existing_graph(
                embedding=self.llm_embbedding,
                url=url,
                username=username,
                password=password,
                node_label=self.entity_node_label,
                text_node_properties=['id', 'description'],
                embedding_node_property=self.entity_description_embedding_property,
                index_name=self.entity_description_index_name,
            )
            print(f"Population of description embeddings for '{self.entity_node_label}' nodes initiated.")
        except Exception as e_desc_emb:
            print(f"CRITICAL ERROR during population of description embeddings: {e_desc_emb}")
            traceback.print_exc()


    async def _create_standard_document_and_chunks(self, 
                                                 file_name: str, 
                                                 translated_chunks: List[Document], 
                                                 standard_title: Optional[str] = None,
                                                 # --- FIX 2: เพิ่ม parameter `standard_code` ---
                                                 standard_code: Optional[str] = None):
        doc_id = file_name
        doc_title = standard_title if standard_title else file_name
        
        # (Merge StandardDocument logic remains the same)
        merge_doc_query = "MERGE (d:StandardDocument {doc_id: $doc_id}) SET d.title = $title, d.source_file_name = $source_file_name"
        await asyncio.to_thread(self.graph.query, merge_doc_query, params={'doc_id': doc_id, 'title': doc_title, 'source_file_name': file_name})

        chunk_nodes_data_for_cypher = []
        documents_for_transformer = []

        for i, chunk_doc in enumerate(translated_chunks):
            chunk_text = chunk_doc.page_content
            try:
                embedding_vector = await asyncio.to_thread(self.llm_embbedding.embed_query, chunk_text)
            except (InternalServerError, ServiceUnavailable) as e:
                self.logger.warning(f"Embedding failed for a chunk in doc '{doc_id}'. Skipping. Error: {e}")
                continue

            chunk_id = f"{doc_id}_chunk_{i:04d}"
            
            # --- FIX 3: เพิ่ม standard_code เข้าไปใน dictionary ---
            chunk_data = {
                'chunk_id': chunk_id,
                'text': chunk_text,
                'embedding': embedding_vector,
                'doc_id': doc_id,
                'standard_code': standard_code, # <--- เพิ่มบรรทัดนี้
                'page_number': chunk_doc.metadata.get('page_number')
            }
            # ----------------------------------------------------
            chunk_nodes_data_for_cypher.append(chunk_data)
            documents_for_transformer.append(Document(page_content=chunk_text, metadata={'chunk_id': chunk_id, 'doc_id': doc_id}))

        if chunk_nodes_data_for_cypher:
            # --- FIX 3: เพิ่มการ SET `standard_code` ใน Cypher Query ---
            create_chunks_query = f"""
            UNWIND $chunks_data AS props
            MATCH (d:StandardDocument {{doc_id: props.doc_id}})
            MERGE (sc:{self.standard_chunk_node_label} {{chunk_id: props.chunk_id}})
            SET 
                sc.text = props.text,
                sc.embedding = props.embedding,
                sc.doc_id = props.doc_id,
                sc.standard_code = props.standard_code,
                sc.page_number = props.page_number,
                sc.last_updated = timestamp()
            MERGE (d)-[:HAS_CHUNK]->(sc)
            """
            # ---------------------------------------------------------
            await asyncio.to_thread(self.graph.query, create_chunks_query, params={'chunks_data': chunk_nodes_data_for_cypher})
            self.logger.info(f"Processed {len(chunk_nodes_data_for_cypher)} chunks for document {doc_id}.")

        return documents_for_transformer

    async def _apply_llm_graph_transformer_to_chunks(self, 
                                                    chunk_documents: List[Document],
                                                    llm_transformer: LLMGraphTransformer):
        print(f"Applying LLMGraphTransformer to {len(chunk_documents)} chunk documents...")
        if not chunk_documents:
            return

        graph_documents_from_llm: List[GraphDocument] = await llm_transformer.aconvert_to_graph_documents(chunk_documents)
        
        if not graph_documents_from_llm:
            print("LLMGraphTransformer did not return any graph documents.")
            return
        
        print(f"LLMGraphTransformer returned {len(graph_documents_from_llm)} graph documents.")
        
        try:
            await asyncio.to_thread(
                self.graph.add_graph_documents,
                graph_documents_from_llm,
                baseEntityLabel=True, 
                include_source=True 
            )
            print(f"Added nodes and relationships from LLMGraphTransformer.")
        except Exception as e_add_graph:
            print(f"Error adding graph documents from LLMTransformer: {e_add_graph}")
            traceback.print_exc()
            return

        link_params = []
        for gd in graph_documents_from_llm:
            source_chunk_id = gd.source.metadata.get('chunk_id')
            if not source_chunk_id:
                print(f"Warning: GraphDocument source missing chunk_id: {gd.source.page_content[:50]}...")
                continue
            
            for node_in_gd in gd.nodes:
                # --- START: เพิ่มการตรวจสอบ ID ตรงนี้ ---
                if not node_in_gd.id or not node_in_gd.id.strip():
                    print(f"WARNING: Skipping a node because it has a missing or empty ID. Node Type: {node_in_gd.type}, Source Chunk: {source_chunk_id}")
                    continue # ข้ามไปยัง Node ถัดไป
                # --- END: เพิ่มการตรวจสอบ ---

                entity_id = node_in_gd.id
                
                link_params.append({
                    'chunk_id': source_chunk_id,
                    'entity_id_prop_val': entity_id,
                })
        
        if link_params:
            link_entities_query = """
            UNWIND $link_data AS item
            MATCH (sc:StandardChunk {chunk_id: item.chunk_id})
            MATCH (e:__Entity__ {id: item.entity_id_prop_val}) 
            MERGE (sc)-[:CONTAINS_ENTITY]->(e)
            RETURN count(DISTINCT sc) AS linked_chunks, count(DISTINCT e) as linked_entities
            """
            try:
                result = await asyncio.to_thread(self.graph.query, link_entities_query, params={'link_data': link_params})
                if result and result[0]:
                    print(f"Linked StandardChunks to __Entity__ nodes: {result[0]}")
            except Exception as e_link:
                print(f"Error linking LLMTransformer __Entity__ nodes to StandardChunks: {e_link}")
                traceback.print_exc()

    async def flow(self, files: List[IO[bytes]], file_names: List[str]) -> List[str]:
        """
        Orchestrates the end-to-end data ingestion and knowledge graph creation pipeline.
        This includes reading, chunking, translating, and extracting graph data from documents.
        """
        loop = asyncio.get_running_loop()
        
        # 1. Read PDFs and create initial Document objects
        if self.document_intelligence_client:
            self.logger.info("Using Azure Document Intelligence for PDF processing.")
            documents_from_pdf_loader = await self.read_PDFs_and_create_documents_azure(files, file_names)
        else:
            self.logger.info("Azure client not available, falling back to Unstructured.")
            documents_from_pdf_loader = await loop.run_in_executor(None, self.read_PDFs_and_create_documents, files, file_names)
        
        if not documents_from_pdf_loader:
            self.logger.error("No documents were loaded from PDFs. Aborting flow.")
            return []

        # 2. Split loaded documents into smaller chunks
        split_chunks_lc_docs = await self.split_documents_into_chunks(documents_from_pdf_loader)
        if not split_chunks_lc_docs:
            self.logger.error("No chunks were created after splitting. Aborting flow.")
            return []

        # 3. Translate chunks from Thai to English
        self.logger.info(f"Starting translation for {len(split_chunks_lc_docs)} chunks...")
        translated_chunks_lc_docs = await self.translate_documents_with_gemini(split_chunks_lc_docs)
        self.logger.info("Translation complete.")
        
        # 4. Group translated chunks by their original source file
        docs_by_source_file: Dict[str, List[Document]] = {}
        for chunk_lc_doc in translated_chunks_lc_docs:
            source_fn = chunk_lc_doc.metadata.get('source_file_name', "unknown_source_file.pdf")
            docs_by_source_file.setdefault(source_fn, []).append(chunk_lc_doc)

        # --- CORRECTED POSITION ---
        # Get the list of processed document IDs AFTER populating the dictionary
        processed_doc_ids = list(docs_by_source_file.keys())
        # ------------------------

        # 5. Initialize the Graph Transformer
        esg_allowed_nodes = [ "Organization", "Person", "Standard", "Framework", "Guideline", "Regulation", "Law", "Report", "Document", "Disclosure", "Topic", "SubTopic", "Category", "Metric", "Indicator", "KPI", "Target", "Goal", "Objective", "Risk", "Opportunity", "Impact", "Strategy", "Policy", "Process", "Practice", "Activity", "Initiative", "Program", "Project", "Stakeholder", "Investor", "Employee", "Customer", "Supplier", "Community", "Government", "Region", "Country", "Location", "Facility", "EnvironmentalFactor", "SocialFactor", "GovernanceFactor", "ClimateChange", "GreenhouseGas", "GHGEmissions", "Energy", "Water", "Waste", "Biodiversity", "HumanRights", "LaborPractices", "HealthAndSafety", "DiversityAndInclusion", "CommunityEngagement", "BoardComposition", "Ethics", "Corruption", "ShareholderRights", "EconomicValue", "FinancialPerformance", "OperatingCost", "Revenue", "Investment", "Date", "Year", "Period", "Unit", "Value", "Percentage", "Currency", "Source", "Reference", "Term", "Concept" ]
        esg_allowed_relationships = [ "HAS_FRAMEWORK", "COMPLIES_WITH", "REFERENCES_STANDARD", "DISCLOSES_INFORMATION_ON", "REPORTS_ON", "APPLIES_TO", "MENTIONS", "DEFINES", "HAS_METRIC", "MEASURES", "TRACKS_KPI", "SETS_TARGET", "ACHIEVES_GOAL", "IDENTIFIES_RISK", "PRESENTS_OPPORTUNITY", "HAS_IMPACT_ON", "MITIGATES_RISK", "HAS_STRATEGY", "IMPLEMENTS_POLICY", "FOLLOWS_PROCESS", "ENGAGES_IN_ACTIVITY", "INVESTS_IN", "OPERATES_IN", "LOCATED_IN", "GOVERNED_BY", "PART_OF", "COMPONENT_OF", "SUBCLASS_OF", "INSTANCE_OF", "RELATES_TO", "ASSOCIATED_WITH", "CAUSES", "INFLUENCES", "CONTRIBUTES_TO", "REQUIRES", "ADDRESSES", "MANAGES", "MONITORS", "EVALUATES", "ISSUED_BY", "DEVELOPED_BY", "PUBLISHED_IN", "EFFECTIVE_DATE" ]
        
        llm_transformer_for_kg = LLMGraphTransformer(
            llm=self.llm,
            node_properties=["description", "name"],
            relationship_properties=["description"],
            strict_mode=True,
            allowed_nodes=esg_allowed_nodes,
            allowed_relationships=esg_allowed_relationships,
            prompt=custom_esg_graph_extraction_prompt
        )
        
        # 6. Process each file's chunks
        for file_name in processed_doc_ids: # Use the generated list of doc IDs
            chunks_for_this_file = docs_by_source_file.get(file_name)
            if not chunks_for_this_file:
                self.logger.warning(f"No chunks found for file {file_name} after grouping. Skipping.")
                continue

            self.logger.info(f"Starting graph processing for file: {file_name}")
            
            standard_title, standard_code = self._extract_info_from_filename(file_name)

            chunk_docs_with_ids_for_transformer = await self._create_standard_document_and_chunks(
                file_name=file_name,
                translated_chunks=chunks_for_this_file,
                standard_title=standard_title,
                standard_code=standard_code
            )

            if chunk_docs_with_ids_for_transformer:
                await self._apply_llm_graph_transformer_to_chunks(
                    chunk_docs_with_ids_for_transformer,
                    llm_transformer_for_kg
                )
            
            self.logger.info(f"Completed KG extraction for file: {file_name}\n--------------------")

        # 7. Final step: Populate embeddings for new entity descriptions
        self.logger.info("All files processed. Adding description embeddings for new __Entity__ nodes...")
        await loop.run_in_executor(None, self.add_description_embeddings_to_nodes)
        self.logger.info("PDF processing flow completed successfully.")
        
        return processed_doc_ids

    async def get_relate(self, query, k):     
        if not self.store:
            print("ERROR: Vector store not initialized in get_relate.")
            return []
        try:
            docs_with_score = await self.store.asimilarity_search_with_score(query, k=k)
            
            # --- ส่วนที่เพิ่มเข้ามาเพื่อลบ Embedding ---
            cleaned_docs = []
            for doc, score in docs_with_score:
                # ตรวจสอบและลบ key 'embedding' ออกจาก metadata
                if 'embedding' in doc.metadata:
                    del doc.metadata['embedding']
                if 'description_embedding' in doc.metadata:
                    del doc.metadata['description_embedding']
                cleaned_docs.append(doc)
            # -----------------------------------------
                
            return cleaned_docs
            
        except Exception as e_relate:
            print(f"Error in get_relate: {e_relate}")
            traceback.print_exc()
            return []

    async def get_cypher_answer(self, query):
        url, username, password = self.__get_auth_env()
        
        # --- แก้ไข Cypher Query ตรงนี้ ---
        retrieval_query_for_entities = """
            OPTIONAL MATCH (node)-[rel*1..2]-(neighbor)
            WITH node, score, collect({
                relationship: [r IN rel | type(r)], 
                relationship_properties: [r IN rel | properties(r)], 
                neighbor: neighbor { 
                    id: neighbor.id, 
                    name: neighbor.name, 
                    description: neighbor.description,
                    labels: labels(neighbor)
                }
            })[0..50] AS neighbors_and_relations
            RETURN coalesce(node.description, "No description available") AS text,
                score,
                {
                    id: node.id,
                    name: node.name,
                    description: node.description,
                    labels: labels(node),
                    neighbors_and_relations: neighbors_and_relations
                } AS metadata
        """
        # ----------------------------------

        try:
            entity_vector_store = Neo4jVector.from_existing_index(
                embedding=self.llm_embbedding,
                url=url,
                username=username,
                password=password,
                index_name=self.entity_description_index_name,
                text_node_property="description",
                embedding_node_property=self.entity_description_embedding_property,
                retrieval_query=retrieval_query_for_entities
            )
            docs = await entity_vector_store.asimilarity_search(query=query, k=3)
            
            # จัดรูปแบบผลลัพธ์ให้สวยงามขึ้น
            context_parts = []
            for doc in docs:
                # ลบ embedding ที่อาจจะยังหลงเหลืออยู่ออกจาก metadata อีกชั้นหนึ่ง
                if 'description_embedding' in doc.metadata:
                    del doc.metadata['description_embedding']
                context_parts.append(f"Entity Info: {doc.page_content}\nMETADATA: {str(doc.metadata)}")

            return "\n---\n".join(context_parts)

        except ValueError as ve:
            print(f"CRITICAL ValueError in get_cypher_answer: {ve}")
            return "Could not retrieve answer from entity graph: entity description index NOT FOUND."
        except Exception as e_cypher_ans:
            print(f"Error in get_cypher_answer: {e_cypher_ans}")
            traceback.print_exc()
            return "Error retrieving answer from entity graph."

            
    async def get_output(self, query, k=5) -> RetrieveAnswer:
        related_standard_chunks = await self.get_relate(query, k) 
        cypher_answer_from_entities = await self.get_cypher_answer(query)
        
        return RetrieveAnswer(
            cypher_answer=cypher_answer_from_entities,
            relate_documents=related_standard_chunks
        )

    async def is_graph_substantially_empty(self, threshold: int = 10) -> bool:
        if not self.graph:
            print("[NEO4J_SERVICE ERROR] Neo4jGraph not initialized.")
            return True 
        query = "MATCH (n) RETURN count(n) AS node_count"
        try:
            loop = asyncio.get_running_loop()
            result_data = await loop.run_in_executor(None, self.graph.query, query)
            if result_data and "node_count" in result_data[0]:
                node_count = result_data[0]["node_count"]
                print(f"[NEO4J_SERVICE LOG] Current node count: {node_count}")
                return node_count < threshold
            return False 
        except Exception as e:
            print(f"[NEO4J_SERVICE ERROR] Error checking if graph is empty: {e}.")
            return False
        
    async def get_all_standard_chunks_for_theme_generation(self) -> List[dict[str, any]]:
            query = """
            MATCH (sc:StandardChunk)
            WHERE sc.text IS NOT NULL AND trim(sc.text) <> "" AND sc.embedding IS NOT NULL
            RETURN 
                sc.chunk_id AS chunk_id,
                sc.text AS text,
                sc.embedding AS embedding,
                sc.doc_id AS source_document_doc_id,
                sc.original_section_id AS original_section_id,
                sc.page_number AS page_number
            """
            loop = asyncio.get_running_loop()
            try:
                results = await loop.run_in_executor(None, self.graph.query, query)
                if results:
                    print(f"[NEO4J_SERVICE LOG] Retrieved {len(results)} StandardChunks for theme generation.")
                    return results
                return []
            except Exception as e:
                print(f"[NEO4J_SERVICE ERROR] Error retrieving StandardChunks: {e}")
                traceback.print_exc()
                return []
            
    async def get_graph_context_for_theme_chunks_v2(
        self,
        theme_name: str,
        theme_keywords: str,
        max_central_entities: int = 2,
        max_hops_for_central_entity: int = 1,
        max_relations_to_collect_per_central_entity: int = 5,
        max_total_context_items_str_len: int = 1000000
    ) -> Optional[str]:
        if not theme_name and not theme_keywords:
            return "Theme name or keywords required."

        central_entity_ids: List[str] = []
        try:
            url, username, password = self.__get_auth_env()
            temp_entity_retrieval_query = """
            RETURN COALESCE(node.description, node.name, node.id, '') AS text,
                score, 
                {
                    id: node.id,
                    name:COALESCE(node.name, node.id), 
                    description: node.description,
                    labels: labels(node)
                } AS metadata
            """

            entity_store_for_seed_search = Neo4jVector.from_existing_index(
                embedding=self.llm_embbedding,
                url=url, username=username, password=password,
                index_name=self.entity_description_index_name,
                text_node_property="description",
                embedding_node_property=self.entity_description_embedding_property,
                retrieval_query=temp_entity_retrieval_query
            )

            search_query_for_central_entities = f"{theme_name} {theme_keywords}"
            seed_entity_docs = await entity_store_for_seed_search.asimilarity_search(
                query=search_query_for_central_entities, k=max_central_entities
            )

            if seed_entity_docs:
                for doc in seed_entity_docs:
                    if doc.metadata and doc.metadata.get('id'):
                        central_entity_ids.append(doc.metadata.get('id'))
            
            if not central_entity_ids:
                return "No central entities identified for this theme."

        except Exception as e_seed_search:
            print(f"[NEO4J ERROR] Error finding central entities for theme '{theme_name}': {e_seed_search}")
            traceback.print_exc()
            return "Error finding central entities for graph context."

        final_graph_context_parts = []
        loop = asyncio.get_running_loop()
        
        entity_neighborhood_query_template = """
        MATCH (node:__Entity__ {id: $center_entity_id}) 
        OPTIONAL MATCH (node)-[rel*1..{max_hops}]-(neighbor) 
        WHERE neighbor IS NOT NULL AND (NOT 'StandardChunk' IN labels(neighbor) AND NOT 'StandardDocument' IN labels(neighbor))
        WITH node, 
             COLLECT(DISTINCT {
                 relationship_path: [r_hop IN rel | type(r_hop)], 
                 neighbor_details: neighbor { 
                     id: COALESCE(neighbor.id, elementId(neighbor)), 
                     description: neighbor.description, 
                     labels: [l IN labels(neighbor) WHERE l <> '__Entity__']
                 }
             })[0..{max_rels}] AS neighbors_and_relations_list
        RETURN node.id as id, 
               COALESCE(node.description, "No description available") as description, 
               [l IN labels(node) WHERE l <> '__Entity__'] as labels,
               neighbors_and_relations_list AS neighbors_and_relations
        """
        entity_neighborhood_query = entity_neighborhood_query_template.replace(
            "{max_hops}", str(max_hops_for_central_entity)
        ).replace(
            "{max_rels}", str(max_relations_to_collect_per_central_entity)
        )

        total_context_len = 0
        for center_id in central_entity_ids:
            if total_context_len >= max_total_context_items_str_len : break
            try:
                query_callable_entity_context = functools.partial(
                    self.graph.query,
                    entity_neighborhood_query,
                    params={'center_entity_id': center_id}
                )
                entity_context_results = await loop.run_in_executor(None, query_callable_entity_context)

                if entity_context_results:
                    record = entity_context_results[0] 
                    
                    e_id = record.get('id', 'UnknownEntity')
                    e_desc = record.get('description', 'N/A')
                    e_labels_list = record.get('labels', [])
                    display_e_labels = e_labels_list if e_labels_list else ["Entity"]
                    
                    part_str = f"CENTRAL ENTITY: {e_id} (Type: {', '.join(display_e_labels)}; Desc: {e_desc})"
                    
                    relations_list = record.get('neighbors_and_relations', [])
                    if relations_list:
                        part_str += "\n  RELATIONS:"
                        for rel_detail in relations_list:
                            neighbor_details = rel_detail.get('neighbor_details', {})
                            n_id = neighbor_details.get('id', 'UnknownNeighbor')
                            n_desc = neighbor_details.get('description', 'N/A')
                            n_labels_list = neighbor_details.get('labels', [])
                            display_n_labels = n_labels_list if n_labels_list else ["Node"]
                            
                            rel_path_types_str = ", ".join(rel_detail.get('relationship_path', ['RELATED_TO']))
                            part_str += f"\n    --[{rel_path_types_str}]--> NEIGHBOR: {n_id} (Type: {', '.join(display_n_labels)}; Desc: {n_desc})"
                    
                    if total_context_len + len(part_str) < max_total_context_items_str_len:
                        final_graph_context_parts.append(part_str)
                        total_context_len += len(part_str)
                    else:
                        final_graph_context_parts.append("... [Context truncated]")
                        break
            except Exception as e_entity_ctx:
                print(f"[NEO4J ERROR] Error fetching context for central entity '{center_id}': {e_entity_ctx}")
        
        return "\n\n".join(final_graph_context_parts)
    
    def _extract_info_from_filename(self, file_name: str) -> Tuple[str, Optional[str]]:
        title = file_name.replace(".pdf", "").replace("_", " ")
        match = re.search(r'GRI[\s_-]?(\d{3})', file_name, re.IGNORECASE)
        if match:
            standard_code = f"GRI-{match.group(1)}"
            return title, standard_code
        return title, None