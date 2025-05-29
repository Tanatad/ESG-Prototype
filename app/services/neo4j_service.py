import re
import os
import asyncio
import inspect
import traceback
import functools
import neo4j
from typing import IO, List, Optional
from dotenv import load_dotenv
from langchain_community.graphs import Neo4jGraph
from langchain_google_genai import ChatGoogleGenerativeAI #, GoogleGenerativeAIEmbeddings # GoogleGenerativeAIEmbeddings not used directly
from langchain.prompts import PromptTemplate
from langchain_core.documents import Document
from langchain.chains import GraphCypherQAChain # Not used in the provided flow yet
from langchain_unstructured import UnstructuredLoader
from langchain_community.vectorstores import Neo4jVector
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.graphs.graph_document import GraphDocument, Node as GraphNode, Relationship as GraphRelationship # Import GraphNode and GraphRelationship
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_cohere import CohereEmbeddings
from app.services.rate_limit import llm_rate_limiter
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate # Import specific prompt types
from pydantic import BaseModel
from azure.core.credentials import AzureKeyCredential
from azure.ai.documentintelligence import DocumentIntelligenceClient
load_dotenv()

# placeholder_data is not ideal for initializing a specific schema.
# If an empty index is needed, it's better to let from_existing_index fail
# and then ensure the first data load populates it.
# placeholder_data = [
# Document(page_content="Initial document to create vector index if it doesn't exist.")
# ]
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
        self.llm = ChatGoogleGenerativeAI(
            temperature=0.8,
            model=os.getenv("NEO4J_MODEL", "gemini-2.5-flash-preview-05-20"), # Adjusted to a common Gemini model
            max_retries=10,
            rate_limiter=llm_rate_limiter # Assuming llm_rate_limiter is defined elsewhere
        )
        self.llm_embbedding = CohereEmbeddings( # This is your embedding model
            model='embed-v4.0', # Or your preferred cohere model like 'embed-english-v3.0'
            cohere_api_key=os.getenv("COHERE_API_KEY"),
            max_retries=10
        )
        self.azure_doc_intelligence_endpoint = os.getenv("AZURE_DOCUMENTINTELLIGENCE_ENDPOINT")
        self.azure_doc_intelligence_key = os.getenv("AZURE_DOCUMENTINTELLIGENCE_KEY")
        
        # Define these before init_vector uses them
        self.standard_chunk_vector_index_name = "standard_chunk_embedding_index"
        self.standard_chunk_node_label = "StandardChunk"
        self.standard_chunk_embedding_property = "embedding"
        self.standard_chunk_text_property = "text" # This is the property Neo4jVector will use for Document.page_content


        self.init_graph()
        self.setup_neo4j_indexes()
        self.init_vector()
        self.init_cypher()

        self.azure_doc_intelligence_endpoint = os.getenv("AZURE_DOCUMENTINTELLIGENCE_ENDPOINT")
        self.azure_doc_intelligence_key = os.getenv("AZURE_DOCUMENTINTELLIGENCE_KEY")

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

    def init_graph(self, url=None, username=None, password=None):
        if not self.graph:
            self.graph = Neo4jGraph(
                url=url or os.environ["NEO4J_URI"],
                username=username or os.environ["NEO4J_USERNAME"],
                password=password or os.environ["NEO4J_PASSWORD"],
            )

    def reinit_graph(self, url=None, username=None, password=None):
        self.graph = Neo4jGraph(
            url=url or os.environ["NEO4J_URI"],
            username=username or os.environ["NEO4J_USERNAME"],
            password=password or os.environ["NEO4J_PASSWORD"]
        )

    def setup_neo4j_indexes(self):
        # Index for StandardDocument
        try:
            self.graph.query("CREATE CONSTRAINT standard_document_doc_id IF NOT EXISTS FOR (d:StandardDocument) REQUIRE d.doc_id IS UNIQUE")
            print("Constraint/Index for StandardDocument.doc_id ensured.")
        except Exception as e:
            print(f"Error creating constraint/index for StandardDocument: {e} (This might be fine if it already exists and error handling for that is not robust in the driver/langchain version)")

        # Index for StandardChunk (chunk_id)
        try:
            self.graph.query("CREATE CONSTRAINT standard_chunk_chunk_id IF NOT EXISTS FOR (sc:StandardChunk) REQUIRE sc.chunk_id IS UNIQUE")
            print("Constraint/Index for StandardChunk.chunk_id ensured.")
        except Exception as e:
            print(f"Error creating constraint/index for StandardChunk: {e} (This might be fine if it already exists)")
            
        cohere_embedding_dimension = 1536
        # For 'embed-v4.0' it's 1024 for input type "search_document" or "search_query"
        # Check your specific Cohere model's output dimension if not 1024.
        # For 'embed-english-v3.0' it is 1024.
        # For 'embed-multilingual-light-v3.0' it is 384.

        try:
            self.graph.query(
                f"""CREATE VECTOR INDEX {self.standard_chunk_vector_index_name} IF NOT EXISTS 
                   FOR (sc:{self.standard_chunk_node_label}) ON (sc.{self.standard_chunk_embedding_property}) 
                   OPTIONS {{indexConfig: {{
                    `vector.dimensions`: {cohere_embedding_dimension},
                    `vector.similarity_function`: 'cosine'
                   }}}}"""
            )
            print(f"Vector Index '{self.standard_chunk_vector_index_name}' on :{self.standard_chunk_node_label}({self.standard_chunk_embedding_property}) ensured.")
        except Exception as e:
            print(f"Error creating vector index '{self.standard_chunk_vector_index_name}': {e}")
            print("Please ensure your Neo4j version supports vector indexes (e.g., Aura or Enterprise 5.11+ with appropriate APOC for vector functions if needed by Langchain). Community Edition might have limitations.")

    def init_cypher(self):
        graph_chain = GraphCypherQAChain.from_llm(
            self.llm, 
            graph=self.graph,  
            allow_dangerous_requests=True,
            verbose=False,
            validate_cypher=True
        )
        self.graph_chain = graph_chain
        
    def init_vector(self, url=None, username=None, password=None):
        # This method now strictly tries to load an existing index.
        # The index should have been created by setup_neo4j_indexes.
        # Data is added to this index via the `_create_standard_document_and_chunks` method,
        # which populates the :StandardChunk nodes with their `embedding` property.
        db_url = url or os.environ["NEO4J_URI"]
        db_username = username or os.environ["NEO4J_USERNAME"]
        db_password = password or os.environ["NEO4J_PASSWORD"]
        
        try:
            self.store = Neo4jVector.from_existing_index(
                embedding=self.llm_embbedding, # Corrected parameter name
                url=db_url,
                username=db_username,
                password=db_password,
                index_name=self.standard_chunk_vector_index_name,
                node_label=self.standard_chunk_node_label,
                text_node_property=self.standard_chunk_text_property, # Property to treat as Document.page_content
                embedding_node_property=self.standard_chunk_embedding_property, # Property where embedding vector is stored
            )
            print(f"Vector store '{self.standard_chunk_vector_index_name}' for '{self.standard_chunk_node_label}' initialized from existing index.")
        except ValueError as ve: # Catch specific ValueError for "index does not exist"
            print(f"ValueError during Neo4jVector.from_existing_index: {ve}")
            print(f"This means the vector index '{self.standard_chunk_vector_index_name}' on label '{self.standard_chunk_node_label}' might not exist or is not accessible.")
            print("Please ensure `setup_neo4j_indexes()` ran successfully and created the index in Neo4j.")
            print("If this is the first run, the index will be empty and populated as documents are processed by the `flow` method.")
            # Initialize a new instance that's configured to work with the index once it's populated
            # This allows the application to start, and similarity search will return empty results
            # until data is processed and the index gets populated.
            self.store = Neo4jVector(
                embedding=self.llm_embbedding,
                url=db_url,
                username=db_username,
                password=db_password,
                index_name=self.standard_chunk_vector_index_name,
                node_label=self.standard_chunk_node_label,
                text_node_properties=[self.standard_chunk_text_property], # Expects a list
                embedding_node_property=self.standard_chunk_embedding_property
            )
            print(f"Neo4jVector instance created for '{self.standard_chunk_node_label}'. Index '{self.standard_chunk_vector_index_name}' should exist (created by setup_neo4j_indexes).")

        except Exception as e:
            print(f"An unexpected error occurred during init_vector: {e}")
            traceback.print_exc()
            # Fallback to a basic instance if all else fails, so app can potentially start
            # but vector search will likely not work as expected.
            self.store = Neo4jVector(
                embedding=self.llm_embbedding,
                url=db_url,
                username=db_username,
                password=db_password,
                index_name=self.standard_chunk_vector_index_name, # Attempt to use the configured index
                node_label=self.standard_chunk_node_label,
                text_node_properties=[self.standard_chunk_text_property],
                embedding_node_property=self.standard_chunk_embedding_property
            )
            print(f"CRITICAL FALLBACK: Neo4jVector instance created due to unexpected error. Vector search might be impaired. Error: {e}")


    def reinit_vector(self, url=None, username=None, password=None):
        # Same logic as init_vector for re-initialization
        self.init_vector(url, username, password)


    def read_PDFs_and_create_documents(self, files: List[IO[bytes]], file_names: List[str], 
                                     chunking_strategy: str = "basic", 
                                     # max_characters: float = 1e6, # UnstructuredLoader might not have this for file IO
                                     # include_orig_elements: bool = False # Same as above
                                     ):
        docs = []
        temp_dir = "temp_files_for_unstructured"
        os.makedirs(temp_dir, exist_ok=True)

        for i, file_content_io in enumerate(files):
            file_name = file_names[i]
            temp_file_path = os.path.join(temp_dir, file_name)
            
            # Write the IO[bytes] to a temporary file
            with open(temp_file_path, "wb") as temp_file:
                temp_file.write(file_content_io.read())
            
            # UnstructuredLoader works with file paths
            loader = UnstructuredLoader(
                file_path=temp_file_path,
                # chunking_strategy=chunking_strategy, # This is for Unstructured itself, not Langchain text splitter
                mode="elements", # 'elements' or 'single' or 'paged'
                strategy=chunking_strategy if chunking_strategy in ["fast", "hi_res", "ocr_only"] else "fast", # Unstructured strategy
                # max_characters=int(max_characters), # if using unstructured chunking
                # include_orig_elements=include_orig_elements,
                # metadata_filename=file_name # UnstructuredFileLoader sets 'filename' in metadata
            )
            try:
                loaded_docs_for_file = loader.load()
                # Add original filename to metadata if not already present or to reinforce
                for ld_doc in loaded_docs_for_file:
                    ld_doc.metadata["source_file_name"] = file_name 
                docs.extend(loaded_docs_for_file)
            except Exception as e_load:
                print(f"Error loading file {file_name} with UnstructuredFileLoader: {e_load}")
                traceback.print_exc()
            finally:
                # Clean up the temporary file
                if os.path.exists(temp_file_path):
                    os.remove(temp_file_path)
        
        # Clean up temp_dir if empty (optional)
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
        loop = asyncio.get_running_loop() # Get the current event loop

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

                # ถ้า poller.result() เป็น blocking call ให้รันใน executor
                # สร้าง partial function เพื่อให้ poller.result ถูกเรียกใน thread ที่ถูกต้อง
                blocking_call = functools.partial(poller.result)
                result = await loop.run_in_executor(None, blocking_call)

                if result and hasattr(result, 'pages') and result.pages:
                    for page_idx, page in enumerate(result.pages): # ใช้ _ ถ้า page_idx ไม่ได้ใช้
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

    async def split_documents_into_chunks(self, documents: List[Document], chunk_size: int = 2048, chunk_overlap: int = 1024): # Reduced overlap
        # Using RecursiveCharacterTextSplitter which is good for general text
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            is_separator_regex=False,
            separators=["\n\n", "\n", ". ", " ", ""], # Common separators
        )
        # split_documents is synchronous, so run in thread
        return await asyncio.to_thread(splitter.split_documents, documents)

    async def translate_documents_with_openai(self, documents: List[Document], batch_size: int = 10): # Reduced batch_size for safety
        def contains_thai(text: str) -> bool:
            thai_pattern = re.compile("[\u0E00-\u0E7F]")
            return bool(thai_pattern.search(text))

        llm = self.llm # Assuming self.llm is ChatGoogleGenerativeAI
        prompt = PromptTemplate(
            input_variables=["text"],
            template="You are a professional English translator. Translate the following Thai text accurately and fluently into English. Return ONLY the translated English text, without any additional explanations, conversational phrases, or original Thai text.\nThai Text:\n{text}\n\nEnglish Translation:"
        )
        chain = prompt | llm

        async def run_chain(text_content): # Changed to text_content to avoid confusion with Document object
            if not text_content or not contains_thai(text_content):
                return text_content
            try:
                response = await chain.ainvoke({"text": text_content})
                return response.content
            except Exception as e_translate:
                print(f"Error translating text: {e_translate}. Returning original text.")
                return text_content # Return original on error

        tasks = []
        for doc_idx in range(0, len(documents), batch_size):
            batch_docs = documents[doc_idx:doc_idx + batch_size]
            batch_input_texts = [doc.page_content for doc in batch_docs]
            
            # Create tasks for the current batch
            current_batch_tasks = [run_chain(text) for text in batch_input_texts]
            tasks.extend(current_batch_tasks)

        # Gather results from all tasks
        all_translated_texts = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Assign translated texts back to documents
        # This assumes all_translated_texts aligns with the original documents list
        # and that tasks were added in the correct order.
        # Need to be careful here if batching modified the order or if tasks were processed out of order.
        
        # Safer way: Reconstruct the results based on original document list
        processed_idx = 0
        for i in range(len(documents)):
            # This logic assumes one task per document, which is how tasks were created
            if isinstance(all_translated_texts[processed_idx], Exception):
                print(f"Translation task for document index {i} failed: {all_translated_texts[processed_idx]}")
                # Keep original content if translation failed
            elif all_translated_texts[processed_idx]: # Check if not None or empty
                documents[i].page_content = all_translated_texts[processed_idx]
            processed_idx += 1
            
        return documents


    def __get_auth_env(self): # This remains synchronous
        # Simplified: Always use the main env vars unless explicitly overridden
        # Test environment should be handled by test configurations / separate .env files for tests
        return os.environ["NEO4J_URI"], os.environ["NEO4J_USERNAME"], os.environ["NEO4J_PASSWORD"]

    def add_description_embeddings_to_nodes(self): # This remains synchronous
        print("Adding description embeddings to specified nodes (e.g., __Entity__)...")
        url, username, password = self.__get_auth_env()
        try:
            # This creates a new vector index named "description" for nodes with label "__Entity__"
            # Make sure these nodes and properties exist.
            Neo4jVector.from_existing_graph(
                embedding=self.llm_embbedding, # Corrected parameter name
                url=url,
                username=username,
                password=password,
                node_label='__Entity__', # Only for nodes created by LLMGraphTransformer with this label
                text_node_properties=['id', 'description'], # Properties to embed from __Entity__
                embedding_node_property="description_embedding", # Where to store this new embedding
                index_name="entity_description_embedding_index", # Give it a unique name
            )
            print("Description embeddings potentially added/updated for __Entity__ nodes.")
        except Exception as e_desc_emb:
            print(f"Error in add_description_embeddings_to_nodes: {e_desc_emb}")
            print("This might happen if no '__Entity__' nodes with 'id' or 'description' exist yet.")
            traceback.print_exc()


    async def _create_standard_document_and_chunks(self, 
                                               file_name: str, 
                                               translated_chunks: List[Document], 
                                               standard_title: str = None):
        doc_id = file_name 
        doc_title = standard_title if standard_title else file_name
        
        merge_doc_query = """
        MERGE (d:StandardDocument {doc_id: $doc_id})
        ON CREATE SET d.title = $title, d.source_file_name = $source_file_name, d.last_processed_timestamp = timestamp()
        ON MATCH SET d.title = $title, d.source_file_name = $source_file_name, d.last_processed_timestamp = timestamp()
        RETURN d.doc_id AS document_id
        """
        doc_result = await asyncio.to_thread(
            self.graph.query, 
            merge_doc_query, 
            params={'doc_id': doc_id, 'title': doc_title, 'source_file_name': file_name}
        )
        created_doc_id = doc_result[0]['document_id'] if doc_result and doc_result[0] else doc_id
        print(f"MERGED StandardDocument: {created_doc_id}")

        chunk_details_for_graph_transformer = []
        chunk_nodes_data_for_cypher = []

        for i, chunk_doc in enumerate(translated_chunks):
            chunk_text = chunk_doc.page_content
            metadata = chunk_doc.metadata
            original_section_id = metadata.get('category', metadata.get('header_1', metadata.get('section', None))) # Try a few common keys from Unstructured
            page_number = metadata.get('page_number', None)

            embedding_vector = await asyncio.to_thread(self.llm_embbedding.embed_query, chunk_text)
            
            chunk_id = f"{created_doc_id}_chunk_{i:04d}"

            chunk_data_for_cypher = {
                'chunk_id': chunk_id,
                'text': chunk_text,
                'original_section_id': str(original_section_id) if original_section_id else None, # Ensure string or null
                'page_number': page_number,
                'sequence_in_document': i,
                'embedding': embedding_vector,
                'doc_id': created_doc_id 
            }
            chunk_nodes_data_for_cypher.append(chunk_data_for_cypher)
            
            # Prepare Langchain Document for LLMGraphTransformer, pass chunk_id in metadata
            transformer_doc_metadata = {'chunk_id': chunk_id, 'doc_id': created_doc_id}
            if 'source_file_name' in metadata: # Propagate original filename if available
                 transformer_doc_metadata['source_file_name'] = metadata['source_file_name']
            if page_number is not None:
                 transformer_doc_metadata['page_number'] = page_number

            chunk_details_for_graph_transformer.append(
                Document(page_content=chunk_text, metadata=transformer_doc_metadata)
            )

        if chunk_nodes_data_for_cypher:
            create_chunks_query = f"""
            UNWIND $chunks_data AS chunk_props
            MATCH (d:StandardDocument {{doc_id: chunk_props.doc_id}})
            MERGE (sc:{self.standard_chunk_node_label} {{chunk_id: chunk_props.chunk_id}})
            ON CREATE SET 
                sc.text = chunk_props.text,
                sc.original_section_id = chunk_props.original_section_id,
                sc.page_number = chunk_props.page_number,
                sc.sequence_in_document = chunk_props.sequence_in_document,
                sc.embedding = chunk_props.embedding,
                sc.doc_id = chunk_props.doc_id,
                sc.last_updated = timestamp()
            ON MATCH SET
                sc.text = chunk_props.text, 
                sc.embedding = chunk_props.embedding,
                sc.original_section_id = chunk_props.original_section_id,
                sc.page_number = chunk_props.page_number,
                sc.last_updated = timestamp()
            MERGE (d)-[r:HAS_CHUNK {{order: chunk_props.sequence_in_document}}]->(sc)
            RETURN count(sc) AS chunks_processed
            """
            result = await asyncio.to_thread(
                self.graph.query, 
                create_chunks_query, 
                params={'chunks_data': chunk_nodes_data_for_cypher}
            )
            if result and result[0]:
                 print(f"Processed {result[0].get('chunks_processed', 0)} StandardChunk nodes for document {created_doc_id}")
            else:
                 print(f"No chunk processing result for document {created_doc_id}")

        return chunk_details_for_graph_transformer

    async def _apply_llm_graph_transformer_to_chunks(self, 
                                                chunk_documents: List[Document], # These are Langchain Docs with chunk_id in metadata
                                                llm_transformer: LLMGraphTransformer):
        print(f"Applying LLMGraphTransformer to {len(chunk_documents)} chunk documents...")
        if not chunk_documents:
            return

        # Process documents in batches for LLMGraphTransformer if the list is very large
        # For now, processing all at once. LLMGraphTransformer's aconvert_to_graph_documents handles a list.
        graph_documents_from_llm: List[GraphDocument] = await llm_transformer.aconvert_to_graph_documents(chunk_documents)
        
        if not graph_documents_from_llm:
            print("LLMGraphTransformer did not return any graph documents.")
            return

        print(f"LLMGraphTransformer returned {len(graph_documents_from_llm)} graph documents.")
        
        # Add entities and relationships extracted by LLMGraphTransformer
        # This uses baseEntityLabel=True, so nodes will get __Entity__ and their specific type.
        try:
            # add_graph_documents is synchronous
            await asyncio.to_thread(
                self.graph.add_graph_documents,
                graph_documents_from_llm,
                baseEntityLabel=True, 
                include_source=False # We handle linking manually for precision
            )
            print(f"Added nodes and relationships from LLMGraphTransformer.")
        except Exception as e_add_graph:
            print(f"Error adding graph documents from LLMTransformer: {e_add_graph}")
            traceback.print_exc()
            return # Stop if we can't add the base entities

        # Manually create relationships from StandardChunk to the extracted __Entity__ nodes
        link_params = []
        for gd in graph_documents_from_llm:
            # gd.source is the original Langchain Document we passed, which has chunk_id in metadata
            source_chunk_id = gd.source.metadata.get('chunk_id')
            if not source_chunk_id:
                print(f"Warning: GraphDocument source missing chunk_id: {gd.source.page_content[:50]}...")
                continue
            
            for node_in_gd in gd.nodes: # node_in_gd is a GraphNode from Langchain
                entity_id = node_in_gd.id # This is typically the 'name' or main identifier of the entity
                entity_label = node_in_gd.type # The specific label like 'Person', 'Organization'
                
                link_params.append({
                    'chunk_id': source_chunk_id,
                    'entity_id_prop_val': entity_id, # The value of the 'id' property of the __Entity__ node
                    'entity_specific_label': entity_label # The specific label transformer assigned
                })
        
        if link_params:
            # Query to link StandardChunk to __Entity__ nodes
            # Assumes __Entity__ nodes were created by add_graph_documents with an 'id' property.
            link_entities_query = """
            UNWIND $link_data AS item
            MATCH (sc:StandardChunk {chunk_id: item.chunk_id})
            MATCH (e:__Entity__ {id: item.entity_id_prop_val}) 
            // Optionally, ensure the specific label exists if LLMGraphTransformer is inconsistent,
            // though baseEntityLabel=True should make it add both __Entity__ and the specific type.
            // CALL apoc.create.addLabels(e, [item.entity_specific_label]) YIELD node AS entityWithCorrectLabel
            MERGE (sc)-[:CONTAINS_ENTITY]->(e) // Link to 'e' which is the __Entity__ node
            RETURN count(DISTINCT sc) AS linked_chunks, count(DISTINCT e) as linked_entities
            """
            try:
                result = await asyncio.to_thread(self.graph.query, link_entities_query, params={'link_data': link_params})
                if result and result[0]:
                    print(f"Linked StandardChunks to __Entity__ nodes: {result[0]}")
                else:
                    print("No links made between StandardChunks and __Entity__ nodes, or query returned empty.")
            except Exception as e_link:
                print(f"Error linking LLMTransformer __Entity__ nodes to StandardChunks: {e_link}")
                traceback.print_exc()
        else:
            print("No valid link parameters generated for linking StandardChunks to __Entity__ nodes.")


    async def flow(self, files: List[IO[bytes]], file_names: List[str]):
        loop = asyncio.get_running_loop()

        # 1. Load PDFs and create Langchain Document objects (one per page using Azure)
        if self.document_intelligence_client:
            print("Using Azure Document Intelligence for PDF processing.")
            # documents_from_pdf_loader จะเป็น List[Document] โดยแต่ละ Document คือ 1 หน้าจาก PDF
            documents_from_pdf_loader = await self.read_PDFs_and_create_documents_azure(files, file_names)
        else:
            print("Azure Document Intelligence client not available, falling back to UnstructuredFileLoader (if implemented).")
            # Fallback to original method if Azure client is not available (optional)
            # หรือคุณอาจจะ raise error ถ้าต้องการให้ Azure เป็น prerequisite
            documents_from_pdf_loader = await loop.run_in_executor(None, self.read_PDFs_and_create_documents, files, file_names)
        
        print(f"Loaded {len(documents_from_pdf_loader)} initial document(s)/element(s) from PDF processing.")

        if not documents_from_pdf_loader:
            print("No documents were loaded from PDFs. Aborting flow.")
            return

        # 2. Split these (potentially large) documents into smaller manageable chunks
        #    This is where the main chunking for our StandardChunk nodes happens.
        split_chunks_lc_docs = await self.split_documents_into_chunks(documents_from_pdf_loader)
        print(f"Split initial documents into {len(split_chunks_lc_docs)} Langchain Document chunks.")
        
        if not split_chunks_lc_docs:
            print("No chunks were created after splitting. Aborting flow.")
            return

        # 3. Translate these smaller chunks
        translated_chunks_lc_docs = await self.translate_documents_with_openai(split_chunks_lc_docs)
        print(f"Translated {len(translated_chunks_lc_docs)} chunks.")
        
        # Group translated chunks by their original source file name for processing per document
        # The metadata 'source_file_name' should have been added in read_PDFs_and_create_documents
        # and propagated by split_documents_into_chunks and translate_documents_with_openai
        
        docs_by_source_file = {}
        for chunk_lc_doc in translated_chunks_lc_docs:
            source_fn = chunk_lc_doc.metadata.get('source_file_name')
            if not source_fn:
                print(f"Warning: Chunk missing 'source_file_name' in metadata: {chunk_lc_doc.page_content[:50]}...")
                # Fallback or skip this chunk if filename is crucial
                source_fn = "unknown_source_file.pdf" # Example fallback
            
            if source_fn not in docs_by_source_file:
                docs_by_source_file[source_fn] = []
            docs_by_source_file[source_fn].append(chunk_lc_doc)

        esg_allowed_relationships: List[str] = [
            "HAS_FRAMEWORK", "COMPLIES_WITH", "REFERENCES_STANDARD", "DISCLOSES_INFORMATION_ON",
            "REPORTS_ON", "APPLIES_TO", "MENTIONS", "DEFINES",
            "HAS_METRIC", "MEASURES", "TRACKS_KPI", "SETS_TARGET", "ACHIEVES_GOAL",
            "IDENTIFIES_RISK", "PRESENTS_OPPORTUNITY", "HAS_IMPACT_ON", "MITIGATES_RISK",
            "HAS_STRATEGY", "IMPLEMENTS_POLICY", "FOLLOWS_PROCESS", "ENGAGES_IN_ACTIVITY",
            "INVESTS_IN", "OPERATES_IN", "LOCATED_IN", "GOVERNED_BY",
            "PART_OF", "COMPONENT_OF", "SUBCLASS_OF", "INSTANCE_OF",
            "RELATES_TO", "ASSOCIATED_WITH", "CAUSES", "INFLUENCES", "CONTRIBUTES_TO",
            "REQUIRES", "ADDRESSES", "MANAGES", "MONITORS", "EVALUATES",
            "ISSUED_BY", "DEVELOPED_BY", "PUBLISHED_IN", "EFFECTIVE_DATE"
        ]

        esg_allowed_nodes: List[str] = [
            "Organization", "Person", "Standard", "Framework", "Guideline", "Regulation", "Law",
            "Report", "Document", "Disclosure", "Topic", "SubTopic", "Category",
            "Metric", "Indicator", "KPI", "Target", "Goal", "Objective",
            "Risk", "Opportunity", "Impact", "Strategy", "Policy", "Process", "Practice",
            "Activity", "Initiative", "Program", "Project",
            "Stakeholder", "Investor", "Employee", "Customer", "Supplier", "Community", "Government",
            "Region", "Country", "Location", "Facility",
            "EnvironmentalFactor", "SocialFactor", "GovernanceFactor",
            "ClimateChange", "GreenhouseGas", "GHGEmissions", "Energy", "Water", "Waste", "Biodiversity",
            "HumanRights", "LaborPractices", "HealthAndSafety", "DiversityAndInclusion", "CommunityEngagement",
            "BoardComposition", "Ethics", "Corruption", "ShareholderRights",
            "EconomicValue", "FinancialPerformance", "OperatingCost", "Revenue", "Investment",
            "Date", "Year", "Period", "Unit", "Value", "Percentage", "Currency", "Source", "Reference", "Term", "Concept"
        ]

        llm_transformer_for_kg = LLMGraphTransformer(
            llm=self.llm,
            node_properties=["description", "name"],
            relationship_properties=["description"],
            strict_mode=True,
            allowed_nodes=esg_allowed_nodes, # <--- เพิ่มรายการ Node ที่อนุญาต
            allowed_relationships=esg_allowed_relationships, # <--- เพิ่มรายการ Relationship ที่อนุญาต
            prompt=custom_esg_graph_extraction_prompt # <--- ใช้ Prompt ที่ปรับแต่งเอง
        )
        
        all_chunk_docs_for_llm_transformer = []

        for original_file_name, chunks_for_this_file in docs_by_source_file.items():
            print(f"Processing StandardDocument and StandardChunks for: {original_file_name}")
            standard_title_from_filename = original_file_name.replace("_", " ").replace(".pdf", "") 

            # Create StandardDocument and StandardChunk nodes in Neo4j
            # This returns Langchain Document objects with chunk_id in metadata, for the LLMTransformer
            chunk_docs_with_ids_for_transformer = await self._create_standard_document_and_chunks(
                file_name=original_file_name,
                translated_chunks=chunks_for_this_file, # These are already translated
                standard_title=standard_title_from_filename 
            )
            all_chunk_docs_for_llm_transformer.extend(chunk_docs_with_ids_for_transformer)
            print(f"Stored StandardDocument and StandardChunks for {original_file_name} in Neo4j.")

        # After all StandardDocument/StandardChunk nodes are created for all files:
        # Apply LLMGraphTransformer to all collected chunk_documents (which now have chunk_id in metadata)
        if all_chunk_docs_for_llm_transformer:
            await self._apply_llm_graph_transformer_to_chunks(
                all_chunk_docs_for_llm_transformer,
                llm_transformer_for_kg
            )
            print("LLMGraphTransformer processing completed for all chunks.")

            # The original add_description_embeddings_to_nodes worked on `__Entity__` nodes.
            # If LLMGraphTransformer creates these, this step is still relevant.
            print("Attempting to add description embeddings to __Entity__ nodes...")
            # This is a synchronous function
            await loop.run_in_executor(None, self.add_description_embeddings_to_nodes)
        else:
            print("No chunk documents prepared for LLMGraphTransformer.")
        
        print("PDF processing flow completed for all files.")


    # Methods like get_relate, get_cypher_answer, get_output, is_graph_substantially_empty
    # will now work with self.store which should be configured for StandardChunk
    async def get_relate(self, query, k):        
        results = []
        if not self.store:
            print("ERROR: Neo4jVector store (self.store) is not initialized in get_relate.")
            return results
        try:
            # Ensure self.store is configured for StandardChunk and the correct index
            docs_with_score = await self.store.asimilarity_search_with_score(query, k=k)
            for doc, score in docs_with_score:
                # doc.page_content will be the 'text' property of StandardChunk
                # doc.metadata will contain other properties of StandardChunk
                results.append(doc)
        except Exception as e_relate:
            print(f"Error in get_relate during similarity search: {e_relate}")
            traceback.print_exc()
        return results

    # get_cypher_answer and get_output might need review if they depend on specific
    # node labels or vector indexes other than the new StandardChunk one.
    # The current get_cypher_answer uses index_name="description" which is for __Entity__ nodes.
    # This might still be relevant if you want to search through entities created by LLMGraphTransformer.

    async def get_cypher_answer(self, query):
        # This method searches the "entity_description_embedding_index" on __Entity__ nodes.
        # This is separate from the main StandardChunk vector search.
        url, username, password = self.__get_auth_env()
        entity_description_index_name = "entity_description_embedding_index" # As defined in add_description_embeddings_to_nodes
        # Updated retrieval_query to be more robust and useful.
        # This query assumes entities have a 'description' and an 'id' (name).
        # It fetches the entity and its direct neighbors.
        retrieval_query_for_entities = """
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
        try:
            entity_vector_store = Neo4jVector.from_existing_index(
                embedding=self.llm_embbedding, # Corrected parameter name
                url=url,
                username=username,
                password=password,
                index_name=entity_description_index_name,
                text_node_property="description", # Text property on __Entity__
                embedding_node_property="description_embedding", # Embedding on __Entity__
                retrieval_query=retrieval_query_for_entities
            )
            docs = await entity_vector_store.asimilarity_search(query=query, k=3) # k is number of entities
            # Format into a string representation as before, if that's the desired output
            print(f"--- [DEBUG] Retrieved docs for query: '{query}' ---")
            if docs:
                for i, doc_result in enumerate(docs):
                    print(f"Doc {i+1} page_content: {doc_result.page_content}")
                    print(f"Doc {i+1} metadata: {doc_result.metadata}")
            else:
                print("No documents retrieved.")
            print("----------------------------------------------------")

            return f"{[doc.page_content + ' METADATA: ' + str(doc.metadata) for doc in docs]}" 
        except ValueError as ve:
            print(f"ValueError in get_cypher_answer (likely index '{entity_description_index_name}' not found): {ve}")
            return "Could not retrieve answer from entity graph: entity description index not found or not populated."
        except Exception as e_cypher_ans:
            print(f"Error in get_cypher_answer: {e_cypher_ans}")
            traceback.print_exc()
            return "Error retrieving answer from entity graph."

            
    async def get_output(self, query, k=5) -> RetrieveAnswer:
        # This will search StandardChunk nodes
        related_standard_chunks = await self.get_relate(query, k) 
        
        # This will search __Entity__ nodes based on their descriptions
        cypher_answer_from_entities = await self.get_cypher_answer(query)
        
        # Combine or decide how to use these two pieces of information
        # For now, returning entity answer as cypher_answer and chunks as relate_documents
        return RetrieveAnswer(
            cypher_answer=cypher_answer_from_entities, # Context from Entities
            relate_documents=related_standard_chunks   # Context from Standard Chunks
        )

    async def is_graph_substantially_empty(self, threshold: int = 10) -> bool:
        if not self.graph:
            print("[NEO4J_SERVICE ERROR] Neo4jGraph (self.graph) is not initialized.")
            return True 
        query = "MATCH (n) RETURN count(n) AS node_count"
        try:
            loop = asyncio.get_running_loop()
            result_data = await loop.run_in_executor(None, self.graph.query, query)
            if result_data and "node_count" in result_data[0]:
                node_count = result_data[0]["node_count"]
                print(f"[NEO4J_SERVICE LOG] Current node count in graph: {node_count}")
                return node_count < threshold
            else:
                print(f"[NEO4J_SERVICE WARNING] Could not retrieve valid node count. Result: {result_data}. Assuming not empty.")
                return False 
        except Exception as e:
            print(f"[NEO4J_SERVICE ERROR] Error checking if graph is empty: {e}. Assuming not empty.")
            return False
        
    async def get_all_standard_chunks_for_theme_generation(self) -> List[dict[str, any]]:
            query = """
            MATCH (sc:StandardChunk)
            WHERE sc.text IS NOT NULL AND trim(sc.text) <> "" AND sc.embedding IS NOT NULL
            RETURN 
                sc.chunk_id AS chunk_id,
                sc.text AS text,
                sc.embedding AS embedding, // embedding ของ chunk
                sc.doc_id AS source_document_doc_id, // ID ของ StandardDocument ต้นทาง
                sc.original_section_id AS original_section_id, // เลขข้อ/หัวข้อเดิมจาก PDF
                sc.page_number AS page_number
            """
            # หมายเหตุ: ถ้า self.graph.query เป็น synchronous
            loop = asyncio.get_running_loop()
            try:
                results = await loop.run_in_executor(None, self.graph.query, query)
                # results จะเป็น list of dictionaries
                # [{ 'chunk_id': '...', 'text': '...', 'embedding': [...], ...}, ...]
                if results:
                    print(f"[NEO4J_SERVICE LOG] Retrieved {len(results)} StandardChunks for theme generation.")
                    return results
                else:
                    print("[NEO4J_SERVICE WARNING] No StandardChunks found for theme generation.")
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
        max_total_context_items_str_len: int = 3000
    ) -> Optional[str]:
        if not theme_name and not theme_keywords:
            print("[NEO4J WARNING] Theme name or keywords required for get_graph_context_for_theme_chunks_v2.")
            return "Theme name or keywords required to find central entities for graph context."

        central_entity_ids: List[str] = []
        try:
            url, username, password = self.__get_auth_env()
            entity_description_index_name = "entity_description_embedding_index"

            # Query เพื่อดึง ID และข้อมูลเบื้องต้นของ Entity ที่เกี่ยวข้องกับ Keywords
            # เราต้องการ text (description) สำหรับ vector search และ metadata (id, labels)
            temp_entity_retrieval_query = """
            RETURN COALESCE(node.description, node.name, node.id, '') AS text, // <--- เพิ่ม node.name, node.id เป็น fallback สำหรับ text
                score, 
                {
                    id: node.id,
                    name: node.name, // <-- ถ้ามี property name
                    description: node.description,
                    labels: labels(node)
                } AS metadata
            """

            entity_store_for_seed_search = Neo4jVector.from_existing_index(
                embedding=self.llm_embbedding,
                url=url, username=username, password=password,
                index_name=entity_description_index_name,
                text_node_property="description",
                embedding_node_property="description_embedding",
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
                print(f"[NEO4J INFO] No distinct central entities found for theme '{theme_name}' using vector search on keywords.")
                # (ส่วน fallback logic เดิมถ้ามี)
                pass 
            
            if not central_entity_ids: # Double check after any fallback
                print(f"[NEO4J WARNING] No central entities found for theme '{theme_name}' to build graph context.")
                return "No central entities identified for this theme to build detailed graph context."

            print(f"[NEO4J INFO] Found central entity IDs for theme '{theme_name}': {central_entity_ids}")

        except neo4j.exceptions.CypherSyntaxError as cse: # ดักจับ CypherSyntaxError โดยเฉพาะ
            print(f"[NEO4J ERROR] CypherSyntaxError finding central entities for theme '{theme_name}': {cse}")
            print(f"Problematic Query (ensure no Python comments like # are inside the query string):\n{temp_entity_retrieval_query}")
            traceback.print_exc()
            return f"CypherSyntaxError finding central entities. Please check query syntax. Error: {cse.message}"
        except ValueError as ve: 
            print(f"[NEO4J ERROR] ValueError finding central entities for theme '{theme_name}': {ve}")
            traceback.print_exc()
            return f"ValueError finding central entities for graph context. Check if entities have descriptions. Error: {ve}"
        except Exception as e_seed_search:
            print(f"[NEO4J ERROR] Error finding central entities for theme '{theme_name}': {e_seed_search}")
            traceback.print_exc()
            return "Error finding central entities for graph context."

        final_graph_context_parts = []
        loop = asyncio.get_running_loop()
        
        # Query เพื่อดึง neighborhood ของแต่ละ central_entity_id
        # แก้ไข {{id: ...}} เป็น {id: ...}
        entity_neighborhood_query_template = """
        MATCH (node:__Entity__ {id: $center_entity_id}) 
        OPTIONAL MATCH (node)-[rel*1..{max_hops}]-(neighbor) 
        WHERE neighbor IS NOT NULL AND (NOT 'StandardChunk' IN labels(neighbor) AND NOT 'StandardDocument' IN labels(neighbor)) // Ensure neighbor is not a chunk/doc
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
            if total_context_len >= max_total_context_items_str_len : break # Check before making more DB calls
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
                        for rel_detail_idx, rel_detail in enumerate(relations_list):
                            if rel_detail_idx >= max_relations_to_collect_per_central_entity: break # Should be handled by Cypher's slice [0..{max_rels}] but double check

                            neighbor_details = rel_detail.get('neighbor_details', {})
                            n_id = neighbor_details.get('id', 'UnknownNeighbor')
                            n_desc = neighbor_details.get('description', 'N/A')
                            n_labels_list = neighbor_details.get('labels', [])
                            display_n_labels = n_labels_list if n_labels_list else ["Node"]
                            
                            rel_path_types_str = ", ".join(rel_detail.get('relationship_path', ['RELATED_TO']))
                            part_str += f"\n    --[{rel_path_types_str}]--> NEIGHBOR: {n_id} (Type: {', '.join(display_n_labels)}; Desc: {n_desc})"
                    else:
                        part_str += "\n  (No distinct relationships/neighbors found for this entity)"
                    
                    if total_context_len + len(part_str) + 2 <= max_total_context_items_str_len: # +2 for \n\n
                        final_graph_context_parts.append(part_str)
                        total_context_len += len(part_str) + 2
                    else:
                        final_graph_context_parts.append("... [Further graph context truncated due to length limit]")
                        break # Stop adding more entities if total length exceeded
            except Exception as e_entity_ctx:
                print(f"[NEO4J ERROR] Error fetching context for central entity '{center_id}': {e_entity_ctx}")
                # traceback.print_exc() # Uncomment for full traceback if needed
        
        if not final_graph_context_parts:
            return "No detailed graph context elements could be formatted from central entities."
        
        return "\n\n".join(final_graph_context_parts)
