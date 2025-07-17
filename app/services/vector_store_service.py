import os
import logging
from typing import List
from langchain_core.documents import Document
from langchain_community.graphs import Neo4jGraph
from langchain_community.vectorstores import Neo4jVector
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import uuid
import json

class VectorStoreService:
    """
    A dedicated service for handling vector search operations for user report generation,
    using a separate Neo4j instance as the backend vector store.
    This version includes manual batching to fix embedding and Cypher syntax errors.
    """
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        uri = os.getenv("VECTOR_STORE_NEO4J_URI")
        username = os.getenv("VECTOR_STORE_NEO4J_USERNAME")
        password = os.getenv("VECTOR_STORE_NEO4J_PASSWORD")
        
        if not all([uri, username, password]):
            raise ValueError("VECTOR_STORE_NEO4J environment variables not set correctly.")

        self.graph = Neo4jGraph(url=uri, username=username, password=password)
        self.embedding_model = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
        self.logger.info(f"VectorStoreService initialized, connected to Neo4j at {uri}.")

    def initialize_vector_index(self, index_name: str, documents: List[Document]):
        """
        Manually handles embedding and upserting documents in batches.
        """
        self.logger.info(f"Initializing Neo4j Vector Index '{index_name}' with {len(documents)} documents...")
        # Replace hyphens for safe use in labels and index names
        safe_index_name = index_name.replace('-', '_')
        node_label = f"Chunk_{safe_index_name}"
        
        try:
            # 1. ลบข้อมูลเก่าและ Index เก่า (ถ้ามี) ด้วย Cypher ที่ถูกต้อง
            self.logger.info(f"Pre-deleting any existing data and index for: {node_label}")
            self.graph.query(f"DROP INDEX `{safe_index_name}` IF EXISTS")
            self.graph.query(f"MATCH (n:`{node_label}`) DETACH DELETE n")

            # 2. สร้าง Vector Index Schema ด้วยตัวเอง
            self.graph.query(
                f"""
                CREATE VECTOR INDEX `{safe_index_name}` IF NOT EXISTS
                FOR (n:`{node_label}`) ON (n.embedding)
                OPTIONS {{ indexConfig: {{
                    `vector.dimensions`: 768,
                    `vector.similarity_function`: 'cosine'
                }} }}
                """
            )
            
            # 3. จัดการ Upsert ข้อมูลเป็น Batch ด้วยตัวเอง
            batch_size = 50 
            for i in range(0, len(documents), batch_size):
                batch_docs = documents[i : i + batch_size]
                texts = [doc.page_content for doc in batch_docs]
                
                self.logger.info(f"Embedding batch {i//batch_size + 1}...")
                embeddings = self.embedding_model.embed_documents(texts)
                
                cypher_query = f"""
                UNWIND $data AS row
                CREATE (n:`{node_label}`)
                SET n.id = row.id,
                    n.text = row.text,
                    n.embedding = row.embedding,
                    n.metadata = row.metadata
                """
                
                data_to_upsert = [
                    {
                        "id": str(uuid.uuid4()),
                        "text": doc.page_content,
                        "embedding": embeddings[j],
                        "metadata": json.dumps(doc.metadata)
                    }
                    for j, doc in enumerate(batch_docs)
                ]
                
                self.logger.info(f"Upserting batch {i//batch_size + 1} to Neo4j...")
                self.graph.query(cypher_query, {"data": data_to_upsert})

            self.logger.info(f"Successfully initialized and populated Neo4j Vector Index '{index_name}'.")

        except Exception as e:
            self.logger.error(f"Failed to initialize Neo4j vector index '{index_name}': {e}", exc_info=True)
            raise

    def vector_search(self, index_name: str, query_text: str, top_k: int = 150) -> List[str]:
        """Performs vector similarity search and returns the text content of matching nodes."""
        safe_index_name = index_name.replace('-', '_')
        node_label = f"Chunk_{safe_index_name}"
        self.logger.info(f"Performing vector search in index '{safe_index_name}' on label '{node_label}'...")
        try:
            vector_store = Neo4jVector.from_existing_index(
                embedding=self.embedding_model,
                graph=self.graph,
                index_name=safe_index_name,
                node_label=node_label,
                text_node_property="text",
            )
            results = vector_store.similarity_search(query=query_text, k=top_k)
            self.logger.info(f"Found {len(results)} relevant chunks from vector search.")
            return [doc.page_content for doc in results]
        except Exception as e:
            self.logger.error(f"Failed to perform vector search on Neo4j index '{index_name}': {e}", exc_info=True)
            return []
        
    def delete_index(self, index_name: str):
        """Deletes all data nodes and the vector index schema for a specific index."""
        safe_index_name = index_name.replace('-', '_')
        node_label = f"Chunk_{safe_index_name}"
        self.logger.info(f"Deleting data and index schema for '{index_name}' (label: {node_label}).")
        try:
            self.graph.query(f"MATCH (n:`{node_label}`) DETACH DELETE n")
            self.logger.info(f"Successfully deleted data nodes for label '{node_label}'.")
            self.graph.query(f"DROP INDEX `{safe_index_name}` IF EXISTS")
            self.logger.info(f"Successfully dropped vector index schema '{safe_index_name}'.")
        except Exception as e:
            self.logger.error(f"Failed to delete Neo4j index '{index_name}': {e}", exc_info=True)