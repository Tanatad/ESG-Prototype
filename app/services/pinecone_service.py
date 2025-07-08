import os
import traceback
import logging
from typing import List
from pinecone import Pinecone as PineconeClient, ServerlessSpec
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

# The circular import was here and has been removed.

class PineconeService:
    def __init__(self, embedding_model: Embeddings):
        self.logger = logging.getLogger(__name__)
        api_key = os.getenv("PINECONE_API_KEY")
        if not api_key:
            raise ValueError("PINECONE_API_KEY environment variable not set.")
        
        self.client = PineconeClient(api_key=api_key)
        self.embedding_model = embedding_model
        self.logger.info("PineconeService initialized.")

    def create_and_get_index(self, index_name: str, dimension: int = 1536):
        """Creates a serverless Pinecone index if it doesn't exist and returns it."""
        if index_name not in self.client.list_indexes().names():
            self.logger.info(f"Creating new Pinecone index: {index_name}")
            self.client.create_index(
                name=index_name,
                dimension=dimension,
                metric="cosine",
                spec=ServerlessSpec(
                    cloud="aws",
                    region="us-east-1" # Or your preferred region
                )
            )
        return self.client.Index(index_name)

    def upsert_documents(self, index_name: str, documents: List[Document], batch_size: int = 100):
        """Embeds and upserts documents into a specified Pinecone index."""
        try:
            index = self.create_and_get_index(index_name)
            
            for i in range(0, len(documents), batch_size):
                batch_docs = documents[i : i + batch_size]
                texts = [doc.page_content for doc in batch_docs]
                embeddings = self.embedding_model.embed_documents(texts)
                
                vectors_to_upsert = []
                for idx, doc in enumerate(batch_docs):
                    vector = {
                        "id": f"doc_{index_name}_chunk_{i+idx}",
                        "values": embeddings[idx],
                        "metadata": {"text": doc.page_content, **doc.metadata}
                    }
                    vectors_to_upsert.append(vector)
                
                if vectors_to_upsert:
                    index.upsert(vectors=vectors_to_upsert)
            self.logger.info(f"Successfully upserted {len(documents)} documents to index '{index_name}'.")
        except Exception as e:
            self.logger.error(f"Failed to upsert documents to Pinecone: {e}", exc_info=True)
            raise

    def query(self, index_name: str, query_text: str, top_k: int = 5) -> List[str]:
        """Queries a Pinecone index and returns the text of the top_k most relevant documents."""
        try:
            if index_name not in self.client.list_indexes().names():
                self.logger.warning(f"Query attempted on non-existent index: {index_name}")
                return []

            index = self.client.Index(index_name)
            query_embedding = self.embedding_model.embed_query(query_text)
            
            results = index.query(
                vector=query_embedding,
                top_k=top_k,
                include_metadata=True
            )
            
            return [match['metadata']['text'] for match in results['matches']]
        except Exception as e:
            self.logger.error(f"Failed to query Pinecone index '{index_name}': {e}", exc_info=True)
            return []

    def delete_index(self, index_name: str):
        """Deletes a Pinecone index."""
        try:
            if index_name in self.client.list_indexes().names():
                self.client.delete_index(index_name)
                self.logger.info(f"Successfully deleted Pinecone index: {index_name}")
        except Exception as e:
            self.logger.error(f"Failed to delete Pinecone index '{index_name}': {e}", exc_info=True)