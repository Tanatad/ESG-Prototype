import os
from dotenv import load_dotenv
from fastapi import Depends
from typing import Optional

# --- Import Service Classes ---
from app.services.neo4j_service import Neo4jService
from app.services.question_generation_service import QuestionGenerationService
from app.services.chat_service import ChatService
from app.services.persistence.mongodb import MongoDBSaver
from app.services.pinecone_service import PineconeService
from app.services.report_generation_service import ReportGenerationService
from app.services.rate_limit import RateLimiter

# --- Import Initializer Classes ---
from langchain_cohere import CohereEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

# --- Singleton Instances to hold created services ---
neo4j_service_instance: Optional[Neo4jService] = None
mongodb_service_instance: Optional[MongoDBSaver] = None
qg_service_instance: Optional[QuestionGenerationService] = None
chat_service_instance: Optional[ChatService] = None
pinecone_service_instance: Optional[PineconeService] = None
report_service_instance: Optional[ReportGenerationService] = None


# --- Central Initializer (Called from FastAPI Lifespan) ---
async def initialize_global_services():
    """
    Initializes all shared services when the FastAPI app starts.
    """
    global neo4j_service_instance, mongodb_service_instance, qg_service_instance, chat_service_instance, pinecone_service_instance, report_service_instance
    
    if neo4j_service_instance is not None:
        print("[FastAPI DI] Services already initialized. Skipping.")
        return

    print("[FastAPI DI] Initializing all global services...")

    # --- FIX: Initialize all shared components here ---
    embedding_model = CohereEmbeddings(
        model='embed-v4.0', 
        cohere_api_key=os.getenv("COHERE_API_KEY")
    )
    rate_limiter = RateLimiter(requests_per_minute=int(os.getenv("REQUESTS_PER_MINUTE", "60")))
    
    llm_instance = ChatGoogleGenerativeAI(
        model=os.getenv("CHAT_MODEL", "gemini-2.5-flash-preview-05-20"),
        temperature=0.4,
        max_retries=3,
        rate_limiter=rate_limiter
    )
    # ----------------------------------------------------

    # Instantiate and assign services to global variables
    neo4j_service_instance = Neo4jService()
    
    mongodb_service_instance = await MongoDBSaver.from_conn_info(
        url=os.getenv("MONGO_URL"),
        db_name=os.getenv("MONGO_DB_NAME"),
        embedding_model=embedding_model # Use the shared instance
    )
    
    pinecone_service_instance = PineconeService(embedding_model=embedding_model) # Use the shared instance
    
    qg_service_instance = QuestionGenerationService(
        llm=llm_instance,
        neo4j_service=neo4j_service_instance,
        mongodb_service=mongodb_service_instance,
        similarity_embedding_model=embedding_model, # Use the shared instance
        rate_limiter=rate_limiter
    )
    
    report_service_instance = ReportGenerationService(
        mongodb_service=mongodb_service_instance,
        pinecone_service=pinecone_service_instance,
        neo4j_service=neo4j_service_instance,
        llm=llm_instance
    )
    
    chat_service_instance = await ChatService.create()
    
    print("[FastAPI DI] All services initialized successfully.")


# --- FastAPI Dependency Injector Functions ---
# These functions are now simplified to just return the pre-initialized singletons.

def get_neo4j_service() -> Neo4jService:
    if neo4j_service_instance is None:
        raise RuntimeError("Neo4jService has not been initialized.")
    return neo4j_service_instance

def get_mongodb_service() -> MongoDBSaver:
    if mongodb_service_instance is None:
        raise RuntimeError("MongoDBSaver has not been initialized.")
    return mongodb_service_instance

def get_pinecone_service() -> PineconeService:
    if pinecone_service_instance is None:
        raise RuntimeError("PineconeService has not been initialized.")
    return pinecone_service_instance

def get_question_generation_service() -> QuestionGenerationService:
    if qg_service_instance is None:
        raise RuntimeError("QuestionGenerationService has not been initialized.")
    return qg_service_instance

def get_report_generation_service() -> ReportGenerationService:
    if report_service_instance is None:
        raise RuntimeError("ReportGenerationService has not been initialized.")
    return report_service_instance

async def get_chat_service() -> ChatService:
    if chat_service_instance is None:
        raise RuntimeError("ChatService has not been initialized.")
    return chat_service_instance