import os
from dotenv import load_dotenv
from fastapi import Depends
from typing import Optional

# --- Import Service Classes ---
from app.services.neo4j_service import Neo4jService
from app.services.question_generation_service import QuestionGenerationService
from app.services.chat_service import ChatService
from app.services.persistence.mongodb import MongoDBSaver
from app.services.rate_limit import RateLimiter

# --- Import Initializer Classes ---
from langchain_cohere import CohereEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

# --- Singleton Instances to hold created services ---
# These variables will be populated ONCE during application startup.
neo4j_service_instance: Optional[Neo4jService] = None
mongodb_service_instance: Optional[MongoDBSaver] = None
qg_service_instance: Optional[QuestionGenerationService] = None
chat_service_instance: Optional[ChatService] = None

# --- Central Initializer (Called from FastAPI Lifespan) ---
async def initialize_global_services():
    """
    Initializes all shared services when the FastAPI app starts.
    This ensures all database connections and models are ready.
    """
    global neo4j_service_instance, mongodb_service_instance, qg_service_instance, chat_service_instance
    
    # Prevents re-initialization on hot-reloads
    if neo4j_service_instance is not None:
        print("[FastAPI DI] Services already initialized. Skipping.")
        return

    print("[FastAPI DI] Initializing all global services...")

    # 1. Initialize components needed by services
    embedding_model = CohereEmbeddings(model='embed-v4.0', cohere_api_key=os.getenv("COHERE_API_KEY"))
    rate_limiter = RateLimiter(requests_per_minute=int(os.getenv("REQUESTS_PER_MINUTE", "60")))
    
    llm_instance = ChatGoogleGenerativeAI(
        model=os.getenv("CHAT_MODEL", "gemini-2.5-flash-preview-05-20"),
        temperature=0.4,
        max_retries=3,
        rate_limiter=rate_limiter
    )

    # 2. Instantiate and assign services to global variables
    # These are created in order of dependency
    
    neo4j_service_instance = Neo4jService()
    
    mongodb_service_instance = await MongoDBSaver.from_conn_info(
        url=os.getenv("MONGO_URL"),
        db_name=os.getenv("MONGO_DB_NAME"),
        embedding_model=embedding_model
    )
    
    qg_service_instance = QuestionGenerationService(
        llm=llm_instance,
        neo4j_service=neo4j_service_instance,
        mongodb_service=mongodb_service_instance,
        similarity_embedding_model=embedding_model,
        rate_limiter=rate_limiter
    )
    
    chat_service_instance = await ChatService.create()
    
    print("[FastAPI DI] All services initialized successfully.")


# --- FastAPI Dependency Injector Functions ---
# These functions simply return the pre-initialized instances.

def get_neo4j_service() -> Neo4jService:
    """Dependency injector that provides the singleton Neo4jService instance."""
    if neo4j_service_instance is None:
        raise RuntimeError("Neo4jService has not been initialized. Check FastAPI startup events.")
    return neo4j_service_instance

def get_mongodb_service() -> MongoDBSaver:
    """Dependency injector that provides the singleton MongoDBSaver instance."""
    if mongodb_service_instance is None:
        raise RuntimeError("MongoDBSaver has not been initialized. Check FastAPI startup events.")
    return mongodb_service_instance

def get_question_generation_service() -> QuestionGenerationService:
    """Dependency injector that provides the singleton QuestionGenerationService instance."""
    if qg_service_instance is None:
        raise RuntimeError("QuestionGenerationService has not been initialized. Check FastAPI startup events.")
    return qg_service_instance

async def get_chat_service() -> ChatService:
    """Dependency injector that provides the singleton ChatService instance."""
    if chat_service_instance is None:
        raise RuntimeError("ChatService has not been initialized. Check FastAPI startup events.")
    return chat_service_instance