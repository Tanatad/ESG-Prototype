# file: app/dependencies.py

import os
import asyncio
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
from llama_index.core.settings import Settings
from llama_index.llms.gemini import Gemini
from  langchain_google_genai import ChatGoogleGenerativeAI
import motor.motor_asyncio
from beanie import init_beanie
from app.models.esg_question_model import ESGQuestion
from app.models.clustering_model import Cluster

load_dotenv()

# --- Singleton Instances to hold created services ---
_neo4j_service: Optional[Neo4jService] = None
_mongodb_saver: Optional[MongoDBSaver] = None
_qg_service: Optional[QuestionGenerationService] = None
_chat_service: Optional[ChatService] = None

# This is a central, one-time setup for async components
async def initialize_global_services():
    """
    An async function to initialize all services that require async setup (like DB connections).
    This should be called from the FastAPI lifespan event.
    """
    global _neo4j_service, _mongodb_saver, _qg_service, _chat_service
    
    # Check if already initialized to prevent re-running
    if _neo4j_service is not None:
        return

    print("[FastAPI DI] Initializing all services...")

    # 1. Configure global LLM setting for LlamaIndex
    Settings.llm = Gemini(model="models/gemini-2.5-flash-preview-05-20", api_key=os.getenv("GOOGLE_API_KEY"))
    
    # 2. Initialize components
    embedding_model = CohereEmbeddings(model='embed-v4.0', cohere_api_key=os.getenv("COHERE_API_KEY"))
    rate_limiter = RateLimiter(requests_per_minute=int(os.getenv("MAX_RPM", "1000")))
    
    # 3. Initialize Neo4j Service (it's synchronous)
    _neo4j_service = Neo4jService()

    # 4. Initialize Database and Beanie
    mongo_uri = os.getenv("MONGO_URL")
    mongo_db_name = os.getenv("MONGO_DB_NAME")
    if not mongo_uri or not mongo_db_name:
        raise ValueError("MONGO_URL and MONGO_DB_NAME must be set in .env file")
    
    client = motor.motor_asyncio.AsyncIOMotorClient(mongo_uri)
    database = client[mongo_db_name]
    await init_beanie(database=database, document_models=[ESGQuestion, Cluster])
    
    _mongodb_saver = MongoDBSaver(client=client, db_name=mongo_db_name, embedding_model=embedding_model)

    # 5. Instantiate services with their dependencies
    llm_instance = ChatGoogleGenerativeAI(
        model=os.getenv("CHAT_MODEL", "gemini-2.5-flash-preview-05-20"),
        max_retries=10,
        rate_limiter=rate_limiter
    )

    # 2. แก้ไขการสร้าง QuestionGenerationService โดยส่ง `llm=llm_instance` เข้าไป
    _qg_service = QuestionGenerationService(
        llm=llm_instance,
        neo4j_service=_neo4j_service,
        mongodb_service=_mongodb_saver,
        similarity_embedding_model=embedding_model,
        rate_limiter=rate_limiter
    )
    
    _chat_service = await ChatService.create()
    
    print("[FastAPI DI] All services initialized successfully.")


# --- FastAPI Dependency Injectors ---
# These functions are now simple and just return the pre-initialized singletons.

def get_neo4j_service() -> Neo4jService:
    if _neo4j_service is None:
        raise RuntimeError("Services not initialized. Ensure lifespan event is configured in main.py.")
    return _neo4j_service

def get_question_generation_service() -> QuestionGenerationService:
    if _qg_service is None:
        raise RuntimeError("Services not initialized. Ensure lifespan event is configured in main.py.")
    return _qg_service

def get_chat_service() -> ChatService:
    if _chat_service is None:
        raise RuntimeError("Services not initialized. Ensure lifespan event is configured in main.py.")
    return _chat_service