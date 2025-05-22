from app.services.chat_service import ChatService
from app.services.neo4j_service import Neo4jService
from app.services.persistence.mongodb import AsyncMongoDBSaver
import os
from fastapi import Depends
from dotenv import load_dotenv
import asyncio
from fastapi import Depends # เพิ่ม Depends ถ้ายังไม่มี
from functools import lru_cache
from app.services.neo4j_service import Neo4jService
from app.services.question_generation_service import QuestionGenerationService
load_dotenv()

@lru_cache() # ใช้ lru_cache เพื่อให้ได้ instance เดียวกัน (Singleton-like)
def get_neo4j_service_dependency() -> Neo4jService:
    return Neo4jService() # หรือวิธีการสร้าง instance ที่คุณใช้อยู่

@lru_cache()
def get_question_generation_service(
    neo4j_service: Neo4jService = Depends(get_neo4j_service_dependency)
) -> QuestionGenerationService:
    return QuestionGenerationService(neo4j_service=neo4j_service)

def get_Neo4jService() ->  Neo4jService:
    return Neo4jService()

async def get_memory():
    url = os.getenv("MONGO_URL") or None
    db_name = os.getenv("MONGO_DB_NAME") or None

    # Use async with to initialize memory
    return await AsyncMongoDBSaver.from_conn_info_async(url, db_name)

async def get_chat_service(
    Neo4jService: Neo4jService = Depends(get_Neo4jService),
    memory = Depends(get_memory)
):
    return ChatService(memory, Neo4jService)
