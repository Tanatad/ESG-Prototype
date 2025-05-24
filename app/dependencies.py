# app/dependencies.py
from fastapi import Depends, HTTPException
from typing import Optional
import os
from dotenv import load_dotenv
# from functools import lru_cache # lru_cache อาจจะไม่จำเป็นถ้าจัดการ instance แบบ global หรือผ่าน app.state

from app.services.neo4j_service import Neo4jService
from app.services.question_generation_service import QuestionGenerationService
from app.services.chat_service import ChatService
from app.services.persistence.mongodb import AsyncMongoDBSaver # Import AsyncMongoDBSaver

load_dotenv()

# --- Global instances หรือ Singleton pattern สำหรับ services (ตัวอย่าง) ---
# วิธีนี้เป็นวิธีหนึ่งในการจัดการ service instances
# หรือคุณอาจจะสร้าง instance ใหม่ทุกครั้งใน dependency function ก็ได้ (แต่ควรระวังสำหรับ connections)
# หรือใช้ app.state ใน main.py (แนะนำสำหรับ FastAPI)

_neo4j_service_instance: Optional[Neo4jService] = None
_qg_service_instance: Optional[QuestionGenerationService] = None
_chat_service_instance: Optional[ChatService] = None
_mongodb_saver_instance: Optional[AsyncMongoDBSaver] = None # เพิ่มสำหรับ memory

def get_neo4j_service() -> Neo4jService:
    global _neo4j_service_instance
    if _neo4j_service_instance is None:
        print("[DEPENDENCY LOG] Initializing Neo4jService instance.")
        _neo4j_service_instance = Neo4jService()
        if not hasattr(_neo4j_service_instance, 'llm_embbedding') or _neo4j_service_instance.llm_embbedding is None:
            print("[DEPENDENCY CRITICAL] Neo4jService.llm_embbedding was not initialized!")
    return _neo4j_service_instance

def get_question_generation_service(
    neo4j_service: Neo4jService = Depends(get_neo4j_service)
) -> QuestionGenerationService:
    global _qg_service_instance
    if _qg_service_instance is None:
        print("[DEPENDENCY LOG] Initializing QuestionGenerationService instance.")
        if not neo4j_service.llm_embbedding: # ตรวจสอบอีกครั้งเพื่อความปลอดภัย
            raise HTTPException(
                status_code=503,
                detail="Core embedding model (Cohere) for Neo4jService is not available. Question Generation Service cannot start."
            )
        # ----- บรรทัดนี้คือจุดที่ต้องแก้ไข -----
        _qg_service_instance = QuestionGenerationService(
            neo4j_service=neo4j_service,
            similarity_embedding_model=neo4j_service.llm_embbedding # <<<< ต้องมีส่วนนี้
        )
        # ------------------------------------
    return _qg_service_instance

async def get_memory() -> AsyncMongoDBSaver:
    # ... (โค้ด get_memory ของคุณ) ...
    mongo_url = os.getenv("MONGO_URL") 
    mongo_db_name = os.getenv("MONGO_DB_NAME") 
    if not mongo_url or not mongo_db_name:
        raise HTTPException(status_code=500, detail="MongoDB configuration for chat memory is missing.")
    mongodb_saver = await AsyncMongoDBSaver.from_conn_info_async(mongo_url, mongo_db_name)
    return mongodb_saver

def get_chat_service( 
    neo4j_service: Neo4jService = Depends(get_neo4j_service),
    memory: AsyncMongoDBSaver = Depends(get_memory) 
) -> ChatService:
    global _chat_service_instance
    if _chat_service_instance is None:
        print("[DEPENDENCY LOG] Initializing ChatService instance.")
        _chat_service_instance = ChatService(
            memory=memory, 
            Neo4jService=neo4j_service
            )
    return _chat_service_instance