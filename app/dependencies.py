from app.services.chat_service import ChatService
from app.services.neo4j_service import Neo4jService
from app.services.persistence.mongodb import AsyncMongoDBSaver
import os
from fastapi import Depends
from dotenv import load_dotenv
import asyncio
load_dotenv()

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
