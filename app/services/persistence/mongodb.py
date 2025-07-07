# แก้ไขไฟล์นี้

import logging
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Dict, Optional, Sequence, Tuple, List

from langchain_core.runnables import RunnableConfig
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase
from pymongo import UpdateOne
from langgraph.checkpoint.base import (
    BaseCheckpointSaver,
    ChannelVersions,
    Checkpoint,
    CheckpointMetadata,
    CheckpointTuple,
    get_checkpoint_id,
)

from beanie import Document, init_beanie
from langchain_core.embeddings import Embeddings
from app.models.esg_question_model import ESGQuestion, RelatedSETQuestion


class MongoDBSaver(BaseCheckpointSaver):
    # ... (ส่วน __init__ ไม่ได้แก้ไข) ...
    def __init__(
        self,
        client: AsyncIOMotorClient,
        db_name: str,
        embedding_model: Optional[Embeddings] = None
    ):
        super().__init__()
        self.client = client
        self.db: AsyncIOMotorDatabase = self.client[db_name]
        self.embedding_model = embedding_model
        self.logger = logging.getLogger(__name__)
        self.checkpoint_collection = self.db["checkpoints"]
        self.checkpoint_writes_collection = self.db["checkpoint_writes"]
        self.esg_question_collection = ESGQuestion.get_motor_collection()
        self.logger.info(f"MongoDBSaver initialized for database '{db_name}'.")

    # --- FIX 1: แก้ไขเมธอดนี้ ---
    # เอา @asynccontextmanager และ try...finally ออก
    @classmethod
    async def from_conn_info(
        cls, *, url: str, db_name: str, embedding_model: Optional[Embeddings] = None
    ) -> "MongoDBSaver":
        """
        สร้าง instance ของ MongoDBSaver และคงการเชื่อมต่อไว้
        """
        client = AsyncIOMotorClient(url)
        # Beanie models ต้องถูก initialize ก่อนใช้งาน
        await init_beanie(database=client[db_name], document_models=[ESGQuestion])
        # สร้างและ return instance ไปเลย โดยไม่ปิด client
        return cls(client=client, db_name=db_name, embedding_model=embedding_model)

    async def get_all_active_questions_raw(self) -> List[Dict]:
        """
        This method uses the raw motor driver to fetch data, which is more robust
        against asyncio loop issues in frameworks like Streamlit.
        """
        self.logger.info("Fetching active questions using raw motor driver...")
        # The collection name must match the one used by Beanie for the ESGQuestion model.
        # By default, it's the name of the class itself.
        cursor = self.db["ESGQuestion"].find({"is_active": True})
        return await cursor.to_list(length=None)
    
    async def get_all_questions_raw(self) -> List[Dict]:
        """
        Fetches ALL questions (active and inactive) from the database.
        """
        self.logger.info("Fetching ALL questions using raw motor driver...")
        cursor = self.db["esg_questions_final"].find({}) # No filter
        return await cursor.to_list(length=None)

    async def get_next_question_version_for_theme(self, theme_name: str) -> int:
        """
        Custom method to find the highest version for a given theme in the 
        'esg_questions_final' collection and return the next version number.
        """
        self.logger.info(f"Checking for latest version of theme: '{theme_name}'")
        
        latest_doc = await self.esg_question_collection.find_one(
            {"theme": theme_name},
            sort=[("version", -1)]
        )

        if latest_doc:
            latest_version = latest_doc.get("version", 0)
            next_version = latest_version + 1
            self.logger.info(f"Found version {latest_version}. Next version will be {next_version}.")
            return next_version
        else:
            self.logger.info(f"Theme '{theme_name}' is new. Starting with version 1.")
            return 1

    # ... (เมธอดอื่นๆ ที่เหลือทั้งหมดเหมือนเดิม ไม่ต้องแก้ไข) ...
    async def aget_tuple(self, config: RunnableConfig) -> Optional[CheckpointTuple]:
        thread_id = config["configurable"]["thread_id"]
        checkpoint_ns = config["configurable"].get("checkpoint_ns", "")
        if checkpoint_id := get_checkpoint_id(config):
            query = {
                "thread_id": thread_id,
                "checkpoint_ns": checkpoint_ns,
                "checkpoint_id": checkpoint_id,
            }
        else:
            query = {"thread_id": thread_id, "checkpoint_ns": checkpoint_ns}

        result = self.checkpoint_collection.find(query).sort("checkpoint_id", -1).limit(1)
        async for doc in result:
            config_values = {
                "thread_id": thread_id,
                "checkpoint_ns": checkpoint_ns,
                "checkpoint_id": doc["checkpoint_id"],
            }
            checkpoint = self.serde.loads_typed((doc["type"], doc["checkpoint"]))
            
            serialized_writes = self.checkpoint_writes_collection.find(config_values)
            pending_writes = [
                (
                    write_doc["task_id"],
                    write_doc["channel"],
                    self.serde.loads_typed((write_doc["type"], write_doc["value"])),
                )
                async for write_doc in serialized_writes
            ]
            
            return CheckpointTuple(
                {"configurable": config_values},
                checkpoint,
                self.serde.loads(doc["metadata"]),
                (
                    {
                        "configurable": {
                            "thread_id": thread_id,
                            "checkpoint_ns": checkpoint_ns,
                            "checkpoint_id": doc["parent_checkpoint_id"],
                        }
                    }
                    if doc.get("parent_checkpoint_id")
                    else None
                ),
                pending_writes,
            )
        return None
    
    # ... (เมธอด alist, aput, aput_writes, delete_by_thread_id และอื่นๆ เหมือนเดิม)
    async def alist(
        self,
        config: Optional[RunnableConfig],
        *,
        filter: Optional[Dict[str, Any]] = None,
        before: Optional[RunnableConfig] = None,
        limit: Optional[int] = None,
    ) -> AsyncIterator[CheckpointTuple]:
        query = {}
        if config is not None:
            query = {
                "thread_id": config["configurable"]["thread_id"],
                "checkpoint_ns": config["configurable"].get("checkpoint_ns", ""),
            }

        if filter:
            for key, value in filter.items():
                query[f"metadata.{key}"] = value

        if before is not None:
            query["checkpoint_id"] = {"$lt": before["configurable"]["checkpoint_id"]}

        result = self.checkpoint_collection.find(query).sort("checkpoint_id", -1)
        if limit is not None:
            result = result.limit(limit)
            
        async for doc in result:
            checkpoint = self.serde.loads_typed((doc["type"], doc["checkpoint"]))
            yield CheckpointTuple(
                {
                    "configurable": {
                        "thread_id": doc["thread_id"],
                        "checkpoint_ns": doc["checkpoint_ns"],
                        "checkpoint_id": doc["checkpoint_id"],
                    }
                },
                checkpoint,
                self.serde.loads(doc["metadata"]),
                (
                    {
                        "configurable": {
                            "thread_id": doc["thread_id"],
                            "checkpoint_ns": doc["checkpoint_ns"],
                            "checkpoint_id": doc["parent_checkpoint_id"],
                        }
                    }
                    if doc.get("parent_checkpoint_id")
                    else None
                ),
            )

    async def aput(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        thread_id = config["configurable"]["thread_id"]
        checkpoint_ns = config["configurable"]["checkpoint_ns"]
        checkpoint_id = checkpoint["id"]
        type_, serialized_checkpoint = self.serde.dumps_typed(checkpoint)
        doc = {
            "parent_checkpoint_id": config["configurable"].get("checkpoint_id"),
            "type": type_,
            "checkpoint": serialized_checkpoint,
            "metadata": self.serde.dumps(metadata),
        }
        upsert_query = {
            "thread_id": thread_id,
            "checkpoint_ns": checkpoint_ns,
            "checkpoint_id": checkpoint_id,
        }
        await self.checkpoint_collection.update_one(
            upsert_query, {"$set": doc}, upsert=True
        )
        return {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": checkpoint_ns,
                "checkpoint_id": checkpoint_id,
            }
        }

    async def aput_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[Tuple[str, Any]],
        task_id: str,
    ) -> None:
        thread_id = config["configurable"]["thread_id"]
        checkpoint_ns = config["configurable"]["checkpoint_ns"]
        checkpoint_id = config["configurable"]["checkpoint_id"]
        operations = []
        for idx, (channel, value) in enumerate(writes):
            upsert_query = {
                "thread_id": thread_id,
                "checkpoint_ns": checkpoint_ns,
                "checkpoint_id": checkpoint_id,
                "task_id": task_id,
                "idx": idx,
            }
            type_, serialized_value = self.serde.dumps_typed(value)
            operations.append(
                UpdateOne(
                    upsert_query,
                    {
                        "$set": {
                            "channel": channel,
                            "type": type_,
                            "value": serialized_value,
                        }
                    },
                    upsert=True,
                )
            )
        if operations:
            await self.checkpoint_writes_collection.bulk_write(operations)

    async def delete_by_thread_id(self, thread_id: str) -> None:
        await self.checkpoint_collection.delete_many({"thread_id": thread_id})
        await self.checkpoint_writes_collection.delete_many({"thread_id": thread_id})
        self.logger.info(f"Deleted all checkpoints and writes for thread_id: {thread_id}")

    async def get_all_active_questions(self) -> List[ESGQuestion]:
        return await ESGQuestion.find(ESGQuestion.is_active == True).to_list()

    async def store_esg_question(self, esg_question: ESGQuestion):
        await esg_question.save()
        self.logger.info(f"Successfully stored/updated theme '{esg_question.theme}'.")
        
    async def deactivate_question_set_in_db(self, question_id: str):
        self.logger.info(f"Deactivating question with ID: {question_id}")
        question = await ESGQuestion.get(question_id)
        if question:
            question.is_active = False
            await question.save()
            self.logger.info(f"Successfully deactivated question ID: {question_id}")
        else:
            self.logger.warning(f"Could not find question with ID '{question_id}' to deactivate.")

    def get_set_benchmark_questions(self) -> List[Dict[str, Any]]:
        from app.data.set_benchmarks import benchmark_questions
        self.logger.info(f"Loading {len(benchmark_questions)} SET benchmark questions.")
        return benchmark_questions

    async def find_similar_questions(self, question_text: str, top_k: int = 1, similarity_threshold: float = 0.75) -> List[ESGQuestion]:
        self.logger.info(f"Finding similar questions for: '{question_text[:50]}...'")
        if not self.embedding_model or not question_text:
            self.logger.warning("Embedding model not available or empty query text.")
            return []
            
        try:
            query_embedding = await self.embedding_model.aembed_query(question_text)
            
            pipeline = [
                {
                    "$vectorSearch": {
                        "index": "main_question_embedding_index",
                        "path": "main_question_embedding",
                        "queryVector": query_embedding,
                        "numCandidates": 10,
                        "limit": top_k,
                    }
                },
                {"$match": {"score": {"$gte": similarity_threshold}}},
                {"$project": {"score": {"$meta": "vectorSearchScore"}, "document": "$$ROOT"}}
            ]
            
            results = await self.esg_question_collection.aggregate(pipeline).to_list(length=top_k)
            
            if not results:
                return []

            similar_docs = [ESGQuestion(**res['document']) for res in results]
            self.logger.info(f"Found {len(similar_docs)} similar questions passing threshold {similarity_threshold}.")
            return similar_docs
            
        except Exception as e:
            self.logger.error(f"Error during semantic search for questions: {e}")
            return []

    async def has_existing_questions(self) -> bool:
        """Checks if there are any documents in the ESGQuestion collection."""
        try:
            # ใช้ find_one() ซึ่งเร็วกว่าการ count สำหรับการเช็คว่ามีข้อมูลหรือไม่
            existing_question = await ESGQuestion.find_one()
            if existing_question:
                self.logger.info("Found existing questions in the database.")
                return True
            else:
                self.logger.info("Question database is empty.")
                return False
        except Exception as e:
            self.logger.error(f"Error checking for existing questions in MongoDB: {e}", exc_info=True)
            return False # Assume it's not empty on error to be safe

    async def update_question_set_mappings(self, question_id: str, new_mappings: List[RelatedSETQuestion]):
        self.logger.info(f"Updating SET mappings for question ID: {question_id}")
        question = await ESGQuestion.get(question_id)
        if question:
            question.related_set_questions = new_mappings
            await question.save()
            self.logger.info("Successfully updated mappings.")
        else:
            self.logger.warning(f"Could not find question with ID '{question_id}' to update mappings.")

    async def clear_all_questions(self):
        """
        Deletes all documents from the ESGQuestion collection.
        Used to ensure a clean state before a baseline run.
        """
        try:
            self.logger.warning("Clearing all documents from the ESGQuestion collection for baseline run...")
            await ESGQuestion.delete_all()
            self.logger.info("Successfully cleared the ESGQuestion collection.")
        except Exception as e:
            self.logger.error(f"Failed to clear ESGQuestion collection: {e}", exc_info=True)
            # You might want to raise the exception to stop the process
            # if a clean state is absolutely required.
            raise