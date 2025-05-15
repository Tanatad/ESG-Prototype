from pydantic import BaseModel
from typing import List
from langchain_core.messages import AnyMessage

class ChatRagRequest(BaseModel):
    thread_id: str
    question: str
    prompt: str | None = None

class ChatRagResponse(BaseModel):
    messages: List[AnyMessage]