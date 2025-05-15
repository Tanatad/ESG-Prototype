from pydantic import BaseModel
from typing import List
from langchain_core.messages import AnyMessage

class GraphRagRequest(BaseModel):
    query: str
    top_k: int | None = None    

class GraphRagResponse(BaseModel):
    topKDocuments: str