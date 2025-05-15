from pydantic import BaseModel
from typing import List
from beanie import Document

class ClusterModel(BaseModel):
    id: int
    points: List[float]
    
class Cluster(Document, ClusterModel):
    pass