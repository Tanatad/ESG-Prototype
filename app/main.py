import os
from fastapi import FastAPI
import uvicorn
from app.routers import routers
from fastapi.middleware.cors import CORSMiddleware
from beanie import init_beanie
from app.models.clustering_model import Cluster
from motor.motor_asyncio import AsyncIOMotorClient
import os
from dotenv import load_dotenv
from contextlib import asynccontextmanager

# Load environment variables
load_dotenv()

# Retrieve environment variables
MONGODB_URL = os.getenv("MONGO_URL")
MONGO_DB_NAME = os.getenv("MONGO_DB_NAME")

origins = [
    "http://localhost",
    "http://localhost:8000",
    # Add more origins as needed
]
# Initialize MongoDB
async def init():
    client = AsyncIOMotorClient(MONGODB_URL)
    database = client[MONGO_DB_NAME]
    await init_beanie(database, document_models=[Cluster])
    
@asynccontextmanager
async def lifespan(app: FastAPI):
    await init()
    yield
    # Clean up 

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # You can also use ["*"] to allow all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods, you can restrict it to specific methods like ["GET", "POST"]
    allow_headers=["*"],  # Allows all headers, you can restrict it to specific headers
)

app.include_router(routers.router, prefix="/api/v1")

def start():
    """Launched with `poetry run start` at root level"""
    uvicorn.run("app.main:app", host="0.0.0.0", port=int(os.getenv("PORT", 5050)), reload=True)

if __name__ == "__main__":
    start()
