from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from typing import List, Any, Dict
import traceback
from datetime import datetime

# --- Import Beanie Document Model ---
from app.models.esg_question_model import ESGQuestion
from beanie import PydanticObjectId

router = APIRouter(
    prefix="/question-ai",
    tags=["Question AI Management"]
)

def custom_serializer(data: Any) -> Any:
    """
    A robust custom serializer to handle non-serializable types
    like PydanticObjectId and datetime.
    """
    if isinstance(data, list):
        # If the data is a list, process each item in the list
        return [custom_serializer(item) for item in data]
    
    if isinstance(data, dict):
        # If the data is a dictionary, process each key-value pair
        new_dict = {}
        for key, value in data.items():
            new_dict[key] = custom_serializer(value)
        return new_dict
        
    if isinstance(data, PydanticObjectId):
        # This is the core logic: convert PydanticObjectId to string
        return str(data)
        
    if isinstance(data, datetime):
        # Also handle datetime objects, converting them to standard ISO format string
        return data.isoformat()
        
    # For all other data types (str, int, float, bool, None), return as is
    return data

@router.get("/questions/active")
async def get_active_questions():
    """
    Retrieves all active ESG questions and uses a custom serializer
    to guarantee a valid JSON response.
    """
    try:
        # 1. Fetch data directly using the Beanie Document model
        questions = await ESGQuestion.find(
            ESGQuestion.is_active == True
        ).sort([("category", 1), ("theme", 1)]).to_list()

        # 2. Convert each Beanie document to a dictionary
        question_dicts = [q.model_dump() for q in questions]

        # 3. Use the robust custom serializer to clean the data
        serializable_data = custom_serializer(question_dicts)
        
        # 4. Return the clean, serializable list using JSONResponse
        return JSONResponse(content=serializable_data)
    
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to retrieve active questions: {str(e)}")