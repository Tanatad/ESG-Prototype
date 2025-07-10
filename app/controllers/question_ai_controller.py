from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from typing import List, Any, Dict
import traceback
from datetime import datetime

# --- Import Beanie Document Model and Special Types ---
from app.models.esg_question_model import ESGQuestion
from beanie import PydanticObjectId

router = APIRouter(
    prefix="/question-ai",
    tags=["Question AI Management"]
)

def robust_json_serializer(data: Any) -> Any:
    """
    A robust custom serializer that recursively traverses data structures
    to convert non-serializable types like PydanticObjectId and datetime.
    """
    if isinstance(data, list):
        # If it's a list, apply the serializer to each item
        return [robust_json_serializer(item) for item in data]
    
    if isinstance(data, dict):
        # If it's a dictionary, apply the serializer to each value
        new_dict = {}
        for key, value in data.items():
            # Special handling for the '_id' key, which is often the source of the issue
            if key == "_id" and isinstance(value, PydanticObjectId):
                new_dict[key] = str(value)
            else:
                new_dict[key] = robust_json_serializer(value)
        return new_dict
        
    if isinstance(data, PydanticObjectId):
        # The core logic: convert any PydanticObjectId to a string
        return str(data)
        
    if isinstance(data, datetime):
        # Also handle datetime objects, converting them to standard ISO format
        return data.isoformat()
        
    # For all other simple data types (str, int, float, bool, None), return them as is
    return data

@router.get("/questions/active")
async def get_active_questions():
    """
    Retrieves all active ESG questions and uses a custom, robust serializer
    to guarantee a valid JSON response, bypassing FastAPI's problematic auto-serialization.
    """
    try:
        # 1. Fetch data directly using the Beanie Document model
        questions = await ESGQuestion.find(
            ESGQuestion.is_active == True
        ).sort([("category", 1), ("theme", 1)]).to_list()

        # 2. Convert each Beanie document to a basic dictionary
        question_dicts = [q.model_dump() for q in questions]

        # 3. Use our robust custom serializer to clean the data
        serializable_data = robust_json_serializer(question_dicts)
        
        # 4. Return the clean, serializable list using FastAPI's JSONResponse
        return JSONResponse(content=serializable_data)
    
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to retrieve active questions: {str(e)}")