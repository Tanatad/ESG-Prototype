from fastapi import APIRouter, Depends, File, UploadFile, BackgroundTasks, HTTPException
from fastapi.responses import JSONResponse
from app.services.neo4j_service import Neo4jService
from app.schemas.graph import GraphRagRequest, GraphRagResponse
from typing import List, Tuple
from io import BytesIO
import time
import traceback
from app.dependencies import get_neo4j_service, get_question_generation_service
from app.services.question_generation_service import QuestionGenerationService
from app.models.esg_question_model import ESGQuestion
import io
router = APIRouter()

@router.post("/uploadfile")
async def upload_file_and_evolve(
    files: List[UploadFile] = File(...),
    is_baseline: bool = False, # เพิ่ม parameter นี้เพื่อรับค่าจาก Streamlit
    neo4j_service: Neo4jService = Depends(get_neo4j_service),
    qg_service: QuestionGenerationService = Depends(get_question_generation_service)
):
    """
    Handles file upload, ingestion, and question evolution synchronously.
    Returns a comparison result of the question set before and after the process.
    """
    try:
        if not files:
            raise HTTPException(status_code=400, detail="No files uploaded.")
            
        file_streams = [io.BytesIO(await file.read()) for file in files]
        file_names = [file.filename for file in files]
        
        # --- Run the entire pipeline synchronously ---

        # 1. Ingest documents into Neo4j
        processed_doc_ids = await neo4j_service.flow(files=file_streams, file_names=file_names)
        
        if not processed_doc_ids:
             raise HTTPException(status_code=500, detail="Document ingestion failed.")

        # 2. Evolve questions and get the comparison result
        #    (This requires modifying QuestionGenerationService to return the result)
        comparison_result = await qg_service.evolve_and_store_questions(
            document_ids=processed_doc_ids, 
            is_baseline_upload=is_baseline
        )

        return comparison_result

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/query", response_model=GraphRagResponse)
async def query(
    request: GraphRagRequest,
    neo4j_service: Neo4jService = Depends(get_neo4j_service)
):
    print(f"[CONTROLLER LOG /query] Received request with query: '{request.query}' and top_k: {request.top_k}")
    start_req_time = time.time()
    try:
        print("[CONTROLLER LOG /query] Calling neo4j_service.get_output...")
        retrieved_data = await neo4j_service.get_output(query=request.query, k=request.top_k)

        documents_content = []
        if hasattr(retrieved_data, 'relate_documents') and retrieved_data.relate_documents:
            documents_content = [doc.page_content for doc in retrieved_data.relate_documents if hasattr(doc, 'page_content')]
        
        if not documents_content:
            return GraphRagResponse(topKDocuments="")

        concatenated_content = "\n\n---\n\n".join(documents_content)

    except Exception as e:
        print(f"[CONTROLLER ERROR /query] Error in /query endpoint: {e}")
        traceback.print_exc()
        return JSONResponse(content={"error": str(e)}, status_code=500)

    req_duration = time.time() - start_req_time
    print(f"[CONTROLLER LOG /query] Request successfully completed in {req_duration:.4f} seconds.")
    return GraphRagResponse(topKDocuments=concatenated_content)