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

router = APIRouter()

@router.post("/uploadfile", status_code=202)
async def upload_file(
    files: List[UploadFile],
    background_tasks: BackgroundTasks,
    neo4j_service: Neo4jService = Depends(get_neo4j_service),
    qg_service: QuestionGenerationService = Depends(get_question_generation_service)
):
    """
    This endpoint ingests a list of documents into the knowledge graph
    and triggers the question evolution process as a background task.
    """
    try:
        print(f"[CONTROLLER LOG /uploadfile] Received {len(files)} file(s).")
        
        # เตรียมไฟล์สำหรับส่งให้ service
        file_streams = [file.file for file in files]
        file_names = [file.filename for file in files]
        
        print("[CONTROLLER LOG /uploadfile] Calling neo4j_service.flow to ingest data...")
        
        # --- CORRECTED PART ---
        # รับค่า document_ids ที่ประมวลผลเสร็จแล้วกลับมา
        processed_doc_ids = await neo4j_service.flow(files=file_streams, file_names=file_names)
        
        print(f"[CONTROLLER LOG /uploadfile] Ingestion complete. Triggering background question evolution for doc_ids: {processed_doc_ids}")
        
        # ส่ง document_ids เข้าไปใน background task
        background_tasks.add_task(
            qg_service.evolve_and_store_questions,
            document_ids=processed_doc_ids,
            is_baseline_upload=True # สมมติว่าการอัปโหลดผ่าน Endpoint นี้เป็น Baseline เสมอ
        )
        # --- END CORRECTED PART ---

        return {"message": f"Successfully started processing for {len(files)} files. Question evolution is running in the background."}

    except Exception as e:
        print(f"[CONTROLLER ERROR /uploadfile] Error during file upload processing: {e}")
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