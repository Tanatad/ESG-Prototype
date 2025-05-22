from fastapi import APIRouter, Depends, File, UploadFile, BackgroundTasks # เพิ่ม BackgroundTasks
from fastapi.responses import JSONResponse
from app.services.neo4j_service import Neo4jService
from app.schemas.graph import GraphRagRequest, GraphRagResponse
from app.dependencies import get_Neo4jService
from typing import List, IO
from io import BytesIO
import time # เพิ่มเข้ามา
import traceback # เพิ่มเข้ามา
from fastapi import BackgroundTasks
from app.dependencies import get_Neo4jService, get_question_generation_service # สมมติว่าสร้าง get_question_generation_service แล้ว
from app.services.question_generation_service import QuestionGenerationService
router = APIRouter()

# We should send the error message to the client if occur

@router.post("/uploadfile")
async def uploadfile(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(...),
    neo4j_service: Neo4jService = Depends(get_Neo4jService),
    qg_service: QuestionGenerationService = Depends(get_question_generation_service)
):
    print(f"[CONTROLLER LOG /uploadfile] Received request with {len(files)} file(s).")
    start_req_time = time.time()

    file_streams: List[IO[bytes]] = []
    file_names: List[str] = []
    service_response = {} # Initialize service_response เผื่อกรณีที่ไปไม่ถึงการเรียก service

    # ขั้นตอนที่ 1: อ่านไฟล์ทั้งหมดและเตรียม streams
    try:
        for i, file in enumerate(files):
            print(f"[CONTROLLER LOG /uploadfile] Reading file: {file.filename}")
            content = await file.read()
            file_streams.append(BytesIO(content))
            file_names.append(f"files_{i}_" + file.filename)
            print(f"[CONTROLLER LOG /uploadfile] Successfully read and prepared stream for file: {file.filename}")
    except Exception as e:
        print(f"[CONTROLLER ERROR /uploadfile] Error reading or preparing file streams: {e}")
        traceback.print_exc()
        req_duration = time.time() - start_req_time
        print(f"[CONTROLLER LOG /uploadfile] Request failed in {req_duration:.4f} seconds during file reading.")
        # คืนค่าเป็น ERROR response
        return JSONResponse(content={"error": f"Error during file reading: {str(e)}"}, status_code=500)

    print(f"[CONTROLLER LOG /uploadfile] Prepared {len(file_streams)} file streams. File names: {file_names}")

    # ขั้นตอนที่ 2: ประมวลผลไฟล์ทั้งหมดด้วย neo4j_service.flow
    try:
        print("[CONTROLLER LOG /uploadfile] Calling neo4j_service.flow...")
        service_response = await neo4j_service.flow(file_streams, file_names) # เรียกเพียงครั้งเดียวด้วย streams ทั้งหมด
        print("[CONTROLLER LOG /uploadfile] neo4j_service.flow completed successfully.")
        print(f"[CONTROLLER LOG /uploadfile] Response from neo4j_service.flow: {service_response}")

        # ขั้นตอนที่ 3: Trigger Question AI Generation ใน Background หลังจากประมวลผลไฟล์สำเร็จ
        print("[CONTROLLER LOG /uploadfile] Triggering Question AI generation in background...")
        background_tasks.add_task(qg_service.generate_and_store_questions)
        print("[CONTROLLER LOG /uploadfile] Question AI generation task added.")

    except Exception as e:
        print(f"[CONTROLLER ERROR /uploadfile] Error in neo4j_service.flow: {e}")
        traceback.print_exc()
        req_duration = time.time() - start_req_time
        print(f"[CONTROLLER LOG /uploadfile] Request failed in {req_duration:.4f} seconds during neo4j_service.flow.")
        # คืนค่าเป็น ERROR response
        return JSONResponse(content={"error": f"Error processing files with Neo4j service: {str(e)}"}, status_code=500)

    # หากทุกอย่างสำเร็จ
    req_duration = time.time() - start_req_time
    print(f"[CONTROLLER LOG /uploadfile] Request successfully completed in {req_duration:.4f} seconds.")
    return JSONResponse(content={"status": "Success", "message": "Files processed successfully and Question AI generation triggered.", "service_response": service_response}, status_code=200)

# ... (ส่วน Endpoint /query ดูเหมือนเดิมและน่าจะโอเคครับ) ...


@router.post("/query", response_model=GraphRagResponse)
async def query(
    request: GraphRagRequest,
    neo4j_service: Neo4jService = Depends(get_Neo4jService)
):
    print(f"[CONTROLLER LOG /query] Received request with query: '{request.query}' and top_k: {request.top_k}")
    start_req_time = time.time()
    try:
        print("[CONTROLLER LOG /query] Calling neo4j_service.get_relate...")
        top_k_documents = await neo4j_service.get_relate(query=request.query, k=request.top_k)

        if not top_k_documents:
            print("[CONTROLLER LOG /query] No documents found for the query.")
            req_duration = time.time() - start_req_time
            print(f"[CONTROLLER LOG /query] Request completed in {req_duration:.4f} seconds. No documents returned.")
            return GraphRagResponse(topKDocuments="")

        print(f"[CONTROLLER LOG /query] Retrieved {len(top_k_documents)} documents from neo4j_service.get_relate.")
        concatenated_content = "".join(doc.page_content for doc in top_k_documents)

    except Exception as e:
        print(f"[CONTROLLER ERROR /query] Error in /query endpoint: {e}")
        traceback.print_exc() # พิมพ์ traceback เต็มๆ ของ error
        req_duration = time.time() - start_req_time
        print(f"[CONTROLLER LOG /query] Request failed in {req_duration:.4f} seconds.")
        return JSONResponse(content={"error": str(e)}, status_code=500)

    req_duration = time.time() - start_req_time
    print(f"[CONTROLLER LOG /query] Request successfully completed in {req_duration:.4f} seconds.")
    return GraphRagResponse(topKDocuments=concatenated_content)