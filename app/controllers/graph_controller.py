from fastapi import APIRouter, Depends, File, UploadFile, BackgroundTasks, HTTPException
from fastapi.responses import JSONResponse
from app.services.neo4j_service import Neo4jService
from app.schemas.graph import GraphRagRequest, GraphRagResponse # สมมติว่า schema นี้ยังใช้งานอยู่
from typing import List, IO, Tuple # แก้ไข Tuple IO[bytes] เป็น Tuple[str, IO[bytes]]
from io import BytesIO
import time
import traceback
# BackgroundTasks ถูก import ซ้ำ เอาออกหนึ่งตัว
from app.dependencies import get_neo4j_service, get_question_generation_service
from app.services.question_generation_service import QuestionGenerationService

router = APIRouter()

@router.post("/uploadfile", summary="Upload multiple PDF files to build/update KG, then generate/evolve questions from KG.")
async def uploadfile_and_generate_baseline( # เปลี่ยนชื่อฟังก์ชันให้สื่อความหมายมากขึ้น
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(..., description="One or more PDF files to process into the Knowledge Graph."),
    neo4j_service: Neo4jService = Depends(get_neo4j_service),
    qg_service: QuestionGenerationService = Depends(get_question_generation_service)
):
    print(f"[CONTROLLER LOG /uploadfile] Received request with {len(files)} file(s).")
    start_req_time = time.time()

    if not files: # การตรวจสอบนี้อาจจะไม่จำเป็นถ้า File(...) ทำให้เป็น required แต่ใส่ไว้เพื่อความปลอดภัย
        req_duration = time.time() - start_req_time
        print(f"[CONTROLLER LOG /uploadfile] No files received. Request failed in {req_duration:.4f} seconds.")
        return JSONResponse(content={"error": "No files received."}, status_code=400)

    # เตรียม List ของ (filename, file_stream) สำหรับส่งให้ neo4j_service
    file_data_for_neo4j: List[Tuple[str, BytesIO]] = []
    original_filenames_for_log: List[str] = []

    # ขั้นตอนที่ 1: อ่านไฟล์ทั้งหมด
    try:
        for i, file_upload in enumerate(files):
            original_filenames_for_log.append(file_upload.filename)
            print(f"[CONTROLLER LOG /uploadfile] Reading file: {file_upload.filename}")
            content = await file_upload.read()
            # สร้างชื่อไฟล์ที่ไม่ซ้ำกันสำหรับ internal processing หากจำเป็น
            processed_filename = f"files_{i}_{file_upload.filename}" 
            file_data_for_neo4j.append((processed_filename, BytesIO(content)))
            print(f"[CONTROLLER LOG /uploadfile] Successfully read and prepared stream for file: {file_upload.filename}")
            await file_upload.close()
    except Exception as e:
        print(f"[CONTROLLER ERROR /uploadfile] Error reading or preparing file streams: {e}")
        traceback.print_exc()
        req_duration = time.time() - start_req_time
        print(f"[CONTROLLER LOG /uploadfile] Request failed in {req_duration:.4f} seconds during file reading.")
        return JSONResponse(content={"error": f"Error during file reading: {str(e)}"}, status_code=500)

    print(f"[CONTROLLER LOG /uploadfile] Prepared {len(file_data_for_neo4j)} file streams for Neo4j ingestion from files: {original_filenames_for_log}")

    # ขั้นตอนที่ 2: นำข้อมูลจากทุกไฟล์เข้า Neo4j KG
    # สมมติว่า neo4j_service.flow ถูกปรับปรุงให้รับ List[Tuple[str, BytesIO]]
    # หรือคุณอาจจะต้องปรับการเรียกให้สอดคล้องกับ neo4j_service.flow ปัจจุบันของคุณ
    neo4j_service_response = None
    try:
        if file_data_for_neo4j:
            print(f"[CONTROLLER LOG /uploadfile] Calling neo4j_service.flow to ingest data from {len(file_data_for_neo4j)} files into KG...")
            
            # แยก streams และ names ตามที่ neo4j_service.flow ของคุณคาดหวัง (ถ้ายังเป็นแบบเดิม)
            streams_only = [fs_tuple[1] for fs_tuple in file_data_for_neo4j]
            names_only = [fs_tuple[0] for fs_tuple in file_data_for_neo4j] # names_only อาจจะไม่ใช่ original filenames โดยตรง
            
            neo4j_service_response = await neo4j_service.flow(streams_only, names_only) # เรียกเพียงครั้งเดียว
            
            print("[CONTROLLER LOG /uploadfile] neo4j_service.flow for KG ingestion completed.")
            print(f"[CONTROLLER LOG /uploadfile] Response from neo4j_service.flow: {neo4j_service_response}")
        else:
            # กรณีนี้ไม่ควรเกิดขึ้นถ้า `File(...)` ทำให้ต้องมีไฟล์อย่างน้อย 1 ไฟล์
            print("[CONTROLLER LOG /uploadfile] No file streams were prepared. Skipping KG ingestion.")
            req_duration = time.time() - start_req_time
            print(f"[CONTROLLER LOG /uploadfile] Request completed in {req_duration:.4f} seconds (no files processed for KG).")
            return JSONResponse(content={"status": "Success", "message": "No files provided to process for KG."}, status_code=200)

    except Exception as e:
        print(f"[CONTROLLER ERROR /uploadfile] Error during neo4j_service.flow (KG ingestion): {e}")
        traceback.print_exc()
        req_duration = time.time() - start_req_time
        print(f"[CONTROLLER LOG /uploadfile] Request failed in {req_duration:.4f} seconds during KG ingestion.")
        return JSONResponse(content={"error": f"Error processing files for Knowledge Graph: {str(e)}"}, status_code=500)

    # ขั้นตอนที่ 3: Trigger Question AI Generation ใน Background (หลังจาก KG อัปเดตสมบูรณ์)
    # ส่ง `uploaded_file_content_bytes=None` เพื่อบอกให้ QG Service ทำงานแบบ KG-driven (Full Scan)
    print("[CONTROLLER LOG /uploadfile] Triggering Question AI baseline generation in background (KG-driven)...")
    background_tasks.add_task(
        qg_service.evolve_and_store_questions,
        uploaded_file_content_bytes=None 
    )
    print("[CONTROLLER LOG /uploadfile] Question AI baseline generation task added.")

    req_duration = time.time() - start_req_time
    print(f"[CONTROLLER LOG /uploadfile] Request successfully completed in {req_duration:.4f} seconds. KG updated from {len(file_data_for_neo4j)} files, Question AI baseline generation triggered.")
    return JSONResponse(
        content={
            "status": "Success", 
            "message": f"Successfully processed {len(file_data_for_neo4j)} files for Knowledge Graph ingestion. Question AI baseline generation has been triggered in the background.",
            "original_filenames": original_filenames_for_log,
            "neo4j_service_response": neo4j_service_response if neo4j_service_response is not None else "N/A" # ส่ง response จาก service กลับไปด้วย (ถ้ามี)
        }, 
        status_code=200
    )


@router.post("/query", response_model=GraphRagResponse)
async def query(
    request: GraphRagRequest,
    neo4j_service: Neo4jService = Depends(get_neo4j_service)
):
    print(f"[CONTROLLER LOG /query] Received request with query: '{request.query}' and top_k: {request.top_k}")
    start_req_time = time.time()
    try:
        print("[CONTROLLER LOG /query] Calling neo4j_service.get_relate...")
        # สมมติว่า get_relate คือ get_output หรือเมธอดที่คล้ายกันใน service ของคุณ
        # และคืนค่า list ของ Document objects ที่มี attribute page_content
        retrieved_data = await neo4j_service.get_output(query=request.query, k=request.top_k) # ปรับถ้าชื่อเมธอดต่างไป

        documents_content = []
        if hasattr(retrieved_data, 'relate_documents') and retrieved_data.relate_documents:
             documents_content = [doc.page_content for doc in retrieved_data.relate_documents if hasattr(doc, 'page_content')]
        
        if not documents_content:
            print("[CONTROLLER LOG /query] No relevant documents found for the query.")
            req_duration = time.time() - start_req_time
            print(f"[CONTROLLER LOG /query] Request completed in {req_duration:.4f} seconds. No documents returned.")
            return GraphRagResponse(topKDocuments="") # หรือ Response ที่เหมาะสม

        print(f"[CONTROLLER LOG /query] Retrieved {len(documents_content)} document contents.")
        concatenated_content = "\n\n---\n\n".join(documents_content) # เชื่อมด้วยตัวคั่นที่ชัดเจน

    except Exception as e:
        print(f"[CONTROLLER ERROR /query] Error in /query endpoint: {e}")
        traceback.print_exc()
        req_duration = time.time() - start_req_time
        print(f"[CONTROLLER LOG /query] Request failed in {req_duration:.4f} seconds.")
        return JSONResponse(content={"error": str(e)}, status_code=500)

    req_duration = time.time() - start_req_time
    print(f"[CONTROLLER LOG /query] Request successfully completed in {req_duration:.4f} seconds.")
    return GraphRagResponse(topKDocuments=concatenated_content)