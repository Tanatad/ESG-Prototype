from fastapi import APIRouter, Depends, File, UploadFile
from fastapi.responses import JSONResponse
from app.services.neo4j_service import Neo4jService
from app.schemas.graph import GraphRagRequest, GraphRagResponse
from app.dependencies import get_Neo4jService
from typing import List, IO
from io import BytesIO
import time # เพิ่มเข้ามา
import traceback # เพิ่มเข้ามา

router = APIRouter()

# We should send the error message to the client if occur

@router.post("/uploadfile")
async def uploadfile(
    files: List[UploadFile] = File(...),
    neo4j_service: Neo4jService = Depends(get_Neo4jService)
):
    print(f"[CONTROLLER LOG /uploadfile] Received request with {len(files)} file(s).")
    start_req_time = time.time()

    file_streams: List[IO[bytes]] = []
    file_names: List[str] = []

    try:
        for i, file in enumerate(files):
            print(f"[CONTROLLER LOG /uploadfile] Reading file: {file.filename}")
            content = await file.read()
            file_streams.append(BytesIO(content))
            file_names.append(f"files_{i}_" + file.filename)
            print(f"[CONTROLLER LOG /uploadfile] Successfully read and prepared stream for file: {file.filename}")
    except Exception as e:
        print(f"[CONTROLLER ERROR /uploadfile] Error reading or preparing file streams: {e}")
        traceback.print_exc() # พิมพ์ traceback เต็มๆ ของ error
        req_duration = time.time() - start_req_time
        print(f"[CONTROLLER LOG /uploadfile] Request failed in {req_duration:.4f} seconds during file reading.")
        return JSONResponse(content={"error": f"Error during file reading: {str(e)}"}, status_code=500)

    print(f"[CONTROLLER LOG /uploadfile] Prepared {len(file_streams)} file streams. File names: {file_names}")
    print("[CONTROLLER LOG /uploadfile] Calling neo4j_service.flow...")

    try:
        service_response = await neo4j_service.flow(file_streams, file_names) # นี่คือจุดที่เรียก service
        print("[CONTROLLER LOG /uploadfile] neo4j_service.flow completed successfully.")
        print(f"[CONTROLLER LOG /uploadfile] Response from neo4j_service.flow: {service_response}")

    except Exception as e:
        print(f"[CONTROLLER ERROR /uploadfile] Error in neo4j_service.flow: {e}")
        traceback.print_exc() # พิมพ์ traceback เต็มๆ ของ error
        req_duration = time.time() - start_req_time
        print(f"[CONTROLLER LOG /uploadfile] Request failed in {req_duration:.4f} seconds during neo4j_service.flow.")
        return JSONResponse(content={"error": f"Error processing files with Neo4j service: {str(e)}"}, status_code=500)

    req_duration = time.time() - start_req_time
    print(f"[CONTROLLER LOG /uploadfile] Request successfully completed in {req_duration:.4f} seconds.")
    return JSONResponse(content={"status": "Success", "message": "Files processed successfully.", "service_response": service_response}, status_code=200)


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