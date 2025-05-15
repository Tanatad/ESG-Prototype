from fastapi import APIRouter, Depends, File, UploadFile
from fastapi.responses import JSONResponse
from app.services.neo4j_service import Neo4jService
from app.schemas.graph import GraphRagRequest, GraphRagResponse
from app.dependencies import get_Neo4jService
from typing import List, Optional, IO
from io import BytesIO

router = APIRouter()

# We should send the error message to the client if occur

@router.post("/uploadfile")
async def uploadfile(
    files: List[UploadFile] = File(...),
    neo4j_service: Neo4jService = Depends(get_Neo4jService)
):
    file_streams: List[IO[bytes]] = [BytesIO(await file.read()) for file in files]
    file_names = [f"files_{i}_" + file.filename for i, file in enumerate(files)]
    
    try:
        response = await neo4j_service.flow(file_streams, file_names)
    except Exception as e:
        return JSONResponse(content={"error": f"{e}"}, status_code=500)
    
    # Possible Error
    # file not found or not a PDF
    # Translating the document failed
            
    return JSONResponse(content={"status": "Success"}, status_code=200)

@router.post("/query", response_model=GraphRagResponse)
async def query(
    request: GraphRagRequest,
    neo4j_service: Neo4jService = Depends(get_Neo4jService)
):
    try:
        # Retrieve documents based on the query using the provided top_k value.
        top_k_documents = await neo4j_service.get_relate(query=request.query, k=request.top_k)
        if not top_k_documents:
            return GraphRagRequest(topKDocuments="")
        concatenated_content = "".join(doc.page_content for doc in top_k_documents)
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
    return GraphRagResponse(topKDocuments=concatenated_content)
