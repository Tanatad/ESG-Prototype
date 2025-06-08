# app/controllers/question_ai_controller.py

from fastapi import APIRouter, Depends, HTTPException, Query, UploadFile, File, Form
from typing import List
import traceback
from app.models.esg_question_model import ESGQuestion
from app.services.question_generation_service import QuestionGenerationService, GeneratedQuestion
from app.services.neo4j_service import Neo4jService
from app.dependencies import get_neo4j_service, get_question_generation_service

router = APIRouter(
    prefix="/question-ai",
    tags=["Question AI Management"]
)

@router.post("/process-and-evolve", response_model=List[GeneratedQuestion])
async def process_documents_and_evolve_questions(
    files: List[UploadFile] = File(..., description="List of PDF files to process and drive question evolution."),
    is_baseline_upload: bool = Form(False, description="Set to true for the initial upload to establish baseline questions without comparing them."),
    qg_service: QuestionGenerationService = Depends(get_question_generation_service),
    neo4j_service: Neo4jService = Depends(get_neo4j_service)
):
    """
    **Primary endpoint for Use Case 3.**
    Processes one or more documents into the Knowledge Graph and then triggers
    the question (AI) evolution process.

    - **For initial setup (Host/Admin):** Upload your 26+ standard files and set `is_baseline_upload` to `true`.
      This will generate the initial set of questions (v1) without trying to compare/deactivate them.
    - **For user updates:** Upload the new user document with `is_baseline_upload` set to `false` (default).
      This will ingest the document and then run the evolution logic to compare the new findings
      against the existing baseline, potentially creating new question versions (v2) or adding new themes.
    """
    if not files:
        raise HTTPException(status_code=400, detail="No files were uploaded.")

    print(f"[CONTROLLER INFO] Starting processing for {len(files)} files. Baseline mode: {is_baseline_upload}")

    # Step 1: Process all uploaded documents into the Knowledge Graph
    try:
        for file in files:
            print(f"[CONTROLLER INFO] Ingesting file into KG: {file.filename}")
            # This assumes your Neo4j service has a method to handle the file pipeline
            # (extract, chunk, embed, store nodes/relationships).
            # Adjust the method call as per your neo4j_service implementation.
            await neo4j_service.run_document_pipeline(file)
        
        print("[CONTROLLER INFO] All files ingested into Knowledge Graph successfully.")

    except Exception as e:
        print(f"[CONTROLLER ERROR] Failed during Knowledge Graph ingestion: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error processing documents into KG: {str(e)}")

    # Step 2: Trigger question evolution logic
    try:
        print(f"[CONTROLLER INFO] Triggering question evolution. is_baseline_upload={is_baseline_upload}")
        # The `is_baseline_upload` flag is passed to the service.
        # The service will use this flag to decide whether to run the comparison logic.
        generated_questions = await qg_service.evolve_and_store_questions(
            is_baseline_upload=is_baseline_upload
        )
        print("[CONTROLLER INFO] Question evolution process completed.")
        return generated_questions

    except Exception as e:
        print(f"[CONTROLLER ERROR] Failed during question evolution: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to evolve questions: {str(e)}")


@router.get("/active-questions", response_model=List[ESGQuestion])
async def get_all_active_esg_questions(
    limit: int = Query(default=50, ge=1, le=200),
    skip: int = Query(default=0, ge=0)
):
    """
    Retrieve a list of currently active ESG questions (Question AI),
    sorted by the most recently generated/updated version.
    """
    try:
        questions = await ESGQuestion.find(
            ESGQuestion.is_active == True
        ).sort(-ESGQuestion.updated_at, -ESGQuestion.version).skip(skip).limit(limit).to_list()

        return questions if questions else []
    except Exception as e:
        print(f"[CONTROLLER ERROR /active-questions] Error retrieving ESG questions: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve ESG questions: {str(e)}")