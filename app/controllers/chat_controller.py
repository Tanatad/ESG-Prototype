from fastapi import APIRouter, Depends, HTTPException
from app.schemas.chat import ChatRagResponse, ChatRagRequest
from app.dependencies import get_chat_service
from app.services.chat_service import ChatService
from langchain_core.messages import SystemMessage, HumanMessage
from fastapi.responses import JSONResponse

router = APIRouter()

@router.post("/", response_model=ChatRagResponse)
async def chat(
    request: ChatRagRequest,
    chat_service: ChatService = Depends(get_chat_service)
):
    chat_graph = chat_service.graph
    try:
        config = {"configurable": {"thread_id": request.thread_id }}
        messages = ([SystemMessage(content=request.prompt)] if request.prompt else []) + [HumanMessage(content=request.question)]
        output = await chat_graph.ainvoke({"messages": messages}, config=config)
        return ChatRagResponse(messages=output["messages"])
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{thread_id}", response_model=ChatRagResponse)
async def read_chat_rag(
    thread_id: str,
    chat_service: ChatService = Depends(get_chat_service)
):
    chat_graph = chat_service.graph
    try:
        config = {"configurable": {"thread_id": thread_id }}
        
        messages = (await chat_graph.aget_state(config)).values.get("messages","")
        if not messages:
            raise HTTPException(status_code=404, detail="No messages found for this thread_id")
        
        return ChatRagResponse(messages=messages)
    
    except HTTPException as http_error:
        raise http_error
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/{thread_id}")
async def delete_chat_rag(
    thread_id: str,
    chat_service: ChatService = Depends(get_chat_service)
):
    try:
        await chat_service.delete_by_thread_id(thread_id)
        return JSONResponse(
            content={"detail": f"Messages for thread_id {thread_id} successfully deleted."},
            status_code=200
        )
    
    except Exception as e:
        return JSONResponse(
            content={"detail": str(e)},
            status_code=500
        )