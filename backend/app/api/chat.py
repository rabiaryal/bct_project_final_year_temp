"""Chat API Routes"""

from fastapi import APIRouter, Depends, HTTPException, Request
from typing import Dict, Any

from app.schemas import ChatRequest, ChatResponse
from app.api.auth import verify_token

router = APIRouter()


def get_dialogue_manager(request: Request):
    """Get the active dialogue manager from app state"""
    if hasattr(request.app.state, 'dialogue_manager'):
        return request.app.state.dialogue_manager
    # Fallback to legacy manager
    from app.dialogue_manager import dialogue_manager
    return dialogue_manager


@router.post("/chat", response_model=ChatResponse)
async def chat(
    request: Request,
    chat_request: ChatRequest,
    _: None = Depends(verify_token),
) -> ChatResponse:
    """Main chat endpoint"""
    try:
        dm = get_dialogue_manager(request)
        response = await dm.process_message(chat_request)
        return response
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chat processing error: {str(e)}")


@router.get("/chat/session/{session_id}")
async def get_session_state(request: Request, session_id: str) -> Dict[str, Any]:
    """Get current session state"""
    try:
        dm = get_dialogue_manager(request)
        debug = dm.get_session_debug(session_id)
        if "error" in debug:
            raise HTTPException(status_code=404, detail="Session not found")
        return debug
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Session retrieval error: {str(e)}")


@router.delete("/chat/session/{session_id}")
async def delete_session(request: Request, session_id: str) -> Dict[str, str]:
    """Delete a chat session"""
    try:
        dm = get_dialogue_manager(request)
        if session_id not in dm.slot_manager.contexts:
            raise HTTPException(status_code=404, detail="Session not found")
        dm.slot_manager.clear_context(session_id)
        return {"message": f"Session {session_id} deleted successfully"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Session deletion error: {str(e)}")