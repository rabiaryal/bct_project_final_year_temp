"""Chat API Routes"""

from fastapi import APIRouter, HTTPException
from typing import Dict, Any

from app.schemas import ChatRequest, ChatResponse
from app.dialogue_manager import dialogue_manager

router = APIRouter()

@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    """Main chat endpoint"""
    try:
        # Process message through dialogue manager
        response = await dialogue_manager.process_message(request)
        return response
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chat processing error: {str(e)}")

@router.get("/chat/session/{session_id}")
async def get_session_state(session_id: str) -> Dict[str, Any]:
    """Get current session state"""
    try:
        state = await dialogue_manager.get_session_state(session_id)
        if not state:
            raise HTTPException(status_code=404, detail="Session not found")
        
        return state
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Session retrieval error: {str(e)}")

@router.delete("/chat/session/{session_id}")
async def delete_session(session_id: str) -> Dict[str, str]:
    """Delete a chat session"""
    try:
        success = await dialogue_manager.delete_session(session_id)
        if not success:
            raise HTTPException(status_code=404, detail="Session not found")
        
        return {"message": f"Session {session_id} deleted successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Session deletion error: {str(e)}")