from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import json
import asyncio
from typing import List, Dict
import sys
import os

# Add the parent directory to sys.path to import our modules
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

# CRITICAL: Set these BEFORE importing any ML libraries to prevent segfaults
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

from stand_alone import load_and_preprocess_data, CONFIG

app = FastAPI(title="College Recommendation API - Unified Pipeline", version="3.0.0")


# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],  # React dev servers
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global chatbot instance
chatbot_instance = None

class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def send_personal_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)

manager = ConnectionManager()

@app.on_event("startup")
async def startup_event():
    """Initialize the unified pipeline on server startup"""
    global chatbot_instance
    
    try:
        print("🚀 Initializing Unified College Recommendation System...")
        
        # Change to parent directory where data files are located
        parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        os.chdir(parent_dir)
        
        # Load data
        result = load_and_preprocess_data(CONFIG['data_path'])
        if len(result) == 6:
            df, scaler, encoder, item_features, query_input_cols, cat_feature_names = result
        else:
            df = result[0]
        
        # Use the unified pipeline system
        try:
            print("🔥 Loading Unified Pipeline with Hierarchical Intent Detection...")
            from app.unified_pipeline import UnifiedPipelineController
            chatbot_instance = UnifiedPipelineController(df)
            print("✅ Unified Pipeline Loaded!")
            print("   🎯 Hierarchical Intent Detection (greeting/recommendation/comparison/direct_question)")
            print("   🏷️  Transformer Entity Extraction (12 entity types)")
            print("   ⚡ FAISS Direct Answer System")
            print("   🧠 Neural Recommendation + RAG")
            print("   💾 Conversation State Manager")
        except Exception as pipeline_error:
            print(f"⚠️  Unified Pipeline error: {pipeline_error}")
            import traceback
            traceback.print_exc()
            print("   Falling back to standard chatbot...")
            # Fallback to standard chatbot
            from stand_alone import QueryParser, Retriever, Chatbot
            parser = QueryParser(df)
            retriever = Retriever(df)
            chatbot_instance = Chatbot(retriever, parser)
            print("✅ Standard chatbot initialized!")
        
    except Exception as e:
        print(f"❌ Failed to initialize chatbot: {e}")
        import traceback
        traceback.print_exc()
        chatbot_instance = None

@app.get("/")
async def root():
    return {
        "message": "College Recommendation API - Unified Pipeline", 
        "status": "running",
        "version": "3.0.0",
        "features": [
            "Hierarchical Intent Detection",
            "FAISS Direct Answers",
            "Neural Recommendations",
            "Conversation Context Memory",
            "Entity Extraction"
        ]
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "chatbot_ready": chatbot_instance is not None
    }

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    
    # Generate unique session ID for this WebSocket connection
    import uuid
    session_id = str(uuid.uuid4())
    
    # Send welcome message
    welcome_message = {
        "type": "system",
        "message": "🎓 Welcome to College Recommendation Chatbot! Ask me about colleges, courses, fees, or get recommendations.",
        "timestamp": asyncio.get_event_loop().time(),
        "session_id": session_id
    }
    await manager.send_personal_message(json.dumps(welcome_message), websocket)
    
    try:
        while True:
            # Receive message from client
            data = await websocket.receive_text()
            
            try:
                message_data = json.loads(data)
                user_message = message_data.get("message", "")
                # Allow client to pass session_id if they want to resume a session
                client_session_id = message_data.get("session_id", session_id)
                
                if not user_message.strip():
                    continue
                
                # Send typing indicator
                typing_message = {
                    "type": "typing",
                    "message": "Bot is typing...",
                    "timestamp": asyncio.get_event_loop().time()
                }
                await manager.send_personal_message(json.dumps(typing_message), websocket)
                
                # Process with chatbot
                if chatbot_instance:
                    try:
                        # Use the appropriate method based on chatbot type
                        if hasattr(chatbot_instance, 'handle_user_message'):
                            # Run in executor to prevent blocking and segfaults with ML models
                            loop = asyncio.get_event_loop()
                            result = await loop.run_in_executor(
                                None,
                                lambda: chatbot_instance.handle_user_message(user_message, session_id=client_session_id)
                            )
                            bot_response = result.get('response', 'I processed your message but got no response.')
                            
                            # Enhanced response metadata for frontend
                            response_data = {
                                "intent": result.get('full_intent', result.get('intent', 'unknown')),
                                "sub_intent": result.get('sub_intent'),
                                "confidence": result.get('intent_confidence', 0.0),
                                "pipeline": result.get('pipeline', 'unknown'),
                                "entities": result.get('entities', {}),
                                "response_type": result.get('type', 'answer'),
                                "session_id": result.get('session_id', client_session_id)
                            }
                        else:
                            # Fallback to standard chatbot
                            bot_response = chatbot_instance.chat(user_message)
                            response_data = {"response_type": "answer"}
                        
                        # Determine response type for frontend
                        response_type_raw = response_data.get("response_type", "")
                        if "recommendation" in response_type_raw:
                            response_type = "recommendations"
                        elif "comparison" in response_type_raw:
                            response_type = "comparison"
                        elif response_data.get("intent") == "greeting":
                            response_type = "greeting"
                        else:
                            response_type = "answer"
                        
                        response_message = {
                            "type": response_type,
                            "message": bot_response,
                            "timestamp": asyncio.get_event_loop().time(),
                            "user_query": user_message,
                            "metadata": response_data
                        }
                        
                    except Exception as e:
                        response_message = {
                            "type": "error",
                            "message": f"Sorry, I encountered an error: {str(e)}",
                            "timestamp": asyncio.get_event_loop().time()
                        }
                else:
                    response_message = {
                        "type": "error",
                        "message": "Sorry, the chatbot is not available right now. Please try again later.",
                        "timestamp": asyncio.get_event_loop().time()
                    }
                
                # Send response
                await manager.send_personal_message(json.dumps(response_message), websocket)
                
            except json.JSONDecodeError:
                error_message = {
                    "type": "error",
                    "message": "Invalid message format. Please send valid JSON.",
                    "timestamp": asyncio.get_event_loop().time()
                }
                await manager.send_personal_message(json.dumps(error_message), websocket)
                
    except WebSocketDisconnect:
        manager.disconnect(websocket)

# REST API endpoint for testing
@app.post("/chat")
async def chat_endpoint(message: dict):
    """REST endpoint for chat (alternative to WebSocket)"""
    if not chatbot_instance:
        return {"error": "Chatbot not initialized"}
    
    user_message = message.get("message", "")
    session_id = message.get("session_id", "rest-session")
    
    if not user_message.strip():
        return {"error": "Empty message"}
    
    try:
        # Use the appropriate method based on chatbot type
        if hasattr(chatbot_instance, 'handle_user_message'):
            # Unified pipeline with session support
            result = chatbot_instance.handle_user_message(user_message, session_id=session_id)
            return {
                "response": result.get('response', 'No response generated.'),
                "user_query": user_message,
                "status": "success",
                "intent": result.get('full_intent', result.get('intent', 'unknown')),
                "sub_intent": result.get('sub_intent'),
                "confidence": result.get('intent_confidence', 0.0),
                "entities": result.get('entities', {}),
                "pipeline": result.get('pipeline', 'unknown'),
                "session_id": result.get('session_id', session_id)
            }
        else:
            # Fallback to standard chatbot
            bot_response = chatbot_instance.chat(user_message)
            return {
                "response": bot_response,
                "user_query": user_message,
                "status": "success"
            }
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {
            "error": str(e),
            "status": "error"
        }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)