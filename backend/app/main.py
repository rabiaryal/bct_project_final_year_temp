"""FastAPI Main Application"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import uvicorn
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env.mongodb'))

from app.api import chat, health
from app.dialogue_manager import dialogue_manager
from app.utils.logger import setup_logger
from app.utils.config import config

# Setup logging
logger = setup_logger("main")

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    logger.info("🚀 Starting dialogue system...")
    
    try:
        await dialogue_manager.initialize()
        app.state.dialogue_manager = dialogue_manager
        logger.info("✅ Dialogue system initialized successfully")
        
        yield
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize dialogue system: {e}")
        raise
    
    finally:
        logger.info("🛑 Shutting down dialogue system...")
        await dialogue_manager.shutdown()
        logger.info("Shutdown complete")

# Create FastAPI app
app = FastAPI(
    title="College Recommendation Dialogue System",
    description="AI-powered dialogue system for college information and recommendations",
    version="1.0.0",
    lifespan=lifespan
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.api.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(chat.router, prefix="/api/v1", tags=["chat"])
app.include_router(health.router, prefix="/api/v1", tags=["health"])

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "College Recommendation Dialogue System API",
        "version": "1.0.0",
        "docs": "/docs"
    }

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler"""
    logger.error(f"Unhandled exception: {exc}")
    return {
        "error": "Internal server error",
        "message": "An unexpected error occurred"
    }

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host=config.api.host,
        port=config.api.port,
        reload=config.api.reload
    )