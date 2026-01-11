"""Health Check API Routes"""

from fastapi import APIRouter
from typing import Dict, Any
from datetime import datetime

from app.schemas import HealthStatus, DetailedHealthStatus
from app.dialogue_manager import dialogue_manager

router = APIRouter()

@router.get("/health", response_model=HealthStatus)
async def health_check() -> HealthStatus:
    """Basic health check"""
    return HealthStatus(
        status="healthy",
        timestamp=datetime.now(),
        version="1.0.0"
    )

@router.get("/health/detailed", response_model=DetailedHealthStatus)
async def detailed_health_check() -> DetailedHealthStatus:
    """Detailed health check with component status"""
    
    # Get dialogue manager health
    dm_health = await dialogue_manager.health_check()
    
    # Build component status
    components = {
        "dialogue_manager": {
            "status": dm_health.get("dialogue_manager", "unknown"),
            "active_sessions": dm_health.get("active_sessions", 0)
        }
    }
    
    # Add individual component health
    for component, status in dm_health.get("components", {}).items():
        components[component] = {"status": status}
    
    # Determine overall status
    overall_status = "healthy"
    if any("error" in str(comp.get("status", "")) for comp in components.values()):
        overall_status = "unhealthy"
    elif any(comp.get("status") == "degraded" for comp in components.values()):
        overall_status = "degraded"
    
    return DetailedHealthStatus(
        status=overall_status,
        timestamp=datetime.now(),
        components=components,
        version="1.0.0"
    )