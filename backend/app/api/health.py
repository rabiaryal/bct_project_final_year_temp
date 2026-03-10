"""Health Check API Routes"""

from fastapi import APIRouter, Request
from typing import Dict, Any
from datetime import datetime

from app.schemas import HealthStatus, DetailedHealthStatus

router = APIRouter()


def get_dialogue_manager(request: Request):
    """Get the active dialogue manager from app state"""
    if hasattr(request.app.state, 'dialogue_manager'):
        return request.app.state.dialogue_manager
    from app.dialogue_manager import dialogue_manager
    return dialogue_manager


@router.get("/health", response_model=HealthStatus)
async def health_check() -> HealthStatus:
    """Basic health check"""
    return HealthStatus(
        status="healthy",
        timestamp=datetime.now(),
        version="2.0.0"
    )


@router.get("/health/detailed", response_model=DetailedHealthStatus)
async def detailed_health_check(request: Request) -> DetailedHealthStatus:
    """Detailed health check with component status"""
    
    dm = get_dialogue_manager(request)
    dm_health = await dm.health_check()
    
    # Build component status
    components = {}
    
    if "components" in dm_health:
        for component, status in dm_health["components"].items():
            if isinstance(status, dict):
                components[component] = status
            else:
                components[component] = {"status": status}
    
    components["sessions"] = {"count": dm_health.get("sessions", dm_health.get("active_sessions", 0))}
    
    # Determine overall status
    overall_status = dm_health.get("status", "healthy")
    if any("error" in str(comp.get("status", "")) for comp in components.values()):
        overall_status = "unhealthy"
    
    return DetailedHealthStatus(
        status=overall_status,
        timestamp=datetime.now(),
        components=components,
        version="2.0.0"
    )