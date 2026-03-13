"""Demo API-key auth — share DEMO_API_KEY with friends for the demo.

How a friend calls /chat:

    curl -X POST https://api.rabiaryal.com.np/api/v1/chat \
         -H "x-api-key: demo-secret-2026" \
         -H "Content-Type: application/json" \
         -d '{"session_id": "abc123", "message": "hello"}'

Change DEMO_API_KEY below (or set it as an env var) before the demo.
"""

import os

from fastapi import Depends, HTTPException, status
from fastapi.security import APIKeyHeader

# ── Single hardcoded keyword ───────────────────────────────────────────────────
DEMO_API_KEY = os.getenv("DEMO_API_KEY", "demo-secret-2026")

_api_key_header = APIKeyHeader(name="x-api-key", auto_error=False)


# ── Reusable dependency ────────────────────────────────────────────────────────

def verify_token(key: str | None = Depends(_api_key_header)) -> None:
    """Inject as Depends(verify_token) on any endpoint you want to protect."""
    if key != DEMO_API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing or incorrect x-api-key",
        )
