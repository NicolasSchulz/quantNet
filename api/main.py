from __future__ import annotations

import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from api.routers.model import router as model_router
    from api.routers.performance import router as performance_router
    from api.routers.trades import router as trades_router
    from api.services.model_service import list_models
    from api.services.trade_service import load_trades
except ImportError:
    from routers.model import router as model_router
    from routers.performance import router as performance_router
    from routers.trades import router as trades_router
    from services.model_service import list_models
    from services.trade_service import load_trades

app = FastAPI(title="AlgoTrader API", version="0.1.0")

cors_origin = os.getenv("CORS_ORIGIN", "http://localhost:5173")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[cors_origin],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(trades_router, prefix="/api")
app.include_router(performance_router, prefix="/api")
app.include_router(model_router, prefix="/api")


@app.on_event("startup")
def startup_checks() -> None:
    app.state.platform_status = {
        "trades_loaded": bool(load_trades()),
        "model_registry_loaded": bool(list_models()["models"]),
        "live_mode": os.getenv("LIVE_MODE", "false").lower() == "true",
    }


@app.get("/health")
def health() -> dict[str, object]:
    return {
        "status": "ok",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "platform": getattr(app.state, "platform_status", {}),
    }
