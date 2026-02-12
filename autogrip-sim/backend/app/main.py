"""FastAPI application entry point for AutoGrip-Sim Engine."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from app.api.v1 import generate, monitor, upload
from app.config import settings
from app.models import SessionCreate, SessionResponse
from app.session_manager import session_manager

logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# Lifespan
# ------------------------------------------------------------------

UPLOAD_DIRS = [
    settings.cad_upload_path,
    settings.manual_upload_path,
    settings.results_path,
]


@asynccontextmanager
async def lifespan(app: FastAPI):
    for d in UPLOAD_DIRS:
        d.mkdir(parents=True, exist_ok=True)
        logger.info("Ensured upload directory: %s", d)
    yield


# ------------------------------------------------------------------
# App
# ------------------------------------------------------------------

app = FastAPI(
    title="AutoGrip-Sim Engine",
    version="0.1.0",
    description="Self-correcting robotic grasping code generator with Isaac Sim validation",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static files
static_dir = Path(__file__).resolve().parent.parent / "static"
if static_dir.is_dir():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

# API routers
app.include_router(upload.router, prefix="/api/v1")
app.include_router(generate.router, prefix="/api/v1")
app.include_router(monitor.router, prefix="/api/v1")


# ------------------------------------------------------------------
# Root endpoints
# ------------------------------------------------------------------

@app.get("/")
async def root():
    """Serve the frontend dashboard."""
    index = static_dir / "index.html"
    if index.is_file():
        return FileResponse(str(index))
    return JSONResponse(
        {"status": "ok", "service": "AutoGrip-Sim Engine", "version": "0.1.0"}
    )


@app.get("/health")
async def health():
    return {"status": "ok"}


# ------------------------------------------------------------------
# Session creation (convenience top-level route)
# ------------------------------------------------------------------


@app.post("/api/v1/sessions", response_model=SessionResponse)
async def create_session(body: SessionCreate) -> SessionResponse:
    """Create a new generation session linking uploaded files."""
    cad_meta = await session_manager.get_file_meta(body.cad_file_id)
    if cad_meta is None:
        raise HTTPException(
            status_code=404, detail=f"CAD file '{body.cad_file_id}' not found"
        )

    if body.manual_file_id:
        manual_meta = await session_manager.get_file_meta(body.manual_file_id)
        if manual_meta is None:
            raise HTTPException(
                status_code=404,
                detail=f"Manual file '{body.manual_file_id}' not found",
            )

    session = await session_manager.create_session(
        cad_file_id=body.cad_file_id,
        manual_file_id=body.manual_file_id,
        robot_model=body.robot_model,
    )
    return SessionResponse(
        session_id=session.session_id,
        status=session.status,
        created_at=session.created_at,
    )
