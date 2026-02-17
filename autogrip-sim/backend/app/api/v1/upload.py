"""File upload endpoints for CAD models and robot manuals."""

from __future__ import annotations

import logging
from pathlib import Path, PurePosixPath
from uuid import uuid4

import aiofiles
import trimesh
from fastapi import APIRouter, HTTPException, UploadFile

from app.config import settings
from app.models import CADMetadata, UploadResponse
from app.session_manager import session_manager

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/upload", tags=["upload"])

ALLOWED_CAD_EXTENSIONS = {".stl", ".obj", ".step", ".stp"}
ALLOWED_MANUAL_EXTENSIONS = {".pdf"}

ALLOWED_CAD_CONTENT_TYPES = {
    "application/octet-stream",
    "model/stl",
    "model/obj",
    "application/step",
    "application/stp",
    "application/vnd.ms-pki.stl",
}
ALLOWED_MANUAL_CONTENT_TYPES = {
    "application/pdf",
    "application/octet-stream",
}


def _sanitize_filename(filename: str) -> str:
    """Strip directory components from a filename to prevent path traversal."""
    return PurePosixPath(filename).name or "unknown"


def _validate_extension(filename: str, allowed: set[str]) -> str:
    suffix = Path(filename).suffix.lower()
    if suffix not in allowed:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file extension '{suffix}'. Allowed: {sorted(allowed)}",
        )
    return suffix


async def _save_upload(file: UploadFile, dest_dir: Path) -> tuple[str, Path, int]:
    file_id = uuid4().hex
    suffix = Path(file.filename or "unknown").suffix.lower()
    dest_path = dest_dir / f"{file_id}{suffix}"
    dest_dir.mkdir(parents=True, exist_ok=True)

    size = 0
    async with aiofiles.open(dest_path, "wb") as out:
        while chunk := await file.read(1024 * 256):
            size += len(chunk)
            if size > settings.max_upload_size_mb * 1024 * 1024:
                dest_path.unlink(missing_ok=True)
                raise HTTPException(
                    status_code=413,
                    detail=f"File exceeds {settings.max_upload_size_mb} MB limit",
                )
            await out.write(chunk)

    return file_id, dest_path, size


def _extract_cad_metadata(filepath: Path, filename: str) -> CADMetadata:
    try:
        mesh = trimesh.load(str(filepath))
        bounds = mesh.bounds  # [[min_x, min_y, min_z], [max_x, max_y, max_z]]
        dims = bounds[1] - bounds[0]
        volume = float(mesh.volume) if hasattr(mesh, "volume") else None
        com = (
            [float(v) for v in mesh.center_mass]
            if hasattr(mesh, "center_mass")
            else None
        )
        return CADMetadata(
            filename=filename,
            format=Path(filename).suffix.lstrip(".").lower(),
            dimensions={"x": float(dims[0]), "y": float(dims[1]), "z": float(dims[2])},
            volume=volume,
            center_of_mass=com,
        )
    except Exception:
        logger.warning("Could not parse CAD metadata for %s", filename, exc_info=True)
        return CADMetadata(
            filename=filename,
            format=Path(filename).suffix.lstrip(".").lower(),
            dimensions={"x": 0.0, "y": 0.0, "z": 0.0},
        )


def _validate_content_type(
    file: UploadFile, allowed: set[str], file_kind: str
) -> None:
    """Validate the MIME content type of an uploaded file."""
    content_type = (file.content_type or "").lower()
    if content_type and content_type not in allowed:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unsupported content type '{content_type}' for {file_kind} upload. "
                f"Allowed: {sorted(allowed)}"
            ),
        )


@router.post("/cad", response_model=UploadResponse)
async def upload_cad(file: UploadFile) -> UploadResponse:
    """Upload a CAD file (.stl, .obj, .step)."""
    filename = _sanitize_filename(file.filename or "unknown")
    _validate_extension(filename, ALLOWED_CAD_EXTENSIONS)
    _validate_content_type(file, ALLOWED_CAD_CONTENT_TYPES, "CAD")

    file_id, dest_path, size = await _save_upload(file, settings.cad_upload_path)

    cad_meta = _extract_cad_metadata(dest_path, filename)

    await session_manager.store_file_meta(
        file_id,
        {
            "filename": filename,
            "file_type": "cad",
            "size_bytes": size,
            "path": str(dest_path),
            "cad_metadata": cad_meta.model_dump(),
        },
    )

    logger.info(
        "CAD file uploaded: %s (%s bytes, id=%s)", filename, size, file_id
    )
    return UploadResponse(
        id=file_id, filename=filename, file_type="cad", size_bytes=size
    )


@router.post("/manual", response_model=UploadResponse)
async def upload_manual(file: UploadFile) -> UploadResponse:
    """Upload a robot manual PDF."""
    filename = _sanitize_filename(file.filename or "unknown")
    _validate_extension(filename, ALLOWED_MANUAL_EXTENSIONS)
    _validate_content_type(file, ALLOWED_MANUAL_CONTENT_TYPES, "manual")

    file_id, dest_path, size = await _save_upload(file, settings.manual_upload_path)

    await session_manager.store_file_meta(
        file_id,
        {
            "filename": filename,
            "file_type": "manual",
            "size_bytes": size,
            "path": str(dest_path),
        },
    )

    logger.info(
        "Manual uploaded: %s (%s bytes, id=%s)", filename, size, file_id
    )
    return UploadResponse(
        id=file_id, filename=filename, file_type="manual", size_bytes=size
    )


@router.get("/presets/list")
async def list_presets() -> list[dict]:
    """List available preset CAD objects for quick testing."""
    preset_dir = Path(__file__).resolve().parent.parent.parent.parent / "static" / "presets"
    if not preset_dir.exists():
        return []
    presets = []
    labels = {
        "box_6cm": "Box 6cm",
        "flat_box_8cm": "Flat Box 8cm",
        "cylinder_5x10": "Cylinder 5x10cm",
        "cylinder_4x5": "Small Cylinder 4x5cm",
    }
    for f in sorted(preset_dir.glob("*.stl")):
        stem = f.stem
        presets.append({
            "name": stem,
            "label": labels.get(stem, stem),
            "filename": f.name,
            "size_bytes": f.stat().st_size,
        })
    return presets


@router.post("/preset/{preset_name}", response_model=UploadResponse)
async def use_preset(preset_name: str) -> UploadResponse:
    """Use a preset CAD object (copies it to uploads like a normal upload)."""
    preset_dir = Path(__file__).resolve().parent.parent.parent.parent / "static" / "presets"
    preset_path = preset_dir / f"{preset_name}.stl"
    if not preset_path.exists():
        raise HTTPException(status_code=404, detail=f"Preset '{preset_name}' not found")

    file_id = uuid4().hex
    dest_dir = settings.cad_upload_path
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_path = dest_dir / f"{file_id}.stl"

    import shutil
    shutil.copy2(str(preset_path), str(dest_path))
    size = dest_path.stat().st_size
    filename = f"{preset_name}.stl"

    cad_meta = _extract_cad_metadata(dest_path, filename)
    await session_manager.store_file_meta(
        file_id,
        {
            "filename": filename,
            "file_type": "cad",
            "size_bytes": size,
            "path": str(dest_path),
            "cad_metadata": cad_meta.model_dump(),
        },
    )
    logger.info("Preset CAD loaded: %s (id=%s)", filename, file_id)
    return UploadResponse(id=file_id, filename=filename, file_type="cad", size_bytes=size)


@router.get("/{file_id}")
async def get_file_metadata(file_id: str) -> dict:
    """Return stored metadata for a previously uploaded file."""
    meta = await session_manager.get_file_meta(file_id)
    if meta is None:
        raise HTTPException(status_code=404, detail="File not found")
    return meta
