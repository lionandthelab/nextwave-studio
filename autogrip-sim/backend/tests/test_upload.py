"""Tests for the file upload endpoints (app/api/v1/upload.py)."""

from __future__ import annotations

import pytest
import pytest_asyncio
from httpx import AsyncClient


pytestmark = pytest.mark.asyncio


async def test_upload_cad_stl(client: AsyncClient, sample_stl_bytes: bytes):
    """Upload a valid .stl file and verify the response schema."""
    resp = await client.post(
        "/api/v1/upload/cad",
        files={"file": ("cube.stl", sample_stl_bytes, "application/octet-stream")},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "id" in data
    assert data["filename"] == "cube.stl"
    assert data["file_type"] == "cad"
    assert data["size_bytes"] == len(sample_stl_bytes)


async def test_upload_cad_invalid_extension(client: AsyncClient):
    """Uploading a .txt file to the CAD endpoint should return 400."""
    resp = await client.post(
        "/api/v1/upload/cad",
        files={"file": ("readme.txt", b"hello", "text/plain")},
    )
    assert resp.status_code == 400
    assert "Unsupported file extension" in resp.json()["detail"]


async def test_upload_manual_pdf(client: AsyncClient, sample_pdf_bytes: bytes):
    """Upload a valid .pdf file to the manual endpoint."""
    resp = await client.post(
        "/api/v1/upload/manual",
        files={"file": ("robot_manual.pdf", sample_pdf_bytes, "application/pdf")},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["filename"] == "robot_manual.pdf"
    assert data["file_type"] == "manual"
    assert data["size_bytes"] == len(sample_pdf_bytes)


async def test_upload_manual_invalid(client: AsyncClient):
    """Uploading a non-PDF file to the manual endpoint should return 400."""
    resp = await client.post(
        "/api/v1/upload/manual",
        files={"file": ("notes.docx", b"fake-docx", "application/octet-stream")},
    )
    assert resp.status_code == 400
    assert "Unsupported file extension" in resp.json()["detail"]


async def test_get_file_metadata(client: AsyncClient, sample_stl_bytes: bytes):
    """Upload a file then GET its metadata by ID."""
    upload_resp = await client.post(
        "/api/v1/upload/cad",
        files={"file": ("part.stl", sample_stl_bytes, "application/octet-stream")},
    )
    file_id = upload_resp.json()["id"]

    meta_resp = await client.get(f"/api/v1/upload/{file_id}")
    assert meta_resp.status_code == 200
    meta = meta_resp.json()
    assert meta["filename"] == "part.stl"
    assert meta["file_type"] == "cad"
    assert meta["size_bytes"] == len(sample_stl_bytes)
    assert "cad_metadata" in meta


async def test_get_nonexistent_file(client: AsyncClient):
    """GET metadata for an unknown file ID should return 404."""
    resp = await client.get("/api/v1/upload/nonexistent_id_12345")
    assert resp.status_code == 404
    assert "not found" in resp.json()["detail"].lower()
