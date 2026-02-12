"""Tests for the generation loop endpoints (app/api/v1/generate.py)."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, patch

import pytest
from httpx import AsyncClient

pytestmark = pytest.mark.asyncio


async def _upload_cad(client: AsyncClient, stl_bytes: bytes) -> str:
    """Helper: upload a CAD file and return its file_id."""
    resp = await client.post(
        "/api/v1/upload/cad",
        files={"file": ("part.stl", stl_bytes, "application/octet-stream")},
    )
    assert resp.status_code == 200
    return resp.json()["id"]


async def test_start_generation_no_session(
    client: AsyncClient, sample_stl_bytes: bytes
):
    """Starting generation without a session_id should auto-create one."""
    cad_id = await _upload_cad(client, sample_stl_bytes)

    # Patch the correction loop so it does not actually run external services
    with patch(
        "app.api.v1.generate.run_correction_loop", new_callable=AsyncMock
    ) as mock_loop:
        resp = await client.post(
            "/api/v1/generate/start",
            json={"cad_file_id": cad_id},
        )

    assert resp.status_code == 200
    data = resp.json()
    assert "session_id" in data
    assert data["status"] == "running"
    assert data["current_iteration"] == 0


async def test_start_generation_missing_cad(client: AsyncClient):
    """Starting generation with a non-existent cad_file_id should return 404."""
    resp = await client.post(
        "/api/v1/generate/start",
        json={"cad_file_id": "nonexistent_cad_id"},
    )
    assert resp.status_code == 404
    assert "CAD file not found" in resp.json()["detail"]


async def test_stop_generation(client: AsyncClient, sample_stl_bytes: bytes):
    """Starting and then stopping generation should set status to 'stopped'."""
    cad_id = await _upload_cad(client, sample_stl_bytes)

    with patch(
        "app.api.v1.generate.run_correction_loop", new_callable=AsyncMock
    ):
        start_resp = await client.post(
            "/api/v1/generate/start",
            json={"cad_file_id": cad_id},
        )

    session_id = start_resp.json()["session_id"]

    stop_resp = await client.post(f"/api/v1/generate/stop/{session_id}")
    assert stop_resp.status_code == 200
    assert stop_resp.json()["status"] == "stopped"


async def test_stop_nonexistent_session(client: AsyncClient):
    """Stopping a non-existent session should return 404."""
    resp = await client.post("/api/v1/generate/stop/nonexistent_session")
    assert resp.status_code == 404


async def test_get_status(client: AsyncClient, sample_stl_bytes: bytes):
    """Getting status of a session should return a valid LoopStatus."""
    cad_id = await _upload_cad(client, sample_stl_bytes)

    with patch(
        "app.api.v1.generate.run_correction_loop", new_callable=AsyncMock
    ):
        start_resp = await client.post(
            "/api/v1/generate/start",
            json={"cad_file_id": cad_id},
        )

    session_id = start_resp.json()["session_id"]

    status_resp = await client.get(f"/api/v1/generate/status/{session_id}")
    assert status_resp.status_code == 200
    data = status_resp.json()
    assert data["session_id"] == session_id
    assert "status" in data
    assert "current_iteration" in data
    assert "max_iterations" in data
    assert "results" in data


async def test_get_status_nonexistent(client: AsyncClient):
    """Getting status of a non-existent session should return 404."""
    resp = await client.get("/api/v1/generate/status/nonexistent")
    assert resp.status_code == 404


async def test_get_code(client: AsyncClient, sample_stl_bytes: bytes):
    """Getting code for a session should return code (null initially)."""
    cad_id = await _upload_cad(client, sample_stl_bytes)

    with patch(
        "app.api.v1.generate.run_correction_loop", new_callable=AsyncMock
    ):
        start_resp = await client.post(
            "/api/v1/generate/start",
            json={"cad_file_id": cad_id},
        )

    session_id = start_resp.json()["session_id"]

    code_resp = await client.get(f"/api/v1/generate/code/{session_id}")
    assert code_resp.status_code == 200
    data = code_resp.json()
    assert data["session_id"] == session_id
    assert "code" in data
    # Initially code is None since the loop was mocked
    assert data["code"] is None


async def test_get_code_nonexistent(client: AsyncClient):
    """Getting code for a non-existent session should return 404."""
    resp = await client.get("/api/v1/generate/code/nonexistent")
    assert resp.status_code == 404


async def test_stop_not_running_session(
    client: AsyncClient, sample_stl_bytes: bytes
):
    """Stopping a session that is not running should return 409."""
    cad_id = await _upload_cad(client, sample_stl_bytes)

    with patch(
        "app.api.v1.generate.run_correction_loop", new_callable=AsyncMock
    ):
        start_resp = await client.post(
            "/api/v1/generate/start",
            json={"cad_file_id": cad_id},
        )

    session_id = start_resp.json()["session_id"]

    # Stop it first
    await client.post(f"/api/v1/generate/stop/{session_id}")

    # Try to stop again - should be 409
    resp = await client.post(f"/api/v1/generate/stop/{session_id}")
    assert resp.status_code == 409
