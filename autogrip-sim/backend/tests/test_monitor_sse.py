"""Tests for the SSE monitoring endpoints (app/api/v1/monitor.py).

Covers: SSE event streaming, log retrieval, and GIF endpoint.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, patch

import pytest
from httpx import AsyncClient

pytestmark = pytest.mark.asyncio


async def _create_session_with_cad(
    client: AsyncClient, stl_bytes: bytes
) -> str:
    """Helper: upload a CAD file, start generation, return session_id."""
    upload_resp = await client.post(
        "/api/v1/upload/cad",
        files={"file": ("part.stl", stl_bytes, "application/octet-stream")},
    )
    cad_id = upload_resp.json()["id"]

    with patch(
        "app.api.v1.generate.run_correction_loop", new_callable=AsyncMock
    ):
        start_resp = await client.post(
            "/api/v1/generate/start",
            json={"cad_file_id": cad_id},
        )
    return start_resp.json()["session_id"]


class TestMonitorLogs:
    """Tests for the /monitor/logs endpoint."""

    async def test_get_logs_empty(
        self, client: AsyncClient, sample_stl_bytes: bytes
    ):
        """A new session should have no logs (or only startup logs)."""
        session_id = await _create_session_with_cad(client, sample_stl_bytes)

        resp = await client.get(f"/api/v1/monitor/logs/{session_id}")
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)

    async def test_get_logs_nonexistent_session(self, client: AsyncClient):
        """Requesting logs for a non-existent (but valid format) session should return 404."""
        valid_hex = "b" * 32
        resp = await client.get(f"/api/v1/monitor/logs/{valid_hex}")
        assert resp.status_code == 404

    async def test_get_logs_invalid_session_id(self, client: AsyncClient):
        """Requesting logs with an invalid session ID format should return 400."""
        resp = await client.get("/api/v1/monitor/logs/not-valid-hex!")
        assert resp.status_code == 400


class TestMonitorGif:
    """Tests for the /monitor/result/{session_id}/gif endpoint."""

    async def test_gif_not_available(
        self, client: AsyncClient, sample_stl_bytes: bytes
    ):
        """Requesting a GIF before it's generated should return 404."""
        session_id = await _create_session_with_cad(client, sample_stl_bytes)

        resp = await client.get(
            f"/api/v1/monitor/result/{session_id}/gif"
        )
        assert resp.status_code == 404
        assert "not yet available" in resp.json()["detail"].lower()

    async def test_gif_nonexistent_session(self, client: AsyncClient):
        """Requesting a GIF for a non-existent session should return 404."""
        valid_hex = "c" * 32
        resp = await client.get(f"/api/v1/monitor/result/{valid_hex}/gif")
        assert resp.status_code == 404


class TestMonitorStream:
    """Tests for the /monitor/stream/{session_id} SSE endpoint."""

    async def test_stream_nonexistent_session(self, client: AsyncClient):
        """Streaming a non-existent (but valid format) session should return 404."""
        valid_hex = "d" * 32
        resp = await client.get(f"/api/v1/monitor/stream/{valid_hex}")
        assert resp.status_code == 404

    async def test_stream_invalid_session_id(self, client: AsyncClient):
        """Streaming with an invalid session ID should return 400."""
        resp = await client.get("/api/v1/monitor/stream/invalid-id!")
        assert resp.status_code == 400
