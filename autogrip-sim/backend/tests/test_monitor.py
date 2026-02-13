"""Tests for the monitoring endpoints (app/api/v1/monitor.py).

Covers: SSE streaming, log retrieval, session validation, event generator.
"""

from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, patch

import pytest
from httpx import AsyncClient

from app.session_manager import SessionManager

pytestmark = pytest.mark.asyncio


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _create_session_with_logs(client: AsyncClient, stl_bytes: bytes) -> str:
    """Upload a CAD file, start generation, and return the session_id."""
    # Upload CAD file
    resp = await client.post(
        "/api/v1/upload/cad",
        files={"file": ("part.stl", stl_bytes, "application/octet-stream")},
    )
    assert resp.status_code == 200
    cad_id = resp.json()["id"]

    # Start generation (mocked loop)
    with patch(
        "app.api.v1.generate.run_correction_loop", new_callable=AsyncMock
    ):
        start_resp = await client.post(
            "/api/v1/generate/start",
            json={"cad_file_id": cad_id},
        )
    assert start_resp.status_code == 200
    return start_resp.json()["session_id"]


# ---------------------------------------------------------------------------
# Log endpoint tests
# ---------------------------------------------------------------------------


class TestLogEndpoint:
    """Tests for the GET /monitor/logs/{session_id} endpoint."""

    async def test_get_logs_returns_list(
        self, client: AsyncClient, sample_stl_bytes: bytes
    ):
        """Logs endpoint should return a list."""
        session_id = await _create_session_with_logs(client, sample_stl_bytes)
        resp = await client.get(f"/api/v1/monitor/logs/{session_id}")
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, list)

    async def test_get_logs_nonexistent_session(self, client: AsyncClient):
        """Logs for nonexistent session should return 404."""
        valid_id = "b" * 32
        resp = await client.get(f"/api/v1/monitor/logs/{valid_id}")
        assert resp.status_code == 404

    async def test_get_logs_invalid_session_id(self, client: AsyncClient):
        """Invalid session ID format should return 400."""
        resp = await client.get("/api/v1/monitor/logs/invalid-id!")
        assert resp.status_code == 400
        assert "Invalid session ID" in resp.json()["detail"]


# ---------------------------------------------------------------------------
# SSE stream endpoint tests
# ---------------------------------------------------------------------------


class TestStreamEndpoint:
    """Tests for the GET /monitor/stream/{session_id} SSE endpoint."""

    async def test_stream_nonexistent_session(self, client: AsyncClient):
        """Streaming a nonexistent session should return 404."""
        valid_id = "c" * 32
        resp = await client.get(f"/api/v1/monitor/stream/{valid_id}")
        assert resp.status_code == 404

    async def test_stream_invalid_session_id(self, client: AsyncClient):
        """Invalid session ID for stream should return 400."""
        resp = await client.get("/api/v1/monitor/stream/bad-session!")
        assert resp.status_code == 400


# ---------------------------------------------------------------------------
# GIF endpoint tests
# ---------------------------------------------------------------------------


class TestGifEndpoint:
    """Tests for the GET /monitor/result/{session_id}/gif endpoint."""

    async def test_gif_nonexistent_session(self, client: AsyncClient):
        """GIF for nonexistent session should return 404."""
        valid_id = "d" * 32
        resp = await client.get(f"/api/v1/monitor/result/{valid_id}/gif")
        assert resp.status_code == 404

    async def test_gif_no_file_yet(
        self, client: AsyncClient, sample_stl_bytes: bytes
    ):
        """GIF endpoint for a session without results should return 404."""
        session_id = await _create_session_with_logs(client, sample_stl_bytes)
        resp = await client.get(f"/api/v1/monitor/result/{session_id}/gif")
        assert resp.status_code == 404
        assert "not yet available" in resp.json()["detail"].lower()

    async def test_gif_invalid_session_id(self, client: AsyncClient):
        """Invalid session ID for GIF should return 400 or 404."""
        resp = await client.get("/api/v1/monitor/result/invalid!/gif")
        assert resp.status_code in (400, 404)


# ---------------------------------------------------------------------------
# Event generator unit tests
# ---------------------------------------------------------------------------


class TestEventGenerator:
    """Unit tests for the _event_generator async generator."""

    async def test_event_generator_session_not_found(self):
        """Event generator should yield error event for missing session."""
        from app.api.v1.monitor import _event_generator

        events = []
        async for event in _event_generator("nonexistent_session_id"):
            events.append(event)
            break  # Only need the first event

        assert len(events) == 1
        assert events[0]["event"] == "error"
        data = json.loads(events[0]["data"])
        assert "not found" in data["message"].lower()

    async def test_event_generator_complete_on_stopped_session(self):
        """Event generator should emit 'complete' for a stopped session."""
        from app.api.v1.monitor import _event_generator
        from app.session_manager import session_manager

        # Create a session and mark it stopped
        session = await session_manager.create_session(
            cad_file_id="cad_test",
            manual_file_id=None,
            robot_model="test_bot",
        )
        await session_manager.update_session(session.session_id, status="stopped")

        events = []
        async for event in _event_generator(session.session_id):
            events.append(event)
            if event.get("event") == "complete":
                break

        event_types = [e["event"] for e in events]
        assert "complete" in event_types

        # Verify the complete event data
        complete_event = next(e for e in events if e["event"] == "complete")
        complete_data = json.loads(complete_event["data"])
        assert complete_data["status"] == "stopped"

    async def test_event_generator_complete_on_success_session(self):
        """Event generator should emit 'complete' for a successful session."""
        from app.api.v1.monitor import _event_generator
        from app.session_manager import session_manager

        session = await session_manager.create_session(
            cad_file_id="cad_ok",
            manual_file_id=None,
            robot_model="test_bot",
        )
        await session_manager.update_session(session.session_id, status="success")

        events = []
        async for event in _event_generator(session.session_id):
            events.append(event)
            if event.get("event") == "complete":
                break

        event_types = [e["event"] for e in events]
        assert "complete" in event_types

        complete_event = next(e for e in events if e["event"] == "complete")
        complete_data = json.loads(complete_event["data"])
        assert complete_data["status"] == "success"

    async def test_event_generator_complete_on_failed_session(self):
        """Event generator should emit 'complete' for a failed session."""
        from app.api.v1.monitor import _event_generator
        from app.session_manager import session_manager

        session = await session_manager.create_session(
            cad_file_id="cad_fail",
            manual_file_id=None,
            robot_model="test_bot",
        )
        await session_manager.update_session(session.session_id, status="failed")

        events = []
        async for event in _event_generator(session.session_id):
            events.append(event)
            if event.get("event") == "complete":
                break

        complete_event = next(e for e in events if e["event"] == "complete")
        complete_data = json.loads(complete_event["data"])
        assert complete_data["status"] == "failed"

    async def test_event_generator_emits_log_events(self):
        """Event generator should emit log events for session logs."""
        from app.api.v1.monitor import _event_generator
        from app.session_manager import session_manager

        session = await session_manager.create_session(
            cad_file_id="cad_log",
            manual_file_id=None,
            robot_model="test_bot",
        )
        await session_manager.add_log(session.session_id, "INFO", "Test log message")
        # Mark as stopped so the generator will terminate
        await session_manager.update_session(session.session_id, status="stopped")

        events = []
        async for event in _event_generator(session.session_id):
            events.append(event)
            if event.get("event") == "complete":
                break

        event_types = [e["event"] for e in events]
        assert "log" in event_types

        log_event = next(e for e in events if e["event"] == "log")
        log_data = json.loads(log_event["data"])
        assert log_data["message"] == "Test log message"
        assert log_data["level"] == "INFO"


# ---------------------------------------------------------------------------
# MAX_SSE_DURATION constant
# ---------------------------------------------------------------------------


class TestSSEConstants:
    """Verify SSE configuration constants."""

    def test_max_sse_duration_defined(self):
        """MAX_SSE_DURATION should be defined and positive."""
        from app.api.v1.monitor import MAX_SSE_DURATION

        assert isinstance(MAX_SSE_DURATION, int)
        assert MAX_SSE_DURATION > 0
        assert MAX_SSE_DURATION == 3600
