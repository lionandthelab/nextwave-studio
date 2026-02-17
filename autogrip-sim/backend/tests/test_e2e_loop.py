"""End-to-end integration test for the self-correcting loop.

Tests the full flow: upload -> create session -> run correction loop
with mocked LLM but real sim_server (mock mode) and GraspValidator.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, patch

import pytest
import pytest_asyncio
from httpx import AsyncClient

from app.api.v1.generate import run_correction_loop
from app.session_manager import session_manager
from app.sim_interface.connector import IsaacSimConnector

pytestmark = pytest.mark.asyncio


@pytest_asyncio.fixture(autouse=True)
async def _inject_sim_connector(sim_http_client):
    """Ensure runner uses a connector connected to the test sim_server."""
    from app.sim_interface import runner

    runner._connector = IsaacSimConnector(http_client=sim_http_client)
    yield
    runner._connector = None


async def _upload_and_create_session(
    client: AsyncClient, stl_bytes: bytes
) -> str:
    """Upload a CAD file and create a session, return session_id."""
    upload_resp = await client.post(
        "/api/v1/upload/cad",
        files={"file": ("cube.stl", stl_bytes, "application/octet-stream")},
    )
    assert upload_resp.status_code == 200
    cad_id = upload_resp.json()["id"]

    session = await session_manager.create_session(
        cad_file_id=cad_id,
        manual_file_id=None,
        robot_model="franka_allegro",
    )
    await session_manager.update_session(
        session.session_id,
        max_iterations=5,
        success_threshold=2,
    )
    return session.session_id


class TestCorrectionLoopE2E:
    """E2E tests for run_correction_loop with mocked LLM."""

    async def test_loop_runs_to_completion(
        self, client: AsyncClient, sample_stl_bytes: bytes
    ):
        """The loop should run iterations and reach a terminal state."""
        session_id = await _upload_and_create_session(client, sample_stl_bytes)

        # Mock LLM generate/refine to return improving code
        mock_code_v1 = "torque = 3.0\ngrasp_width = 0.08"
        mock_code_v2 = (
            "torque = 8.0\ngrasp_width = 0.06\n"
            "try:\n    contact = get_contact()\n    hold_phase()\n    sleep(5)\n"
            "except:\n    pass"
        )
        call_count = 0

        async def mock_generate(**kwargs):
            return mock_code_v1

        async def mock_refine(**kwargs):
            nonlocal call_count
            call_count += 1
            return mock_code_v2

        with patch("app.core.llm_engine.generate_code", side_effect=mock_generate):
            with patch("app.core.llm_engine.refine_code", side_effect=mock_refine):
                await run_correction_loop(session_id)

        # Check that the session reached a terminal state
        session = await session_manager.get_session(session_id)
        assert session is not None
        assert session.status in ("success", "failed")
        assert session.current_iteration > 0
        assert len(session.results) > 0

    async def test_loop_respects_max_iterations(
        self, client: AsyncClient, sample_stl_bytes: bytes
    ):
        """The loop should not exceed max_iterations."""
        session_id = await _upload_and_create_session(client, sample_stl_bytes)

        # Set low max iterations
        await session_manager.update_session(session_id, max_iterations=3)

        # Mock LLM to return bad code that will likely fail
        async def mock_generate(**kwargs):
            return "pass"

        async def mock_refine(**kwargs):
            return "pass"

        with patch("app.core.llm_engine.generate_code", side_effect=mock_generate):
            with patch("app.core.llm_engine.refine_code", side_effect=mock_refine):
                await run_correction_loop(session_id)

        session = await session_manager.get_session(session_id)
        assert session is not None
        assert session.current_iteration <= 3

    async def test_loop_stops_on_external_stop(
        self, client: AsyncClient, sample_stl_bytes: bytes
    ):
        """Setting status to 'stopped' should terminate the loop."""
        session_id = await _upload_and_create_session(client, sample_stl_bytes)

        stop_after_iteration = 1

        original_generate_code = None

        async def mock_generate(**kwargs):
            return "torque = 5.0\ngrasp_width = 0.06"

        async def mock_refine(**kwargs):
            # Stop the session after the first iteration's refine
            await session_manager.update_session(session_id, status="stopped")
            return "torque = 5.0"

        with patch("app.core.llm_engine.generate_code", side_effect=mock_generate):
            with patch("app.core.llm_engine.refine_code", side_effect=mock_refine):
                await run_correction_loop(session_id)

        session = await session_manager.get_session(session_id)
        assert session is not None
        assert session.status == "stopped"

    async def test_loop_handles_llm_error(
        self, client: AsyncClient, sample_stl_bytes: bytes
    ):
        """If LLM raises an exception, the loop should fail gracefully."""
        session_id = await _upload_and_create_session(client, sample_stl_bytes)

        async def mock_generate(**kwargs):
            raise RuntimeError("LLM API connection failed")

        with patch("app.core.llm_engine.generate_code", side_effect=mock_generate):
            with patch("app.core.llm_engine.refine_code", new_callable=AsyncMock):
                await run_correction_loop(session_id)

        session = await session_manager.get_session(session_id)
        assert session is not None
        assert session.status == "failed"

        logs = await session_manager.get_logs(session_id)
        error_logs = [l for l in logs if l["level"] == "ERROR"]
        assert len(error_logs) > 0

    async def test_loop_generates_results_with_checks(
        self, client: AsyncClient, sample_stl_bytes: bytes
    ):
        """Each iteration should produce a SimulationResult with checks dict."""
        session_id = await _upload_and_create_session(client, sample_stl_bytes)
        await session_manager.update_session(session_id, max_iterations=2)

        async def mock_generate(**kwargs):
            return "torque = 5.0\ngrasp_width = 0.06"

        async def mock_refine(**kwargs):
            return "torque = 8.0\ngrasp_width = 0.06"

        with patch("app.core.llm_engine.generate_code", side_effect=mock_generate):
            with patch("app.core.llm_engine.refine_code", side_effect=mock_refine):
                await run_correction_loop(session_id)

        session = await session_manager.get_session(session_id)
        assert len(session.results) > 0

        for result in session.results:
            assert isinstance(result.checks, dict)
            assert isinstance(result.success, bool)
            assert result.iteration > 0


class TestCorrectionLoopCancellation:
    """Tests for asyncio.CancelledError handling (H3 fix)."""

    async def test_cancelled_error_sets_stopped(
        self, client: AsyncClient, sample_stl_bytes: bytes
    ):
        """CancelledError should set session status to 'stopped'."""
        session_id = await _upload_and_create_session(client, sample_stl_bytes)

        async def mock_generate(**kwargs):
            # Simulate cancellation during code generation
            raise asyncio.CancelledError()

        with patch("app.core.llm_engine.generate_code", side_effect=mock_generate):
            with patch("app.core.llm_engine.refine_code", new_callable=AsyncMock):
                await run_correction_loop(session_id)

        session = await session_manager.get_session(session_id)
        assert session is not None
        assert session.status == "stopped"
