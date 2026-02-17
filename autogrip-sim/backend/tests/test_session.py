"""Tests for the SessionManager (app/session_manager.py)."""

from __future__ import annotations

import asyncio

import pytest

from app.models import SimulationResult
from app.session_manager import SessionManager

pytestmark = pytest.mark.asyncio


async def test_create_session():
    """Creating a session should populate all required fields."""
    mgr = SessionManager()
    session = await mgr.create_session(
        cad_file_id="cad_001",
        manual_file_id="man_001",
        robot_model="franka_allegro",
    )
    assert session.session_id
    assert session.cad_file_id == "cad_001"
    assert session.manual_file_id == "man_001"
    assert session.robot_model == "franka_allegro"
    assert session.status == "created"
    assert session.created_at  # ISO timestamp string
    assert session.current_iteration == 0
    assert session.results == []
    assert session.logs == []


async def test_get_session():
    """A created session should be retrievable by its ID."""
    mgr = SessionManager()
    session = await mgr.create_session(
        cad_file_id="cad_002",
        manual_file_id=None,
        robot_model="franka",
    )
    retrieved = await mgr.get_session(session.session_id)
    assert retrieved is not None
    assert retrieved.session_id == session.session_id
    assert retrieved.robot_model == "franka"


async def test_get_session_nonexistent():
    """Getting a non-existent session should return None."""
    mgr = SessionManager()
    assert await mgr.get_session("does_not_exist") is None


async def test_update_session():
    """Updating session fields should persist the changes."""
    mgr = SessionManager()
    session = await mgr.create_session(
        cad_file_id="cad_003",
        manual_file_id=None,
        robot_model="ur5",
    )
    updated = await mgr.update_session(
        session.session_id, status="running", current_iteration=5
    )
    assert updated is not None
    assert updated.status == "running"
    assert updated.current_iteration == 5


async def test_update_nonexistent_session():
    """Updating a non-existent session should return None."""
    mgr = SessionManager()
    assert await mgr.update_session("missing", status="x") is None


async def test_add_log():
    """Adding logs should be retrievable in order."""
    mgr = SessionManager()
    session = await mgr.create_session(
        cad_file_id="cad_004",
        manual_file_id=None,
        robot_model="test_bot",
    )
    await mgr.add_log(session.session_id, "INFO", "First log")
    await mgr.add_log(session.session_id, "WARNING", "Second log", data={"key": 1})

    logs = await mgr.get_logs(session.session_id)
    assert len(logs) == 2
    assert logs[0]["level"] == "INFO"
    assert logs[0]["message"] == "First log"
    assert logs[1]["level"] == "WARNING"
    assert logs[1]["data"] == {"key": 1}
    # Each log should have a timestamp
    assert "timestamp" in logs[0]
    assert "timestamp" in logs[1]


async def test_add_log_nonexistent_session():
    """Adding a log to a non-existent session should be a no-op."""
    mgr = SessionManager()
    await mgr.add_log("missing_session", "ERROR", "should not crash")
    logs = await mgr.get_logs("missing_session")
    assert logs == []


async def test_add_result():
    """Adding SimulationResults should update loop status correctly."""
    mgr = SessionManager()
    session = await mgr.create_session(
        cad_file_id="cad_005",
        manual_file_id=None,
        robot_model="test_bot",
    )

    result1 = SimulationResult(
        iteration=1,
        success=False,
        checks={"hold_test": False, "contact_test": True},
        error_log="hold failed",
    )
    await mgr.add_result(session.session_id, result1)

    result2 = SimulationResult(
        iteration=2,
        success=True,
        checks={"hold_test": True, "contact_test": True},
    )
    await mgr.add_result(session.session_id, result2)

    status = await mgr.get_loop_status(session.session_id)
    assert status is not None
    assert status.current_iteration == 2
    assert len(status.results) == 2
    assert status.results[0].success is False
    assert status.results[1].success is True


async def test_concurrent_access():
    """Multiple concurrent create_session calls should not conflict."""
    mgr = SessionManager()

    async def create(i: int):
        return await mgr.create_session(
            cad_file_id=f"cad_{i}",
            manual_file_id=None,
            robot_model=f"bot_{i}",
        )

    sessions = await asyncio.gather(*[create(i) for i in range(20)])

    # All sessions should have unique IDs
    ids = [s.session_id for s in sessions]
    assert len(set(ids)) == 20

    # All should be retrievable
    for s in sessions:
        retrieved = await mgr.get_session(s.session_id)
        assert retrieved is not None
        assert retrieved.session_id == s.session_id


async def test_file_meta_store_and_retrieve():
    """File metadata should be storable and retrievable."""
    mgr = SessionManager()
    meta = {"filename": "test.stl", "file_type": "cad", "size_bytes": 1234}
    await mgr.store_file_meta("file_001", meta)

    retrieved = await mgr.get_file_meta("file_001")
    assert retrieved == meta


async def test_file_meta_nonexistent():
    """Getting metadata for a non-existent file should return None."""
    mgr = SessionManager()
    assert await mgr.get_file_meta("nope") is None


async def test_get_loop_status():
    """get_loop_status should return a LoopStatus model with all fields."""
    mgr = SessionManager()
    session = await mgr.create_session(
        cad_file_id="cad_006",
        manual_file_id=None,
        robot_model="test_bot",
    )
    await mgr.update_session(session.session_id, status="running")

    status = await mgr.get_loop_status(session.session_id)
    assert status is not None
    assert status.session_id == session.session_id
    assert status.status == "running"
    assert status.max_iterations == 20
    assert status.results == []
    assert status.final_code is None


async def test_get_loop_status_nonexistent():
    """get_loop_status for a non-existent session should return None."""
    mgr = SessionManager()
    assert await mgr.get_loop_status("nope") is None
