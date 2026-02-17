"""Code generation and self-correcting loop endpoints."""

from __future__ import annotations

import asyncio
import logging
import re
from typing import Any, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.config import settings
from app.models import LoopStatus, SimulationResult
from app.session_manager import session_manager

_SESSION_ID_RE = re.compile(r"^[a-f0-9]{32}$")


def _validate_session_id(session_id: str) -> None:
    """Validate that session_id is a valid hex UUID to prevent path traversal."""
    if not _SESSION_ID_RE.match(session_id):
        raise HTTPException(
            status_code=400, detail="Invalid session ID format"
        )


class GenerateStartRequest(BaseModel):
    """Combined session creation + generation start."""
    cad_file_id: str
    manual_file_id: str | None = None
    robot_model: str = "franka_allegro"
    max_iterations: int = 20
    success_threshold: int = 3
    session_id: str | None = None  # If provided, reuse existing session
    mode: str = "grasp_only"  # "grasp_only" or "pick_and_place"
    place_target: list[float] | None = None  # [x, y, z] for pick-and-place

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/generate", tags=["generate"])

_PHASE_RE = re.compile(
    r"\[(\d+\.\d+)s\]\s+Phase\s+\d+\s+(\w+):",
)


def _extract_replay_data(
    sim_output: dict[str, Any],
    mode: str,
    place_target: Optional[list[float]],
) -> dict[str, Any]:
    """Build a replay_data dict from simulation output for frontend 3D replay."""
    # Parse phase timestamps from logs
    phases: list[dict[str, Any]] = []
    for log_line in sim_output.get("logs", []):
        m = _PHASE_RE.search(log_line)
        if m:
            phases.append({
                "name": m.group(2).upper(),
                "timestamp": float(m.group(1)),
            })

    # If no phases parsed, synthesize from mode
    if not phases:
        if mode == "pick_and_place":
            phases = [
                {"name": "APPROACH", "timestamp": 0.3},
                {"name": "DESCEND", "timestamp": 1.0},
                {"name": "GRASP", "timestamp": 1.5},
                {"name": "LIFT", "timestamp": 2.5},
                {"name": "TRANSPORT", "timestamp": 3.5},
                {"name": "PLACE", "timestamp": 5.5},
                {"name": "RELEASE", "timestamp": 7.0},
                {"name": "RETRACT", "timestamp": 8.0},
            ]
        else:
            phases = [
                {"name": "APPROACH", "timestamp": 0.3},
                {"name": "DESCEND", "timestamp": 1.0},
                {"name": "GRASP", "timestamp": 1.5},
                {"name": "LIFT", "timestamp": 2.5},
                {"name": "HOLD", "timestamp": 3.7},
            ]

    obj_state = sim_output.get("object_final_state", sim_output.get("object_state", {}))

    return {
        "phases": phases,
        "object_trajectory": sim_output.get("object_trajectory", []),
        "contact_forces": sim_output.get("contact_forces", []),
        "joint_states": sim_output.get("joint_states", {}),
        "object_final_position": obj_state.get("position"),
        "duration": sim_output.get("duration", phases[-1]["timestamp"] + 1.0 if phases else 5.0),
        "mode": mode,
        "place_target": place_target,
    }


# ------------------------------------------------------------------
# Self-correcting loop
# ------------------------------------------------------------------


async def run_correction_loop(session_id: str) -> None:
    """Execute the generate -> simulate -> validate -> correct loop.

    The loop continues until either:
    - ``success_threshold`` consecutive successes are achieved, or
    - ``max_iterations`` is reached, or
    - the session status is set to ``stopped`` externally.
    """
    session = await session_manager.get_session(session_id)
    if session is None:
        return

    await session_manager.update_session(session_id, status="running")
    await session_manager.add_log(session_id, "INFO", "Correction loop started")

    max_iter = session.max_iterations or settings.max_loop_iterations
    success_threshold = session.success_threshold or settings.success_threshold
    mode = getattr(session, "mode", "grasp_only")
    place_target = getattr(session, "place_target", None)
    consecutive_successes = 0
    generated_code: str | None = None

    try:
        # Lazy-import core modules so that the generate router can be loaded
        # even if these haven't been installed / implemented yet.
        from app.core.llm_engine import generate_code, refine_code  # type: ignore[import-untyped]
        from app.sim_interface.runner import run_simulation, validate_result  # type: ignore[import-untyped]
    except ImportError:
        logger.error(
            "Core modules not available; falling back to stub behaviour"
        )
        await session_manager.update_session(session_id, status="failed")
        await session_manager.add_log(
            session_id,
            "ERROR",
            "Core modules (llm_engine / sim_interface) are not yet implemented",
        )
        return

    try:
        file_meta = await session_manager.get_file_meta(session.cad_file_id)
        cad_metadata: dict[str, Any] = (
            file_meta.get("cad_metadata", {}) if file_meta else {}
        )
        manual_path: str | None = None
        if session.manual_file_id:
            manual_meta = await session_manager.get_file_meta(session.manual_file_id)
            if manual_meta:
                manual_path = manual_meta.get("path")

        for iteration in range(1, max_iter + 1):
            # Check if the loop has been stopped externally
            current = await session_manager.get_session(session_id)
            if current is None or current.status == "stopped":
                await session_manager.add_log(
                    session_id, "INFO", "Loop stopped by user"
                )
                return

            await session_manager.update_session(
                session_id, current_iteration=iteration
            )
            await session_manager.add_log(
                session_id, "INFO", f"Iteration {iteration}/{max_iter}"
            )

            # Step 1: Generate or refine code
            # Only refine if the previous iteration failed; keep working code on success.
            try:
                if generated_code is None:
                    await session_manager.add_log(
                        session_id, "INFO", "Generating initial grasping code"
                    )
                    generated_code = await generate_code(
                        cad_metadata=cad_metadata,
                        robot_model=session.robot_model,
                        manual_path=manual_path,
                        mode=mode,
                        place_target=place_target,
                    )
                elif consecutive_successes > 0:
                    # Previous iteration passed — re-use the same code
                    await session_manager.add_log(
                        session_id, "INFO", "Re-running last successful code"
                    )
                else:
                    current = await session_manager.get_session(session_id)
                    last_result = current.results[-1] if current and current.results else None
                    error_log = last_result.error_log if last_result else None
                    await session_manager.add_log(
                        session_id, "INFO", "Refining code based on error feedback"
                    )
                    generated_code = await refine_code(
                        current_code=generated_code,
                        error_log=error_log or "",
                        cad_metadata=cad_metadata,
                        robot_model=session.robot_model,
                        mode=mode,
                        place_target=place_target,
                    )

                await session_manager.update_session(
                    session_id, generated_code=generated_code
                )
            except Exception as exc:
                logger.exception("LLM code generation failed at iteration %d", iteration)
                await session_manager.add_log(
                    session_id, "ERROR", f"Code generation error: {exc}"
                )
                await session_manager.update_session(session_id, status="failed")
                return

            # Step 2: Run simulation
            try:
                await session_manager.add_log(
                    session_id, "INFO", "Running simulation in Isaac Sim"
                )
                sim_output = await run_simulation(
                    code=generated_code,
                    cad_path=(file_meta or {}).get("path", ""),
                    robot_model=session.robot_model,
                    place_target=place_target,
                )
            except Exception as exc:
                logger.exception("Simulation crashed at iteration %d", iteration)
                await session_manager.add_log(
                    session_id, "ERROR", f"Simulation crash: {exc}"
                )
                result = SimulationResult(
                    iteration=iteration,
                    success=False,
                    checks={},
                    error_log=str(exc),
                )
                await session_manager.add_result(session_id, result)
                consecutive_successes = 0
                continue

            # Step 3: Validate result
            try:
                checks, error_log = await validate_result(sim_output)
            except Exception as exc:
                logger.exception("Validation failed at iteration %d", iteration)
                checks = {}
                error_log = str(exc)

            success = all(checks.values()) if checks else False
            place_accuracy_val = sim_output.get("place_accuracy") if place_target else None
            replay = _extract_replay_data(sim_output, mode, place_target)
            replay["success"] = success
            result = SimulationResult(
                iteration=iteration,
                success=success,
                checks=checks,
                error_log=error_log if not success else None,
                place_target=place_target,
                place_accuracy=place_accuracy_val,
                replay_data=replay,
            )
            await session_manager.add_result(session_id, result)

            if success:
                consecutive_successes += 1
                await session_manager.add_log(
                    session_id,
                    "INFO",
                    f"Iteration {iteration} PASSED "
                    f"({consecutive_successes}/{success_threshold} consecutive)",
                )
            else:
                consecutive_successes = 0
                await session_manager.add_log(
                    session_id,
                    "WARNING",
                    f"Iteration {iteration} FAILED: {error_log}",
                )

            # Step 4: Check termination
            if consecutive_successes >= success_threshold:
                await session_manager.update_session(session_id, status="success")
                await session_manager.add_log(
                    session_id,
                    "INFO",
                    f"Success! {success_threshold} consecutive passes reached.",
                )
                return

        # Exhausted iterations
        await session_manager.update_session(session_id, status="failed")
        await session_manager.add_log(
            session_id,
            "WARNING",
            f"Max iterations ({max_iter}) reached without sustained success.",
        )

    except asyncio.CancelledError:
        await session_manager.update_session(session_id, status="stopped")
        await session_manager.add_log(
            session_id, "INFO", "Loop cancelled externally"
        )
        logger.info("Correction loop cancelled for session %s", session_id)


# ------------------------------------------------------------------
# Endpoints
# ------------------------------------------------------------------


@router.post("/start", response_model=LoopStatus)
async def start_generation(body: GenerateStartRequest) -> LoopStatus:
    """Create a session (if needed) and start the self-correcting loop.

    The frontend sends cad_file_id, manual_file_id, robot_model directly.
    This endpoint auto-creates a session, then starts the loop.
    """
    # Validate CAD file exists
    cad_meta = await session_manager.get_file_meta(body.cad_file_id)
    if cad_meta is None:
        raise HTTPException(status_code=404, detail="CAD file not found. Upload first.")

    # Create or reuse session
    session_id = body.session_id
    if session_id:
        session = await session_manager.get_session(session_id)
        if session and session.status == "running":
            raise HTTPException(status_code=409, detail="Loop is already running")
    else:
        session = None

    if session is None:
        session = await session_manager.create_session(
            cad_file_id=body.cad_file_id,
            manual_file_id=body.manual_file_id,
            robot_model=body.robot_model,
        )
        session_id = session.session_id

    # Apply custom settings
    await session_manager.update_session(
        session_id,
        max_iterations=body.max_iterations,
        success_threshold=body.success_threshold,
        mode=body.mode,
        place_target=body.place_target,
        status="running",
        current_iteration=0,
        results=[],
        generated_code=None,
    )

    task = asyncio.create_task(run_correction_loop(session_id))
    await session_manager.update_session(session_id, task=task)

    await asyncio.sleep(0)

    status = await session_manager.get_loop_status(session_id)
    if status is None:
        raise HTTPException(status_code=500, detail="Failed to read loop status")
    return status


@router.post("/stop/{session_id}", response_model=LoopStatus)
async def stop_generation(session_id: str) -> LoopStatus:
    """Request a running loop to stop."""
    _validate_session_id(session_id)
    session = await session_manager.get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")

    if session.status != "running":
        raise HTTPException(status_code=409, detail="Loop is not running")

    await session_manager.update_session(session_id, status="stopped")
    await session_manager.add_log(session_id, "INFO", "Stop requested by user")

    if session.task and not session.task.done():
        session.task.cancel()

    status = await session_manager.get_loop_status(session_id)
    if status is None:
        raise HTTPException(status_code=500, detail="Failed to read loop status")
    return status


@router.get("/status/{session_id}", response_model=LoopStatus)
async def get_status(session_id: str) -> LoopStatus:
    """Return the current loop status."""
    _validate_session_id(session_id)
    status = await session_manager.get_loop_status(session_id)
    if status is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return status


@router.get("/code/{session_id}")
async def get_code(session_id: str) -> dict[str, str | None]:
    """Return the latest generated code for a session."""
    _validate_session_id(session_id)
    session = await session_manager.get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return {"session_id": session_id, "code": session.generated_code}
