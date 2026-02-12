"""Self-correcting loop orchestrator for the grasp generation pipeline."""

from __future__ import annotations

import asyncio
import io
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import imageio.v3 as iio

from app.config import settings
from app.core.llm_engine import GraspCodeGenerator
from app.sim_interface.connector import IsaacSimConnector
from app.sim_interface.validator import GraspValidator, ValidationResult

logger = logging.getLogger(__name__)


@dataclass
class LoopResult:
    """Final result of the correction loop."""

    success: bool
    iterations_used: int
    final_code: str
    all_results: list[dict]
    gif_path: str | None = None


class CorrectionLoopController:
    """Orchestrates the generate -> simulate -> validate -> correct loop.

    The controller:
    1. Generates initial grasping code via the LLM engine.
    2. Runs the code in Isaac Sim (or MockSimulator).
    3. Validates the results against physics criteria.
    4. If validation fails, sends the error log back to the LLM for correction.
    5. Repeats until 3 consecutive successes or max iterations reached.
    """

    CONSECUTIVE_SUCCESSES_REQUIRED = 3

    def __init__(
        self,
        llm_engine: GraspCodeGenerator,
        sim_connector: IsaacSimConnector,
        validator: GraspValidator,
    ):
        self._llm = llm_engine
        self._sim = sim_connector
        self._validator = validator

    async def run(
        self,
        session_id: str,
        cad_metadata: dict,
        cad_file_path: str,
        robot_model: str,
        manual_collection_id: str,
        on_event: Callable[[dict], Any] | None = None,
    ) -> LoopResult:
        """Execute the full correction loop.

        Args:
            session_id: Unique session identifier.
            cad_metadata: Object metadata (dimensions, volume, etc.).
            cad_file_path: Path to the uploaded CAD file.
            robot_model: Robot model identifier.
            manual_collection_id: ChromaDB collection ID for the manual.
            on_event: Callback invoked at each step for SSE streaming.
                      Receives a dict with: event, iteration, data.

        Returns:
            LoopResult with final outcome.
        """
        max_iter = settings.max_loop_iterations
        all_results: list[dict] = []
        consecutive_successes = 0
        current_code = ""
        best_code = ""
        best_score = -1
        last_error_log = ""

        async def emit(event: str, iteration: int, **data: Any):
            payload = {"event": event, "iteration": iteration, "session_id": session_id, **data}
            if on_event:
                result = on_event(payload)
                if asyncio.iscoroutine(result):
                    await result

        await emit("loop_start", 0, max_iterations=max_iter, robot_model=robot_model)

        # -- Start simulation environment --
        await self._sim.start_simulation(headless=settings.isaac_sim_headless)
        await self._sim.load_scene()
        await self._sim.load_robot(robot_model)

        obj_dims = cad_metadata.get("dimensions", {})
        obj_z = obj_dims.get("z", 0.1) / 2 + 0.01
        await self._sim.load_object(cad_file_path, position=(0.5, 0.0, obj_z))

        try:
            for iteration in range(1, max_iter + 1):
                iter_start = time.monotonic()

                # -- Step 1: Generate or correct code --
                if iteration == 1:
                    await emit("generating_code", iteration)
                    current_code = self._llm.generate_initial_code(
                        cad_metadata=cad_metadata,
                        robot_model=robot_model,
                        manual_collection_id=manual_collection_id,
                    )
                else:
                    await emit("correcting_code", iteration, error_log=last_error_log)
                    current_code = self._llm.correct_code(
                        current_code=current_code,
                        error_log=last_error_log,
                        iteration=iteration,
                        cad_metadata=cad_metadata,
                    )

                await emit("code_ready", iteration, code=current_code)

                # -- Step 2: Run simulation --
                await emit("simulating", iteration)
                sim_result = await self._sim.execute_code(current_code)

                # -- Step 3: Validate results --
                await emit("validating", iteration)
                validation = self._validator.validate(sim_result)

                # Score this iteration (number of checks passed)
                passed_count = sum(1 for c in validation.checks.values() if c.passed)
                total_checks = len(validation.checks)
                score = passed_count / total_checks if total_checks > 0 else 0.0

                if score > best_score:
                    best_score = score
                    best_code = current_code

                iter_duration = time.monotonic() - iter_start

                iter_record = {
                    "iteration": iteration,
                    "success": validation.success,
                    "duration": iter_duration,
                    "sim_duration": sim_result.get("duration", 0.0),
                    "checks": {
                        name: {
                            "passed": check.passed,
                            "value": check.value,
                            "threshold": check.threshold,
                            "message": check.message,
                        }
                        for name, check in validation.checks.items()
                    },
                    "error_log": validation.error_log,
                    "suggestions": validation.suggestions,
                    "code_length": len(current_code),
                    "score": score,
                }
                all_results.append(iter_record)

                await emit(
                    "iteration_complete",
                    iteration,
                    success=validation.success,
                    checks={n: c.passed for n, c in validation.checks.items()},
                    score=score,
                    duration=iter_duration,
                )

                # -- Step 4: Check termination conditions --
                if validation.success:
                    consecutive_successes += 1
                    logger.info(
                        "Iteration %d PASSED (%d/%d consecutive)",
                        iteration,
                        consecutive_successes,
                        self.CONSECUTIVE_SUCCESSES_REQUIRED,
                    )

                    if consecutive_successes >= self.CONSECUTIVE_SUCCESSES_REQUIRED:
                        await emit(
                            "loop_success",
                            iteration,
                            message=f"Grasp succeeded {self.CONSECUTIVE_SUCCESSES_REQUIRED} consecutive times.",
                        )

                        # Generate result GIF
                        gif_path = await self._save_result_gif(session_id)

                        return LoopResult(
                            success=True,
                            iterations_used=iteration,
                            final_code=current_code,
                            all_results=all_results,
                            gif_path=gif_path,
                        )
                else:
                    consecutive_successes = 0
                    last_error_log = self._build_error_feedback(validation)
                    logger.info(
                        "Iteration %d FAILED: %s",
                        iteration,
                        validation.error_log[:100] if validation.error_log else "no details",
                    )

            # -- Max iterations reached --
            await emit(
                "loop_max_iterations",
                max_iter,
                message=f"Reached maximum {max_iter} iterations without consistent success.",
            )

            gif_path = await self._save_result_gif(session_id)

            return LoopResult(
                success=False,
                iterations_used=max_iter,
                final_code=best_code,
                all_results=all_results,
                gif_path=gif_path,
            )

        finally:
            await self._sim.stop_simulation()

    def _build_error_feedback(self, validation: ValidationResult) -> str:
        """Build a comprehensive error feedback string for the LLM."""
        lines = []

        if validation.error_log:
            lines.append(validation.error_log)

        lines.append("\n--- Validation Details ---")
        for name, check in validation.checks.items():
            status = "PASS" if check.passed else "FAIL"
            lines.append(
                f"[{status}] {name}: {check.message} "
                f"(value={check.value:.3f}, threshold={check.threshold:.3f})"
            )

        if validation.suggestions:
            lines.append("\n--- Suggestions ---")
            for i, suggestion in enumerate(validation.suggestions, 1):
                lines.append(f"{i}. {suggestion}")

        return "\n".join(lines)

    async def _save_result_gif(self, session_id: str) -> str | None:
        """Capture simulation frames and save as an animated GIF.

        Args:
            session_id: Session identifier for the output filename.

        Returns:
            Path to the saved GIF file, or None if no frames available.
        """
        frames_data = await self._sim.capture_frames()
        if not frames_data:
            return None

        results_dir = settings.results_path
        results_dir.mkdir(parents=True, exist_ok=True)
        gif_path = str(results_dir / f"{session_id}_result.gif")

        await self.create_result_gif(frames_data, gif_path)
        return gif_path

    async def create_result_gif(
        self, frames: list[bytes], output_path: str
    ):
        """Combine captured PNG frames into an animated GIF.

        Args:
            frames: List of PNG image bytes.
            output_path: Destination file path for the GIF.
        """
        if not frames:
            logger.warning("No frames to write to GIF")
            return

        images = []
        for frame_bytes in frames:
            try:
                img = iio.imread(frame_bytes, extension=".png")
                images.append(img)
            except Exception:
                # Skip unreadable frames
                continue

        if not images:
            logger.warning("No valid images decoded from frames")
            return

        # Write animated GIF at 10 fps (100ms per frame)
        iio.imwrite(
            output_path,
            images,
            extension=".gif",
            duration=100,
            loop=0,
        )
        logger.info("Saved result GIF: %s (%d frames)", output_path, len(images))
