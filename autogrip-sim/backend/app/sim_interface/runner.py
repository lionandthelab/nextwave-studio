"""High-level simulation runner that bridges generate.py with connector + validator.

This module provides the simple function interfaces expected by the correction loop
in api/v1/generate.py:
    - run_simulation(code, cad_path, robot_model) -> dict
    - validate_result(sim_output) -> tuple[dict, str]
"""

from __future__ import annotations

import logging

from app.sim_interface.connector import IsaacSimConnector
from app.sim_interface.validator import GraspValidator

logger = logging.getLogger(__name__)

# Module-level singleton instances (created lazily)
_connector: IsaacSimConnector | None = None
_validator: GraspValidator | None = None


def _get_connector() -> IsaacSimConnector:
    global _connector
    if _connector is None:
        _connector = IsaacSimConnector()
    return _connector


def _get_validator() -> GraspValidator:
    global _validator
    if _validator is None:
        _validator = GraspValidator()
    return _validator


async def run_simulation(
    code: str,
    cad_path: str,
    robot_model: str,
) -> dict:
    """Run a single simulation iteration.

    Sets up the scene (if not already running), loads the robot and object,
    executes the generated grasping code, and returns raw simulation output.

    Args:
        code: Generated Python grasping code.
        cad_path: File path to the uploaded CAD file.
        robot_model: Robot model identifier (e.g. "unitree_h1").

    Returns:
        Raw simulation result dict from IsaacSimConnector.execute_code().
    """
    connector = _get_connector()

    # Ensure simulation is running
    if connector.context is None or not connector.context.running:
        await connector.start_simulation(headless=True)
        await connector.load_scene()

    # Load robot and object for this iteration
    await connector.load_robot(robot_model)
    await connector.load_object(cad_path, position=(0.5, 0.0, 0.05))

    # Execute the generated code
    result = await connector.execute_code(code)

    logger.info(
        "Simulation complete: success=%s, duration=%.2fs",
        result.get("success"),
        result.get("duration", 0),
    )
    return result


async def validate_result(sim_output: dict) -> tuple[dict[str, bool], str | None]:
    """Validate a simulation result using the GraspValidator.

    Args:
        sim_output: Raw simulation result from run_simulation().

    Returns:
        Tuple of (checks_dict, error_log).
        checks_dict maps check names to pass/fail booleans.
        error_log is a string describing failures, or None if all passed.
    """
    validator = _get_validator()
    validation = validator.validate(sim_output)

    # Convert to the simple dict format expected by generate.py
    checks = {name: check.passed for name, check in validation.checks.items()}
    error_log = validation.error_log if not validation.success else None

    # Append suggestions to error log for the LLM
    if not validation.success and validation.suggestions:
        suggestions_text = "\nSuggested fixes:\n" + "\n".join(
            f"  - {s}" for s in validation.suggestions
        )
        error_log = (error_log or "") + suggestions_text

    return checks, error_log
