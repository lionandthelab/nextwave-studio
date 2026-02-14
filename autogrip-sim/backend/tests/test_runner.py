"""Tests for the simulation runner module (app/sim_interface/runner.py).

Covers: run_simulation, validate_result integration with connector and validator.
"""

from __future__ import annotations

import pytest
import pytest_asyncio

from app.sim_interface import runner
from app.sim_interface.connector import IsaacSimConnector


pytestmark = pytest.mark.asyncio


@pytest_asyncio.fixture(autouse=True)
async def reset_runner_singletons(sim_http_client):
    """Reset module-level singletons and inject test connector."""
    runner._connector = IsaacSimConnector(http_client=sim_http_client)
    runner._validator = None
    yield
    runner._connector = None
    runner._validator = None


class TestRunSimulation:
    """Tests for run_simulation function."""

    async def test_run_simulation_returns_dict(self):
        """run_simulation should return a dict with expected keys."""
        result = await runner.run_simulation(
            code="torque = 5.0\ngrasp_width = 0.06",
            cad_path="/tmp/test.stl",
            robot_model="unitree_h1",
        )
        assert isinstance(result, dict)
        assert "success" in result
        assert "duration" in result
        assert "logs" in result
        assert "frames" in result
        assert "object_final_state" in result
        assert "contact_forces" in result

    async def test_run_simulation_starts_context_if_needed(self):
        """Should auto-start the simulation context if not running."""
        connector = runner._get_connector()
        assert connector.context is None

        await runner.run_simulation(
            code="torque = 3.0",
            cad_path="/tmp/obj.stl",
            robot_model="franka",
        )

        assert connector.context is not None
        assert connector.context.running is True

    async def test_run_simulation_loads_robot_and_object(self):
        """Should load robot and object into the scene."""
        await runner.run_simulation(
            code="torque = 5.0",
            cad_path="/tmp/cube.stl",
            robot_model="ur5",
        )
        connector = runner._get_connector()
        assert connector.context.robot is not None
        assert connector.context.robot.model == "ur5"
        assert len(connector.context.objects) >= 1


class TestValidateResult:
    """Tests for validate_result function."""

    async def test_validate_successful_result(self):
        """A good simulation result should pass all checks."""
        sim_output = {
            "success": True,
            "duration": 6.0,
            "frames": [],
            "logs": [],
            "object_final_state": {
                "position": [0.5, 0.0, 0.3],
                "velocity": [0.0, 0.0, 0.0],
                "angular_velocity": [0.01, 0.01, 0.01],
                "contact_count": 2,
            },
            "contact_forces": [
                {"finger": "left", "force_n": 5.0},
                {"finger": "right", "force_n": 5.0},
            ],
            "joint_states": {
                "gripper_left": {"position": 0.04, "torque": 3.0},
                "gripper_right": {"position": -0.04, "torque": 3.0},
            },
        }
        checks, error_log = await runner.validate_result(sim_output)
        assert isinstance(checks, dict)
        assert all(checks.values())
        assert error_log is None

    async def test_validate_failed_result(self):
        """A dropped object should fail validation with error log."""
        sim_output = {
            "success": False,
            "duration": 6.0,
            "frames": [],
            "logs": ["ERROR: Object dropped"],
            "object_final_state": {
                "position": [0.5, 0.0, 0.0],
                "velocity": [0.0, 0.0, -1.0],
                "angular_velocity": [0.0, 0.0, 0.0],
                "contact_count": 0,
            },
            "contact_forces": [],
            "joint_states": {
                "gripper_left": {"position": 0.0, "torque": 0.0},
            },
        }
        checks, error_log = await runner.validate_result(sim_output)
        assert isinstance(checks, dict)
        assert not all(checks.values())  # At least one check should fail
        assert error_log is not None
        assert len(error_log) > 0

    async def test_validate_appends_suggestions(self):
        """Failed validation should include suggestions in error_log."""
        sim_output = {
            "success": False,
            "duration": 6.0,
            "frames": [],
            "logs": [],
            "object_final_state": {
                "position": [0.5, 0.0, 0.0],  # dropped
                "velocity": [0.0, 0.0, 0.0],
                "angular_velocity": [0.0, 0.0, 0.0],
                "contact_count": 0,
            },
            "contact_forces": [],
            "joint_states": {},
        }
        checks, error_log = await runner.validate_result(sim_output)
        assert "Suggested fixes" in error_log


class TestRunAndValidateIntegration:
    """Integration test: run_simulation then validate_result."""

    async def test_run_then_validate(self):
        """Running a simulation and validating should produce consistent results."""
        sim_output = await runner.run_simulation(
            code="torque = 8.0\ngrasp_width = 0.06",
            cad_path="/tmp/test.stl",
            robot_model="unitree_h1",
        )

        checks, error_log = await runner.validate_result(sim_output)
        assert isinstance(checks, dict)
        assert len(checks) == 5  # hold, contact, stability, force, workspace checks

        if all(checks.values()):
            assert error_log is None
        else:
            assert error_log is not None
