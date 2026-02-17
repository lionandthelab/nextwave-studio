"""End-to-end integration test of the full self-correcting loop.

Tests the full pipeline: generate code -> simulate -> validate -> correct,
using the sim_server (mock mode) and mocked LLM engine.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.sim_interface.connector import IsaacSimConnector, MockSimulator
from app.sim_interface.validator import GraspValidator, ValidationResult

pytestmark = pytest.mark.asyncio


@pytest.fixture()
def connector(sim_http_client) -> IsaacSimConnector:
    """Provide a fresh IsaacSimConnector backed by the test sim_server."""
    return IsaacSimConnector(http_client=sim_http_client)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_good_code() -> str:
    """Return code that the MockSimulator considers high-quality."""
    return """\
torque = 8.0
grasp_width = 0.06
approach_height = 0.12
try:
    contact = get_contact()
    hold_phase()
    sleep(5)
except Exception:
    pass
"""


def _make_bad_code() -> str:
    """Return code that the MockSimulator considers low-quality."""
    return "pass"


# ---------------------------------------------------------------------------
# Full loop integration
# ---------------------------------------------------------------------------


class TestSelfCorrectingLoop:
    """Integration tests for the generate -> simulate -> validate -> correct pipeline."""

    async def test_full_loop_with_mock_simulator(self, connector):
        """Run the full loop with sim_server until success or max iterations."""
        validator = GraspValidator()

        await connector.start_simulation(headless=True)
        await connector.load_scene()
        await connector.load_robot("franka_allegro")
        await connector.load_object("/tmp/test.stl", position=(0.5, 0.0, 0.05))

        max_iterations = 15
        consecutive_successes = 0
        required_successes = 3
        current_code = _make_bad_code()
        all_results = []

        for iteration in range(1, max_iterations + 1):
            # Execute code in mock simulator
            sim_result = await connector.execute_code(current_code)
            assert isinstance(sim_result, dict)
            assert "success" in sim_result

            # Validate result
            validation = validator.validate(sim_result)
            assert isinstance(validation, ValidationResult)
            assert isinstance(validation.success, bool)
            assert len(validation.checks) == 5

            all_results.append({
                "iteration": iteration,
                "success": validation.success,
                "checks_passed": sum(1 for c in validation.checks.values() if c.passed),
            })

            if validation.success:
                consecutive_successes += 1
                if consecutive_successes >= required_successes:
                    break
            else:
                consecutive_successes = 0
                # Improve code for next iteration
                current_code = _make_good_code()

        await connector.stop_simulation()
        assert connector.context is None

        # Verify we got results
        assert len(all_results) > 0
        assert all(isinstance(r["success"], bool) for r in all_results)

    async def test_loop_produces_frames(self, connector):
        """Running simulation should produce capturable frames."""
        await connector.start_simulation(headless=True)
        await connector.load_scene()
        await connector.load_robot("franka_allegro")
        await connector.load_object("/tmp/test.stl")

        await connector.execute_code("torque = 5.0\ngrasp_width = 0.06")

        frames = await connector.capture_frames()
        assert isinstance(frames, list)
        assert len(frames) > 0
        # Each frame should be PNG bytes
        for frame in frames:
            assert isinstance(frame, bytes)
            assert frame[:4] == b"\x89PNG"

        await connector.stop_simulation()

    async def test_validator_results_structure(self, connector):
        """Validation results should have consistent structure across iterations."""
        validator = GraspValidator()

        await connector.start_simulation(headless=True)
        await connector.load_scene()
        await connector.load_robot("franka_allegro")
        await connector.load_object("/tmp/test.stl")

        expected_checks = {"hold_test", "contact_test", "stability_test", "force_test", "workspace_test"}

        for _ in range(5):
            sim_result = await connector.execute_code("torque = 5.0")
            validation = validator.validate(sim_result)

            assert set(validation.checks.keys()) == expected_checks

            for name, check in validation.checks.items():
                assert hasattr(check, "passed")
                assert hasattr(check, "value")
                assert hasattr(check, "threshold")
                assert hasattr(check, "message")
                assert isinstance(check.passed, bool)
                assert isinstance(check.value, float)
                assert isinstance(check.message, str)

            if not validation.success:
                assert len(validation.error_log) > 0
                assert len(validation.suggestions) > 0

        await connector.stop_simulation()


# ---------------------------------------------------------------------------
# Runner module integration
# ---------------------------------------------------------------------------


class TestRunnerIntegration:
    """Integration tests using the runner module functions."""

    async def test_run_and_validate_pipeline(self, sim_http_client):
        """run_simulation + validate_result should produce valid output."""
        from app.sim_interface import runner

        # Inject connector with test sim_server
        runner._connector = IsaacSimConnector(http_client=sim_http_client)
        runner._validator = None

        sim_output = await runner.run_simulation(
            code="torque = 5.0\ngrasp_width = 0.06",
            cad_path="/tmp/test.stl",
            robot_model="franka_allegro",
        )

        checks, error_log = await runner.validate_result(sim_output)

        assert isinstance(checks, dict)
        assert len(checks) == 5
        assert all(isinstance(v, bool) for v in checks.values())

        if all(checks.values()):
            assert error_log is None
        else:
            assert isinstance(error_log, str)
            assert len(error_log) > 0

        # Clean up
        runner._connector = None
        runner._validator = None

    async def test_multiple_iterations_consistent(self, sim_http_client):
        """Multiple run+validate cycles should produce consistent check keys."""
        from app.sim_interface import runner

        runner._connector = IsaacSimConnector(http_client=sim_http_client)
        runner._validator = None

        check_keys_set = None
        for _ in range(3):
            sim_output = await runner.run_simulation(
                code="torque = 5.0",
                cad_path="/tmp/obj.stl",
                robot_model="franka",
            )
            checks, _ = await runner.validate_result(sim_output)
            keys = set(checks.keys())
            if check_keys_set is None:
                check_keys_set = keys
            else:
                assert keys == check_keys_set

        runner._connector = None
        runner._validator = None


# ---------------------------------------------------------------------------
# Code quality affects outcomes
# ---------------------------------------------------------------------------


class TestCodeQualityAffectsOutcome:
    """Verify that better code quality correlates with better simulation outcomes."""

    def test_good_code_has_higher_success_probability(self):
        """Good code should have a higher success probability than bad code."""
        mock = MockSimulator()
        bad_quality = mock.evaluate_code_quality("pass")
        good_quality = mock.evaluate_code_quality(_make_good_code())

        bad_prob = mock.compute_success_probability(bad_quality, iteration=1)
        good_prob = mock.compute_success_probability(good_quality, iteration=1)

        assert good_prob > bad_prob

    def test_iteration_bonus_increases_probability(self):
        """Later iterations should have slightly higher probability."""
        mock = MockSimulator()
        quality = mock.evaluate_code_quality("torque = 5.0")

        prob_early = mock.compute_success_probability(quality, iteration=1)
        prob_late = mock.compute_success_probability(quality, iteration=10)

        assert prob_late > prob_early

    def test_probability_capped_at_95(self):
        """Success probability should never exceed 0.95."""
        mock = MockSimulator()
        quality = mock.evaluate_code_quality(_make_good_code())

        prob = mock.compute_success_probability(quality, iteration=100)
        assert prob <= 0.95
