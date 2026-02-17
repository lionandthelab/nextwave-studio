"""End-to-end pick-and-place pipeline test.

Exercises the full mock pipeline:
  _generate_default_code → MockSimulationEngine.execute → GraspValidator.validate

Runs up to 100 iterations and asserts that 3 consecutive successes (all 8 validation
checks passing) are achieved.
"""

import sys
from pathlib import Path

import pytest

# Make app and sim_server importable
_backend_dir = str(Path(__file__).resolve().parent.parent)
if _backend_dir not in sys.path:
    sys.path.insert(0, _backend_dir)

_sim_scripts_dir = str(Path(__file__).resolve().parent.parent / "docker" / "sim_scripts")
if _sim_scripts_dir not in sys.path:
    sys.path.insert(0, _sim_scripts_dir)

from app.core.llm_engine import _generate_default_code
from app.sim_interface.validator import GraspValidator
from sim_server import MockSimulationEngine

# Mug dimensions from test_mug.stl analysis
MUG_METADATA = {
    "dimensions": {"x": 0.11, "y": 0.08, "z": 0.09},
    "center_of_mass": [0.015, 0.0, 0.045],
}

PLACE_TARGET = [0.5, 0.4, 0.05]
ROBOT_MODEL = "franka_allegro"
MAX_ITERATIONS = 100
SUCCESS_THRESHOLD = 3  # 3 consecutive passes


class TestPnPEndToEnd:
    """Full pipeline test: code generation → mock simulation → validation."""

    def test_pnp_succeeds_within_100_iterations(self):
        """The pick-and-place pipeline must achieve 3 consecutive successes
        within 100 iterations using the default generated code + MockSimulationEngine."""
        code = _generate_default_code(
            cad_metadata=MUG_METADATA,
            robot_model=ROBOT_MODEL,
            mode="pick_and_place",
            place_target=PLACE_TARGET,
        )

        # Verify generated code contains PnP keywords
        code_lower = code.lower()
        assert "transport" in code_lower, "Generated code missing TRANSPORT phase"
        assert "place" in code_lower, "Generated code missing PLACE phase"
        assert "release" in code_lower, "Generated code missing RELEASE phase"

        engine = MockSimulationEngine()
        validator = GraspValidator()

        consecutive = 0
        results_log = []

        for iteration in range(1, MAX_ITERATIONS + 1):
            # Execute in mock simulator
            sim_output = engine.execute(
                code=code, timeout=30.0, place_target=PLACE_TARGET
            )

            # Validate result
            validation = validator.validate(sim_output)
            checks = {name: check.passed for name, check in validation.checks.items()}
            all_passed = all(checks.values())

            results_log.append({
                "iteration": iteration,
                "success": all_passed,
                "checks": checks,
            })

            if all_passed:
                consecutive += 1
                if consecutive >= SUCCESS_THRESHOLD:
                    print(
                        f"\nPnP SUCCESS: {SUCCESS_THRESHOLD} consecutive passes "
                        f"at iteration {iteration}"
                    )
                    print(f"  Total iterations run: {iteration}")
                    print(f"  Total passes: {sum(1 for r in results_log if r['success'])}")
                    print(f"  Total failures: {sum(1 for r in results_log if not r['success'])}")
                    return  # Test passes
            else:
                if consecutive > 0:
                    print(f"  Streak of {consecutive} broken at iteration {iteration}")
                consecutive = 0

                # Log which checks failed
                failed = [k for k, v in checks.items() if not v]
                print(f"  Iteration {iteration} FAILED: {failed}")

        # If we get here, we never achieved the threshold
        total_passes = sum(1 for r in results_log if r["success"])
        total_fails = sum(1 for r in results_log if not r["success"])

        # Print detailed failure analysis
        print(f"\nFAILED: Did not reach {SUCCESS_THRESHOLD} consecutive passes")
        print(f"  Total passes: {total_passes}/{MAX_ITERATIONS}")
        print(f"  Total failures: {total_fails}/{MAX_ITERATIONS}")

        # Count failure reasons
        failure_reasons: dict[str, int] = {}
        for r in results_log:
            if not r["success"]:
                for check, passed in r["checks"].items():
                    if not passed:
                        failure_reasons[check] = failure_reasons.get(check, 0) + 1
        print(f"  Failure reasons: {failure_reasons}")

        pytest.fail(
            f"PnP pipeline failed: only {total_passes}/{MAX_ITERATIONS} passes, "
            f"never reached {SUCCESS_THRESHOLD} consecutive. "
            f"Failure reasons: {failure_reasons}"
        )

    def test_pnp_validator_passes_for_mock_success(self):
        """When MockSimulationEngine returns a successful PnP result,
        ALL 8 validator checks must pass."""
        code = _generate_default_code(
            cad_metadata=MUG_METADATA,
            robot_model=ROBOT_MODEL,
            mode="pick_and_place",
            place_target=PLACE_TARGET,
        )

        engine = MockSimulationEngine()
        validator = GraspValidator()

        # Force a successful result by running many iterations until we get one
        for _ in range(50):
            sim_output = engine.execute(
                code=code, timeout=30.0, place_target=PLACE_TARGET
            )
            if sim_output.get("success"):
                break
        else:
            pytest.skip("Could not get a successful mock result in 50 tries")

        # Validate the successful result
        validation = validator.validate(sim_output)
        checks = {name: check.passed for name, check in validation.checks.items()}

        # All 8 checks must pass
        assert len(checks) == 8, f"Expected 8 checks, got {len(checks)}: {list(checks.keys())}"

        failed = {k: v for k, v in checks.items() if not v}
        assert not failed, (
            f"Mock successful PnP result failed validator checks: {failed}\n"
            f"Sim output keys: {list(sim_output.keys())}\n"
            f"Object state: {sim_output.get('object_final_state')}\n"
            f"Contact forces: {sim_output.get('contact_forces')}\n"
            f"Place accuracy: {sim_output.get('place_accuracy')}"
        )

    def test_grasp_only_still_works(self):
        """Grasp-only mode must still achieve 3 consecutive passes within 100 iters."""
        code = _generate_default_code(
            cad_metadata=MUG_METADATA,
            robot_model=ROBOT_MODEL,
            mode="grasp_only",
        )

        engine = MockSimulationEngine()
        validator = GraspValidator()

        consecutive = 0
        for iteration in range(1, MAX_ITERATIONS + 1):
            sim_output = engine.execute(code=code, timeout=30.0)
            validation = validator.validate(sim_output)
            checks = {name: check.passed for name, check in validation.checks.items()}

            # Should have exactly 5 checks (no PnP checks)
            assert len(checks) == 5, (
                f"Expected 5 checks, got {len(checks)}: {list(checks.keys())}"
            )

            if all(checks.values()):
                consecutive += 1
                if consecutive >= SUCCESS_THRESHOLD:
                    print(f"\nGrasp-only SUCCESS at iteration {iteration}")
                    return
            else:
                consecutive = 0

        pytest.fail(
            f"Grasp-only did not reach {SUCCESS_THRESHOLD} consecutive passes "
            f"within {MAX_ITERATIONS} iterations"
        )
