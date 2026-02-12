"""Tests for the GraspValidator (app/sim_interface/validator.py)."""

from __future__ import annotations

import math

import pytest

from app.sim_interface.validator import GraspValidator


@pytest.fixture()
def validator() -> GraspValidator:
    """Provide a fresh GraspValidator instance."""
    return GraspValidator()


def _make_sim_result(
    *,
    obj_z: float = 0.3,
    duration: float = 6.0,
    contact_forces: list[dict] | None = None,
    angular_velocity: list[float] | None = None,
    joint_states: dict | None = None,
    logs: list[str] | None = None,
) -> dict:
    """Build a sim_result dict with sensible defaults for a successful grasp."""
    if contact_forces is None:
        contact_forces = [
            {"finger": "left", "force_n": 5.0},
            {"finger": "right", "force_n": 5.0},
        ]
    if angular_velocity is None:
        angular_velocity = [0.01, 0.01, 0.01]
    if joint_states is None:
        joint_states = {
            "joint_0": {"position": 0.0, "torque": 1.0},
            "gripper_left": {"position": 0.04, "torque": 3.0},
            "gripper_right": {"position": -0.04, "torque": 3.0},
        }
    if logs is None:
        logs = []

    return {
        "success": True,
        "duration": duration,
        "frames": [],
        "logs": logs,
        "object_final_state": {
            "position": [0.5, 0.0, obj_z],
            "velocity": [0.0, 0.0, 0.0],
            "angular_velocity": angular_velocity,
            "contact_count": len(contact_forces),
        },
        "contact_forces": contact_forces,
        "joint_states": joint_states,
    }


class TestSuccessfulGrasp:
    """Tests for a fully passing simulation result."""

    def test_successful_grasp(self, validator: GraspValidator):
        """All 4 checks should pass with good sim_result values."""
        result = validator.validate(_make_sim_result())
        assert result.success is True
        assert len(result.checks) == 4
        assert all(c.passed for c in result.checks.values())
        assert result.error_log == ""
        assert result.suggestions == []


class TestHoldFailure:
    """Tests for hold check failures."""

    def test_hold_failure_dropped(self, validator: GraspValidator):
        """Object at z=0 (dropped to ground) should fail the hold test."""
        sim = _make_sim_result(obj_z=0.0)
        result = validator.validate(sim)
        assert result.checks["hold_test"].passed is False
        assert "dropped" in result.checks["hold_test"].message.lower()

    def test_hold_failure_short_duration(self, validator: GraspValidator):
        """Simulation that ran less than 5s should fail the hold test."""
        sim = _make_sim_result(duration=3.0)
        result = validator.validate(sim)
        assert result.checks["hold_test"].passed is False
        assert "duration" in result.checks["hold_test"].message.lower()


class TestContactFailure:
    """Tests for contact check failures."""

    def test_contact_failure_empty(self, validator: GraspValidator):
        """Empty contact forces should fail the contact test."""
        sim = _make_sim_result(contact_forces=[])
        result = validator.validate(sim)
        assert result.checks["contact_test"].passed is False
        assert "no contact" in result.checks["contact_test"].message.lower()

    def test_contact_failure_low_force(self, validator: GraspValidator):
        """Total contact force below threshold should fail."""
        sim = _make_sim_result(
            contact_forces=[{"finger": "left", "force_n": 0.1}]
        )
        result = validator.validate(sim)
        assert result.checks["contact_test"].passed is False


class TestStabilityFailure:
    """Tests for stability check failures."""

    def test_stability_failure(self, validator: GraspValidator):
        """High angular velocity should fail the stability test."""
        sim = _make_sim_result(angular_velocity=[2.0, 2.0, 2.0])
        result = validator.validate(sim)
        assert result.checks["stability_test"].passed is False
        expected_mag = math.sqrt(4.0 + 4.0 + 4.0)
        assert result.checks["stability_test"].value == pytest.approx(
            expected_mag, abs=0.01
        )
        assert "spinning" in result.checks["stability_test"].message.lower()


class TestForceExceeded:
    """Tests for force limit check failures."""

    def test_force_exceeded_torque(self, validator: GraspValidator):
        """Excessive joint torque should fail the force test."""
        sim = _make_sim_result(
            joint_states={
                "gripper_left": {"position": 0.04, "torque": 600.0},
                "gripper_right": {"position": -0.04, "torque": 600.0},
            }
        )
        result = validator.validate(sim)
        assert result.checks["force_test"].passed is False
        assert result.checks["force_test"].value == 600.0

    def test_force_exceeded_contact(self, validator: GraspValidator):
        """Excessive contact force should fail the force test."""
        sim = _make_sim_result(
            contact_forces=[{"finger": "left", "force_n": 700.0}],
            joint_states={"j1": {"position": 0.0, "torque": 1.0}},
        )
        result = validator.validate(sim)
        assert result.checks["force_test"].passed is False
        assert result.checks["force_test"].value == 700.0


class TestMixedFailures:
    """Tests for multiple simultaneous failures."""

    def test_mixed_failures(self, validator: GraspValidator):
        """Multiple checks should fail simultaneously and all be reported."""
        sim = _make_sim_result(
            obj_z=0.0,  # hold fail
            contact_forces=[],  # contact fail
            angular_velocity=[3.0, 3.0, 3.0],  # stability fail
        )
        result = validator.validate(sim)
        assert result.success is False
        failed_names = [
            name for name, chk in result.checks.items() if not chk.passed
        ]
        assert "hold_test" in failed_names
        assert "contact_test" in failed_names
        assert "stability_test" in failed_names
        # force_test may pass since joint_states defaults have low torques
        assert len(result.suggestions) >= 3


class TestEdgeCaseThresholds:
    """Tests for values exactly at validation thresholds."""

    def test_hold_at_threshold(self, validator: GraspValidator):
        """Object z exactly at HOLD_HEIGHT_THRESHOLD should pass."""
        sim = _make_sim_result(obj_z=GraspValidator.HOLD_HEIGHT_THRESHOLD)
        result = validator.validate(sim)
        assert result.checks["hold_test"].passed is True

    def test_hold_just_below_threshold(self, validator: GraspValidator):
        """Object z just below HOLD_HEIGHT_THRESHOLD should fail."""
        sim = _make_sim_result(
            obj_z=GraspValidator.HOLD_HEIGHT_THRESHOLD - 0.001
        )
        result = validator.validate(sim)
        assert result.checks["hold_test"].passed is False

    def test_contact_at_threshold(self, validator: GraspValidator):
        """Contact force exactly at MIN_CONTACT_FORCE should pass."""
        sim = _make_sim_result(
            contact_forces=[
                {"finger": "left", "force_n": GraspValidator.MIN_CONTACT_FORCE}
            ]
        )
        result = validator.validate(sim)
        assert result.checks["contact_test"].passed is True

    def test_stability_at_threshold(self, validator: GraspValidator):
        """Angular velocity magnitude exactly at MAX_ANGULAR_VELOCITY should pass."""
        # Magnitude of (v, 0, 0) = v; set to threshold exactly
        threshold = GraspValidator.MAX_ANGULAR_VELOCITY
        sim = _make_sim_result(angular_velocity=[threshold, 0.0, 0.0])
        result = validator.validate(sim)
        assert result.checks["stability_test"].passed is True

    def test_stability_just_above_threshold(self, validator: GraspValidator):
        """Angular velocity just above MAX_ANGULAR_VELOCITY should fail."""
        threshold = GraspValidator.MAX_ANGULAR_VELOCITY
        sim = _make_sim_result(angular_velocity=[threshold + 0.001, 0.0, 0.0])
        result = validator.validate(sim)
        assert result.checks["stability_test"].passed is False

    def test_force_at_threshold(self, validator: GraspValidator):
        """Force exactly at MAX_SAFE_FORCE should pass."""
        sim = _make_sim_result(
            joint_states={
                "gripper": {
                    "position": 0.0,
                    "torque": GraspValidator.MAX_SAFE_FORCE,
                }
            }
        )
        result = validator.validate(sim)
        assert result.checks["force_test"].passed is True

    def test_force_just_above_threshold(self, validator: GraspValidator):
        """Force just above MAX_SAFE_FORCE should fail."""
        sim = _make_sim_result(
            joint_states={
                "gripper": {
                    "position": 0.0,
                    "torque": GraspValidator.MAX_SAFE_FORCE + 0.1,
                }
            }
        )
        result = validator.validate(sim)
        assert result.checks["force_test"].passed is False

    def test_duration_at_threshold(self, validator: GraspValidator):
        """Duration exactly at HOLD_DURATION_THRESHOLD should pass."""
        sim = _make_sim_result(
            duration=GraspValidator.HOLD_DURATION_THRESHOLD
        )
        result = validator.validate(sim)
        assert result.checks["hold_test"].passed is True
