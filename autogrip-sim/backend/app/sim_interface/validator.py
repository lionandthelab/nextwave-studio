"""Physics validation for grasp simulation results."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# Object-category-specific threshold presets based on Unitree Z1 + H1 hand specs
OBJECT_THRESHOLDS: dict[str, dict[str, float]] = {
    "small_light": {  # < 5cm max dim, < 200g
        "min_contact_force": 1.5,
        "max_safe_force": 30.0,
        "hold_height": 0.15,
        "hold_duration": 3.0,
    },
    "medium": {  # 5-15cm, 200g-2kg
        "min_contact_force": 5.0,
        "max_safe_force": 80.0,
        "hold_height": 0.15,
        "hold_duration": 5.0,
    },
    "large_heavy": {  # > 15cm, 2-5kg
        "min_contact_force": 15.0,
        "max_safe_force": 100.0,
        "hold_height": 0.10,
        "hold_duration": 5.0,
    },
}

# Unitree Z1 arm workspace boundaries (meters from base)
WORKSPACE_LIMITS = {
    "x_min": 0.15, "x_max": 0.60,
    "y_min": -0.40, "y_max": 0.40,
    "z_min": -0.10, "z_max": 0.50,
    "radius_max": 0.60,
}


@dataclass
class CheckResult:
    """Result of a single validation check."""

    name: str
    passed: bool
    value: float
    threshold: float
    message: str


@dataclass
class ValidationResult:
    """Aggregated result of all validation checks."""

    success: bool
    checks: dict[str, CheckResult]
    error_log: str
    suggestions: list[str]


class GraspValidator:
    """Validates simulation results against physics-based criteria."""

    # Default thresholds -- calibrated for Unitree Z1 arm + H1 hand
    HOLD_HEIGHT_THRESHOLD = 0.15  # meters - object must be lifted above this z
    HOLD_DURATION_THRESHOLD = 5.0  # seconds - must hold for at least this long
    MIN_CONTACT_FORCE = 2.0  # Newtons - minimum reliable contact force
    MAX_ANGULAR_VELOCITY = 1.0  # rad/s - object must not spin
    MAX_SAFE_FORCE = 100.0  # Newtons - Unitree Z1/H1 safe operating limit

    def validate(self, sim_result: dict) -> ValidationResult:
        """Run all validation checks on a simulation result.

        Args:
            sim_result: Dict from IsaacSimConnector.execute_code() containing
                success, duration, frames, logs, object_final_state,
                contact_forces, and joint_states.

        Returns:
            ValidationResult with per-check details, overall pass/fail,
            error log, and improvement suggestions.
        """
        checks: dict[str, CheckResult] = {}
        suggestions: list[str] = []
        error_lines: list[str] = []

        # -- Check 1: Hold Test --
        hold_check = self._check_hold(sim_result)
        checks[hold_check.name] = hold_check
        if not hold_check.passed:
            error_lines.append(f"HOLD FAILED: {hold_check.message}")
            suggestions.append(
                "Increase grip torque or ensure the hold phase lasts at least 5 seconds."
            )

        # -- Check 2: Contact Test --
        contact_check = self._check_contact(sim_result)
        checks[contact_check.name] = contact_check
        if not contact_check.passed:
            error_lines.append(f"CONTACT FAILED: {contact_check.message}")
            suggestions.append(
                "Verify that the gripper fingers make contact with the object. "
                "Adjust grasp position or widen the gripper approach."
            )

        # -- Check 3: Stability Test --
        stability_check = self._check_stability(sim_result)
        checks[stability_check.name] = stability_check
        if not stability_check.passed:
            error_lines.append(f"STABILITY FAILED: {stability_check.message}")
            suggestions.append(
                "The object is rotating excessively. Apply symmetric grasp forces "
                "and ensure the grasp is centered on the object's center of mass."
            )

        # -- Check 4: Force Test --
        force_check = self._check_force(sim_result)
        checks[force_check.name] = force_check
        if not force_check.passed:
            error_lines.append(f"FORCE FAILED: {force_check.message}")
            suggestions.append(
                "Applied force is outside the safe operating range. "
                "Reduce torque to protect the robot and object."
            )

        # -- Check 5: Workspace Test --
        workspace_check = self._check_workspace(sim_result)
        checks[workspace_check.name] = workspace_check
        if not workspace_check.passed:
            error_lines.append(f"WORKSPACE FAILED: {workspace_check.message}")
            suggestions.append(
                "The object position is outside the robot's reachable workspace. "
                "Reposition the object within 0.15-0.60m from the robot base."
            )

        overall_success = all(c.passed for c in checks.values())
        error_log = "\n".join(error_lines) if error_lines else ""

        # Append relevant log lines from simulation
        sim_logs = sim_result.get("logs", [])
        error_sim_lines = [
            line for line in sim_logs if "ERROR" in line or "FAILED" in line
        ]
        if error_sim_lines:
            error_log += "\n--- Simulation Logs ---\n" + "\n".join(error_sim_lines)

        result = ValidationResult(
            success=overall_success,
            checks=checks,
            error_log=error_log,
            suggestions=suggestions,
        )

        logger.info(
            "Validation complete: success=%s, passed=%d/%d checks",
            overall_success,
            sum(1 for c in checks.values() if c.passed),
            len(checks),
        )
        return result

    def _check_hold(self, sim_result: dict) -> CheckResult:
        """Check 1 - Hold Test: Object Z-position maintained above threshold for 5+ seconds."""
        obj_state = sim_result.get("object_final_state", {})
        position = obj_state.get("position", [0, 0, 0])
        obj_z = position[2] if len(position) > 2 else 0.0
        duration = sim_result.get("duration", 0.0)

        # Object must be above the height threshold
        height_ok = obj_z >= self.HOLD_HEIGHT_THRESHOLD
        # Simulation must have run long enough for the hold phase
        duration_ok = duration >= self.HOLD_DURATION_THRESHOLD

        passed = height_ok and duration_ok

        if not height_ok:
            msg = (
                f"Object final height {obj_z:.3f}m is below threshold "
                f"{self.HOLD_HEIGHT_THRESHOLD}m - object was dropped."
            )
        elif not duration_ok:
            msg = (
                f"Simulation duration {duration:.1f}s is below required "
                f"{self.HOLD_DURATION_THRESHOLD}s hold time."
            )
        else:
            msg = (
                f"Object held at {obj_z:.3f}m for {duration:.1f}s. "
                f"Height threshold: {self.HOLD_HEIGHT_THRESHOLD}m."
            )

        return CheckResult(
            name="hold_test",
            passed=passed,
            value=obj_z,
            threshold=self.HOLD_HEIGHT_THRESHOLD,
            message=msg,
        )

    def _check_contact(self, sim_result: dict) -> CheckResult:
        """Check 2 - Contact Test: Contact force between fingers and object > 0."""
        contact_forces = sim_result.get("contact_forces", [])

        if not contact_forces:
            return CheckResult(
                name="contact_test",
                passed=False,
                value=0.0,
                threshold=self.MIN_CONTACT_FORCE,
                message="No contact forces detected between gripper and object.",
            )

        total_force = sum(c.get("force_n", 0.0) for c in contact_forces)
        passed = total_force >= self.MIN_CONTACT_FORCE

        if passed:
            msg = (
                f"Total contact force: {total_force:.1f}N across "
                f"{len(contact_forces)} contact point(s). "
                f"Threshold: {self.MIN_CONTACT_FORCE}N."
            )
        else:
            msg = (
                f"Total contact force {total_force:.1f}N is below minimum "
                f"{self.MIN_CONTACT_FORCE}N."
            )

        return CheckResult(
            name="contact_test",
            passed=passed,
            value=total_force,
            threshold=self.MIN_CONTACT_FORCE,
            message=msg,
        )

    def _check_stability(self, sim_result: dict) -> CheckResult:
        """Check 3 - Stability Test: Object angular velocity below threshold."""
        obj_state = sim_result.get("object_final_state", {})
        ang_vel = obj_state.get("angular_velocity", [0, 0, 0])

        magnitude = math.sqrt(sum(v**2 for v in ang_vel))
        passed = magnitude <= self.MAX_ANGULAR_VELOCITY

        if passed:
            msg = (
                f"Object angular velocity {magnitude:.3f} rad/s is within "
                f"stable range (threshold: {self.MAX_ANGULAR_VELOCITY} rad/s)."
            )
        else:
            msg = (
                f"Object angular velocity {magnitude:.3f} rad/s exceeds "
                f"stability threshold {self.MAX_ANGULAR_VELOCITY} rad/s - "
                f"object is spinning."
            )

        return CheckResult(
            name="stability_test",
            passed=passed,
            value=magnitude,
            threshold=self.MAX_ANGULAR_VELOCITY,
            message=msg,
        )

    def _check_force(self, sim_result: dict) -> CheckResult:
        """Check 4 - Force Test: Applied force within robot's safe operating range."""
        joint_states = sim_result.get("joint_states", {})

        max_torque = 0.0
        for joint_name, state in joint_states.items():
            torque = abs(state.get("torque", 0.0))
            if torque > max_torque:
                max_torque = torque

        # Also check contact forces
        contact_forces = sim_result.get("contact_forces", [])
        max_contact = 0.0
        for cf in contact_forces:
            force = cf.get("force_n", 0.0)
            if force > max_contact:
                max_contact = force

        value = max(max_torque, max_contact)
        passed = value <= self.MAX_SAFE_FORCE

        if passed:
            msg = (
                f"Maximum applied force/torque {value:.1f}N is within safe range "
                f"(limit: {self.MAX_SAFE_FORCE}N)."
            )
        else:
            msg = (
                f"Maximum applied force/torque {value:.1f}N exceeds safe limit "
                f"{self.MAX_SAFE_FORCE}N - risk of damage to robot or object."
            )

        return CheckResult(
            name="force_test",
            passed=passed,
            value=value,
            threshold=self.MAX_SAFE_FORCE,
            message=msg,
        )

    def _check_workspace(self, sim_result: dict) -> CheckResult:
        """Check 5 - Workspace Test: Object position within robot reachable workspace."""
        obj_state = sim_result.get("object_final_state", {})
        position = obj_state.get("position", [0, 0, 0])

        x = position[0] if len(position) > 0 else 0.0
        y = position[1] if len(position) > 1 else 0.0

        dist = math.sqrt(x**2 + y**2)
        within = (
            WORKSPACE_LIMITS["x_min"] <= x <= WORKSPACE_LIMITS["x_max"]
            and WORKSPACE_LIMITS["y_min"] <= y <= WORKSPACE_LIMITS["y_max"]
            and dist <= WORKSPACE_LIMITS["radius_max"]
        )

        if within:
            msg = (
                f"Object at ({x:.3f}, {y:.3f}) is within workspace "
                f"(radius {dist:.3f}m <= {WORKSPACE_LIMITS['radius_max']}m)."
            )
        else:
            msg = (
                f"Object at ({x:.3f}, {y:.3f}) is outside reachable workspace "
                f"(radius {dist:.3f}m, limits: x=[{WORKSPACE_LIMITS['x_min']}, "
                f"{WORKSPACE_LIMITS['x_max']}], y=[{WORKSPACE_LIMITS['y_min']}, "
                f"{WORKSPACE_LIMITS['y_max']}])."
            )

        return CheckResult(
            name="workspace_test",
            passed=within,
            value=dist,
            threshold=WORKSPACE_LIMITS["radius_max"],
            message=msg,
        )
