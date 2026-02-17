"""
Isaac Sim Internal Server
Runs inside the Isaac Sim Docker container.
Provides a REST API for the main backend to control simulations.

Response format is aligned with what GraspValidator expects:
- object_final_state: {position, velocity, angular_velocity, contact_count}
- contact_forces: [{finger, force_n}, ...]
- joint_states: {joint_name: {position, torque}, ...}
"""

import asyncio
import base64
import logging
import os
import queue
import random
import re
import struct
import threading
import time
import zlib
from concurrent.futures import Future

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("isaac-sim-server")

app = FastAPI(title="AutoGrip Isaac Sim Server", version="2.0.0")

# Global simulation state
sim_state = {
    "initialized": False,
    "scene_loaded": False,
    "robot_loaded": False,
    "object_loaded": False,
    "running": False,
    "headless": True,
    "robot_model": None,
    "object_path": None,
    "object_position": [0.5, 0.0, 0.05],
}


class SceneConfig(BaseModel):
    gravity: float = -9.81
    time_step: float = 1.0 / 60.0
    ground_plane: bool = True
    enable_pick_and_place: bool = False
    pick_table_position: list[float] = [0.5, 0.0, 0.0]
    place_tray_position: list[float] = [0.5, 0.4, 0.0]


class RobotConfig(BaseModel):
    model: str = "franka_allegro"
    usd_path: str | None = None
    position: list[float] = [0.0, 0.0, 0.0]


class ObjectConfig(BaseModel):
    cad_file_path: str
    position: list[float] = [0.0, 0.0, 0.05]
    scale: list[float] = [1.0, 1.0, 1.0]


class ExecuteRequest(BaseModel):
    code: str
    timeout: float = 30.0
    place_target: list[float] | None = None


class InitRequest(BaseModel):
    headless: bool = True


_SIMULATION_APP = None

# Try to import Isaac Sim - falls back to mock if not available
try:
    from omni.isaac.core import World
    from omni.isaac.core.utils.stage import add_reference_to_stage
    from omni.isaac.core.prims import XFormPrim, RigidPrim
    from omni.isaac.core.robots import Robot
    from omni.isaac.core.articulations import Articulation
    from omni.isaac.core.utils.nucleus import get_assets_root_path
    from omni.isaac.sensor import ContactSensor
    import omni.isaac.core.utils.physics as physics_utils
    from pxr import UsdPhysics, PhysxSchema, Gf, Sdf, UsdGeom, Usd
    import numpy as np

    ISAAC_SIM_AVAILABLE = True
    logger.info("Isaac Sim Python API loaded successfully")
except ImportError:
    ISAAC_SIM_AVAILABLE = False
    logger.warning("Isaac Sim not available - running in mock mode")


# ---------------------------------------------------------------------------
# Mock simulation engine (used when Isaac Sim is not available)
# ---------------------------------------------------------------------------


class MockSimulationEngine:
    """Mock physics engine that produces realistic grasp simulation results.

    Models approximate physics to provide realistic success/failure patterns.
    Code quality is evaluated via regex analysis of the generated grasping code.
    """

    def __init__(self):
        self._rng = random.Random()
        self._iteration_count = 0

    def evaluate_code_quality(self, code: str, place_target: list[float] | None = None) -> dict:
        """Analyze generated code to estimate its quality."""
        quality = {
            "torque_value": 2.0,
            "grasp_width": 0.08,
            "approach_height": 0.1,
            "has_error_handling": False,
            "has_contact_check": False,
            "has_hold_phase": False,
            "velocity_limit": 1.0,
            "has_transport_phase": False,
            "has_place_phase": False,
            "has_release_phase": False,
            "place_target": place_target,
        }

        torque_matches = re.findall(
            r"torque\s*[=:]\s*(\d+\.?\d*)", code, re.IGNORECASE
        )
        if torque_matches:
            quality["torque_value"] = max(float(t) for t in torque_matches)

        width_matches = re.findall(
            r"(?:grasp|grip)[\s_]*width\s*[=:]\s*(\d+\.?\d*)", code, re.IGNORECASE
        )
        if width_matches:
            quality["grasp_width"] = float(width_matches[-1])

        height_matches = re.findall(
            r"(?:approach|pre.?grasp)[\s_]*height\s*[=:]\s*(\d+\.?\d*)",
            code,
            re.IGNORECASE,
        )
        if height_matches:
            quality["approach_height"] = float(height_matches[-1])

        quality["has_error_handling"] = "try" in code and "except" in code
        quality["has_contact_check"] = any(
            kw in code.lower()
            for kw in ("contact", "force_sensor", "get_contact", "is_touching")
        )
        quality["has_hold_phase"] = any(
            kw in code.lower() for kw in ("hold", "maintain", "sleep(5", "wait(5")
        )

        # Pick-and-place phase detection
        quality["has_transport_phase"] = any(
            kw in code.lower()
            for kw in ("transport", "move_to_place", "lateral", "move_lateral")
        )
        quality["has_place_phase"] = any(
            kw in code.lower()
            for kw in ("place", "lower_to_tray", "descend_to_place", "set_down")
        )
        quality["has_release_phase"] = any(
            kw in code.lower()
            for kw in ("release", "open_finger", "open_gripper", "ungrasp")
        )

        return quality

    def compute_success_probability(
        self, code_quality: dict, iteration: int
    ) -> float:
        """Compute probability of successful grasp based on code quality."""
        base_prob = 0.05

        torque = code_quality["torque_value"]
        if torque < 1.0:
            torque_bonus = 0.0
        elif torque < 5.0:
            torque_bonus = torque * 0.08
        elif torque < 15.0:
            torque_bonus = 0.4
        else:
            torque_bonus = 0.3

        width = code_quality["grasp_width"]
        width_bonus = 0.15 if 0.03 < width < 0.12 else 0.0

        error_bonus = 0.05 if code_quality["has_error_handling"] else 0.0
        contact_bonus = 0.1 if code_quality["has_contact_check"] else 0.0
        hold_bonus = 0.1 if code_quality["has_hold_phase"] else 0.0
        iter_bonus = min(0.15, iteration * 0.03)

        prob = (
            base_prob
            + torque_bonus
            + width_bonus
            + error_bonus
            + contact_bonus
            + hold_bonus
            + iter_bonus
        )
        return min(0.95, prob)

    def determine_failure_mode(self, code_quality: dict) -> str:
        """Determine the most likely failure mode based on code deficiencies."""
        issues: list[tuple[str, float]] = []

        if code_quality["torque_value"] < 2.0:
            issues.append(("slip", 0.5))
        if code_quality["torque_value"] > 12.0:
            issues.append(("overforce", 0.4))
        if code_quality["grasp_width"] > 0.12:
            issues.append(("no_contact", 0.4))
        if code_quality["grasp_width"] < 0.02:
            issues.append(("collision", 0.3))
        if code_quality["approach_height"] < 0.05:
            issues.append(("collision", 0.4))
        if not code_quality["has_hold_phase"]:
            issues.append(("slip", 0.3))
        if not code_quality["has_contact_check"]:
            issues.append(("unstable_grasp", 0.25))
        if code_quality["velocity_limit"] > 2.0:
            issues.append(("timeout", 0.2))
        if code_quality["velocity_limit"] > 1.5:
            issues.append(("collision", 0.2))

        # Pick-and-place specific failure modes
        if code_quality.get("place_target") is not None:
            if not code_quality["has_transport_phase"]:
                issues.append(("transport_drop", 0.5))
            if not code_quality["has_place_phase"]:
                issues.append(("place_miss", 0.4))
            if not code_quality["has_release_phase"]:
                issues.append(("place_miss", 0.3))

        if not issues:
            return self._rng.choice(["slip", "no_contact", "collision"])

        total = sum(w for _, w in issues)
        r = self._rng.random() * total
        cumulative = 0.0
        for mode, weight in issues:
            cumulative += weight
            if r <= cumulative:
                return mode
        return issues[-1][0]

    def execute(
        self, code: str, timeout: float = 30.0, place_target: list[float] | None = None
    ) -> dict:
        """Execute code in the mock simulator and return results.

        Returns results in the format expected by the validator.
        """
        self._iteration_count += 1
        quality = self.evaluate_code_quality(code, place_target=place_target)
        success_prob = self.compute_success_probability(
            quality, self._iteration_count
        )

        # Reduce success probability for pick-and-place (more phases = harder)
        if place_target is not None:
            phase_bonus = sum([
                0.05 if quality["has_transport_phase"] else 0.0,
                0.05 if quality["has_place_phase"] else 0.0,
                0.05 if quality["has_release_phase"] else 0.0,
            ])
            success_prob = success_prob * 0.7 + phase_bonus

        success = self._rng.random() < success_prob
        duration = self._rng.uniform(3.0, 8.0)
        if place_target is not None:
            duration = self._rng.uniform(6.0, 14.0)  # longer for pick-and-place

        if success:
            if place_target is not None:
                sim_data = self._simulate_successful_pick_and_place(quality, duration, place_target)
            else:
                sim_data = self._simulate_successful_grasp(quality, duration)
        else:
            failure_mode = self.determine_failure_mode(quality)
            if place_target is not None and failure_mode in ("place_miss", "transport_drop"):
                sim_data = self._simulate_failed_pick_and_place(
                    quality, duration, failure_mode, place_target
                )
            else:
                sim_data = self._simulate_failed_grasp(quality, duration, failure_mode)

        frames_b64 = [
            base64.b64encode(frame_bytes).decode("ascii")
            for frame_bytes in sim_data["frames"]
        ]

        result = {
            "success": success,
            "duration": duration,
            "object_final_state": sim_data["object_state"],
            "contact_forces": sim_data["contact_forces"],
            "joint_states": sim_data["joint_states"],
            "logs": sim_data["logs"],
            "frames": frames_b64,
            "error": None if success else (
                sim_data["logs"][-1] if sim_data["logs"] else None
            ),
        }

        # Add pick-and-place fields when applicable
        if place_target is not None:
            result["place_target"] = place_target
            result["object_trajectory"] = sim_data.get("object_trajectory", [])
            # Compute place_accuracy
            final_pos = sim_data["object_state"]["position"]
            import math
            result["place_accuracy"] = math.sqrt(
                sum((a - b) ** 2 for a, b in zip(final_pos, place_target))
            )

        return result

    def _simulate_successful_grasp(self, quality: dict, duration: float) -> dict:
        logs = [
            f"[{0.0:.2f}s] Initializing grasp sequence",
            f"[{0.3:.2f}s] Moving to pre-grasp position",
            f"[{1.0:.2f}s] Approaching object - clearance: {quality['approach_height']:.3f}m",
            f"[{1.4:.2f}s] Contact detected on index finger (force: {quality['torque_value'] / 0.03 * 0.30:.1f}N)",
            f"[{1.5:.2f}s] Contact detected on middle finger (force: {quality['torque_value'] / 0.03 * 0.28:.1f}N)",
            f"[{1.6:.2f}s] Contact detected on ring finger (force: {quality['torque_value'] / 0.03 * 0.22:.1f}N)",
            f"[{1.7:.2f}s] Contact detected on thumb (force: {quality['torque_value'] / 0.03 * 0.20:.1f}N)",
            f"[{1.8:.2f}s] Fingers closed - spread: {quality['grasp_width']:.4f}m",
            f"[{2.0:.2f}s] Applying finger torques: {quality['torque_value']:.1f}Nm total",
            f"[{2.2:.2f}s] Grip stable - beginning lift",
            f"[{2.5:.2f}s] Lifting object - height: 0.05m",
            f"[{3.0:.2f}s] Lifting object - height: 0.15m",
            f"[{3.5:.2f}s] Lifting object - height: 0.30m",
            f"[{3.7:.2f}s] Target height reached - holding",
            f"[{4.0:.2f}s] Hold phase: 0.3s - object stable",
            f"[{5.0:.2f}s] Hold phase: 1.3s - object stable",
            f"[{6.0:.2f}s] Hold phase: 2.3s - object stable",
            f"[{7.0:.2f}s] Hold phase: 3.3s - object stable",
            f"[{8.0:.2f}s] Hold phase: 4.3s - object stable",
            f"[{8.7:.2f}s] Hold phase complete (5.0s) - GRASP SUCCESSFUL",
        ]

        num_frames = int(duration * 10)
        frames = [self._generate_frame(i, num_frames, success=True) for i in range(num_frames)]

        final_height = 0.3 + self._rng.uniform(-0.005, 0.005)
        return {
            "logs": logs,
            "frames": frames,
            "object_state": {
                "position": [0.5, 0.0, final_height],
                "velocity": [0.0, 0.0, 0.0],
                "angular_velocity": [
                    self._rng.uniform(-0.01, 0.01),
                    self._rng.uniform(-0.01, 0.01),
                    self._rng.uniform(-0.01, 0.01),
                ],
                "contact_count": 2,
            },
            "contact_forces": [
                {"finger": "index", "force_n": quality["torque_value"] / 0.03 * 0.30},
                {"finger": "middle", "force_n": quality["torque_value"] / 0.03 * 0.28},
                {"finger": "ring", "force_n": quality["torque_value"] / 0.03 * 0.22},
                {"finger": "thumb", "force_n": quality["torque_value"] / 0.03 * 0.20},
            ],
            "joint_states": {
                "panda_joint1": {"position": 0.0, "torque": 0.0},
                "panda_joint2": {"position": -0.5, "torque": 2.1},
                "panda_joint3": {"position": 0.0, "torque": 0.5},
                "panda_joint4": {"position": -2.0, "torque": 3.4},
                "panda_joint5": {"position": 0.0, "torque": 0.3},
                "panda_joint6": {"position": 1.5, "torque": 1.2},
                "panda_joint7": {"position": 0.7, "torque": 0.4},
                "joint_0": {"position": 0.8, "torque": quality["torque_value"] * 0.25},
                "joint_4": {"position": 0.8, "torque": quality["torque_value"] * 0.25},
                "joint_8": {"position": 0.8, "torque": quality["torque_value"] * 0.25},
                "joint_12": {"position": 1.0, "torque": quality["torque_value"] * 0.25},
            },
        }

    def _simulate_failed_grasp(
        self, quality: dict, duration: float, failure_mode: str
    ) -> dict:
        logs = [
            f"[{0.0:.2f}s] Initializing grasp sequence",
            f"[{0.3:.2f}s] Moving to pre-grasp position",
        ]

        if failure_mode == "slip":
            fail_time = self._rng.uniform(2.5, 4.0)
            logs.extend([
                f"[{1.0:.2f}s] Approaching object",
                f"[{1.5:.2f}s] Contact detected - closing gripper",
                f"[{1.8:.2f}s] Gripper closed - width: {quality['grasp_width']:.4f}m",
                f"[{2.0:.2f}s] Applying torque: {quality['torque_value']:.1f}Nm",
                f"[{2.2:.2f}s] Beginning lift",
                f"[{2.5:.2f}s] Lifting - height: 0.05m",
                f"[{fail_time:.2f}s] WARNING: Contact force decreasing rapidly",
                f"[{fail_time + 0.1:.2f}s] WARNING: Object slipping - force below threshold",
                f"[{fail_time + 0.2:.2f}s] ERROR: Object dropped - contact lost",
                f"[{fail_time + 0.5:.2f}s] Object fell to ground plane (z=0.0)",
                f"[{fail_time + 0.7:.2f}s] GRASP FAILED: slip detected - insufficient grip force",
            ])
            object_z = 0.0
            contact_count = 0

        elif failure_mode == "collision":
            fail_time = self._rng.uniform(0.8, 1.5)
            logs.extend([
                f"[{0.8:.2f}s] Approaching object - clearance: {quality['approach_height']:.3f}m",
                f"[{fail_time:.2f}s] ERROR: Collision detected between gripper and object",
                f"[{fail_time + 0.1:.2f}s] Impact force: {self._rng.uniform(50, 200):.1f}N",
                f"[{fail_time + 0.2:.2f}s] Object displaced by collision",
                f"[{fail_time + 0.5:.2f}s] Emergency stop triggered",
                f"[{fail_time + 0.7:.2f}s] GRASP FAILED: collision during approach - adjust approach angle or clearance",
            ])
            object_z = self._rng.uniform(0.0, 0.05)
            contact_count = 0

        elif failure_mode == "no_contact":
            logs.extend([
                f"[{1.0:.2f}s] Approaching target position",
                f"[{1.5:.2f}s] Closing gripper at target position",
                f"[{2.0:.2f}s] Gripper closed - width: {quality['grasp_width']:.4f}m",
                f"[{2.2:.2f}s] WARNING: No contact force detected on fingers",
                f"[{2.5:.2f}s] Retrying grasp closure...",
                f"[{3.0:.2f}s] WARNING: Still no contact detected",
                f"[{3.5:.2f}s] GRASP FAILED: no_contact - gripper missed the object, check grasp position and width",
            ])
            object_z = self._rng.uniform(0.04, 0.06)
            contact_count = 0

        elif failure_mode == "overforce":
            fail_time = self._rng.uniform(1.5, 2.5)
            logs.extend([
                f"[{1.0:.2f}s] Approaching object",
                f"[{1.5:.2f}s] Contact detected - closing gripper",
                f"[{1.8:.2f}s] Gripper closed - width: {quality['grasp_width']:.4f}m",
                f"[{2.0:.2f}s] Applying torque: {quality['torque_value']:.1f}Nm",
                f"[{fail_time:.2f}s] WARNING: Joint torque exceeded safe limit (max 50Nm)",
                f"[{fail_time + 0.1:.2f}s] WARNING: Excessive contact force {quality['torque_value'] / 0.06:.1f}N detected",
                f"[{fail_time + 0.2:.2f}s] ERROR: Emergency stop - force exceeded safe limit, risk of damage",
                f"[{fail_time + 0.5:.2f}s] GRASP FAILED: overforce - reduce gripper torque to prevent damage",
            ])
            object_z = self._rng.uniform(0.0, 0.05)
            contact_count = 0

        elif failure_mode == "unstable_grasp":
            fail_time = self._rng.uniform(3.0, 5.0)
            logs.extend([
                f"[{1.0:.2f}s] Approaching object",
                f"[{1.5:.2f}s] Contact detected - closing gripper",
                f"[{1.8:.2f}s] Gripper closed - width: {quality['grasp_width']:.4f}m",
                f"[{2.0:.2f}s] Applying torque: {quality['torque_value']:.1f}Nm",
                f"[{2.2:.2f}s] Beginning lift",
                f"[{2.5:.2f}s] Lifting - height: 0.05m",
                f"[{3.0:.2f}s] Lifting - height: 0.15m",
                f"[{fail_time:.2f}s] WARNING: Object rotating during lift - angular velocity increasing",
                f"[{fail_time + 0.2:.2f}s] WARNING: Object angular velocity 1.5 rad/s exceeds stability threshold",
                f"[{fail_time + 0.5:.2f}s] ERROR: Unstable grasp - object shifting in gripper",
                f"[{fail_time + 0.7:.2f}s] GRASP FAILED: unstable_grasp - grasp not centered on center of mass",
            ])
            object_z = self._rng.uniform(0.05, 0.15)
            contact_count = 1

        else:  # timeout
            logs.extend([
                f"[{1.0:.2f}s] Computing motion plan...",
                f"[{3.0:.2f}s] Motion plan step 1/15 executing",
                f"[{5.0:.2f}s] Motion plan step 4/15 executing",
                f"[{8.0:.2f}s] Motion plan step 7/15 executing",
                f"[{10.0:.2f}s] WARNING: Execution time exceeding expected duration",
                f"[{12.0:.2f}s] ERROR: Simulation timeout exceeded (12s limit)",
                f"[{12.0:.2f}s] GRASP FAILED: timeout - motion plan too complex, simplify trajectory",
            ])
            object_z = self._rng.uniform(0.04, 0.06)
            contact_count = 0
            duration = 12.0

        num_frames = int(duration * 10)
        frames = [self._generate_frame(i, num_frames, success=False) for i in range(num_frames)]

        return {
            "logs": logs,
            "frames": frames,
            "object_state": {
                "position": [
                    0.5 + self._rng.uniform(-0.05, 0.05),
                    self._rng.uniform(-0.05, 0.05),
                    object_z,
                ],
                "velocity": [
                    self._rng.uniform(-0.1, 0.1),
                    self._rng.uniform(-0.1, 0.1),
                    self._rng.uniform(-0.5, 0.0) if failure_mode == "slip" else 0.0,
                ],
                "angular_velocity": [
                    self._rng.uniform(-0.5, 0.5),
                    self._rng.uniform(-0.5, 0.5),
                    self._rng.uniform(-0.5, 0.5),
                ],
                "contact_count": contact_count,
            },
            "contact_forces": [],
            "joint_states": {
                "panda_joint1": {"position": 0.0, "torque": 0.0},
                "panda_joint2": {"position": -0.3, "torque": 1.5},
                "panda_joint4": {"position": -1.8, "torque": 2.0},
                "joint_0": {"position": 0.0, "torque": 0.0},
                "joint_4": {"position": 0.0, "torque": 0.0},
                "joint_8": {"position": 0.0, "torque": 0.0},
                "joint_12": {"position": 0.0, "torque": 0.0},
            },
        }

    def _simulate_successful_pick_and_place(
        self, quality: dict, duration: float, place_target: list[float]
    ) -> dict:
        """Simulate a successful 8-phase pick-and-place sequence."""
        logs = [
            f"[{0.0:.2f}s] Initializing pick-and-place sequence",
            f"[{0.3:.2f}s] Phase 1 APPROACH: Moving to pre-grasp position",
            f"[{1.0:.2f}s] Phase 2 DESCEND: Lowering to object",
            f"[{1.5:.2f}s] Phase 3 GRASP: Contact detected, closing fingers",
            f"[{2.0:.2f}s] Fingers closed - torque: {quality['torque_value']:.1f}Nm",
            f"[{2.5:.2f}s] Phase 4 LIFT: Raising object to transport height",
            f"[{3.0:.2f}s] Object lifted to 0.30m",
            f"[{3.5:.2f}s] Phase 5 TRANSPORT: Moving to place target ({place_target[0]:.2f}, {place_target[1]:.2f}, {place_target[2]:.2f})",
            f"[{5.0:.2f}s] Arrived above place target",
            f"[{5.5:.2f}s] Phase 6 PLACE: Lowering object into tray",
            f"[{6.5:.2f}s] Object at tray height",
            f"[{7.0:.2f}s] Phase 7 RELEASE: Opening fingers gradually",
            f"[{7.5:.2f}s] Fingers fully open - object released",
            f"[{8.0:.2f}s] Phase 8 RETRACT: Raising arm away",
            f"[{8.5:.2f}s] Retract complete - arm at safe height",
            f"[{9.0:.2f}s] PICK-AND-PLACE SUCCESSFUL",
        ]

        num_frames = int(duration * 10)
        frames = [self._generate_frame(i, num_frames, success=True) for i in range(num_frames)]

        # Object trajectory: pick position -> lift -> transport -> place
        pick_pos = [0.5, 0.0, 0.05]
        lift_pos = [0.5, 0.0, 0.30]
        transport_pos = [place_target[0], place_target[1], 0.30]
        final_pos = [
            place_target[0] + self._rng.uniform(-0.01, 0.01),
            place_target[1] + self._rng.uniform(-0.01, 0.01),
            place_target[2] + self._rng.uniform(-0.005, 0.005),
        ]

        trajectory = [
            {"position": pick_pos, "timestamp": 0.0},
            {"position": [0.5, 0.0, 0.15], "timestamp": 2.5},
            {"position": lift_pos, "timestamp": 3.0},
            {"position": [
                (lift_pos[0] + transport_pos[0]) / 2,
                (lift_pos[1] + transport_pos[1]) / 2,
                0.30,
            ], "timestamp": 4.0},
            {"position": transport_pos, "timestamp": 5.0},
            {"position": [place_target[0], place_target[1], 0.15], "timestamp": 6.0},
            {"position": final_pos, "timestamp": 7.0},
        ]

        return {
            "logs": logs,
            "frames": frames,
            "object_state": {
                "position": final_pos,
                "velocity": [0.0, 0.0, 0.0],
                "angular_velocity": [
                    self._rng.uniform(-0.01, 0.01),
                    self._rng.uniform(-0.01, 0.01),
                    self._rng.uniform(-0.01, 0.01),
                ],
                "contact_count": 0,
            },
            "object_trajectory": trajectory,
            "contact_forces": [
                {"finger": "index", "force_n": quality["torque_value"] / 0.06 * 0.30},
                {"finger": "middle", "force_n": quality["torque_value"] / 0.06 * 0.28},
                {"finger": "ring", "force_n": quality["torque_value"] / 0.06 * 0.22},
                {"finger": "thumb", "force_n": quality["torque_value"] / 0.06 * 0.20},
            ],
            "joint_states": {
                "panda_joint1": {"position": 0.0, "torque": 0.0},
                "panda_joint2": {"position": -0.5, "torque": 2.1},
                "panda_joint3": {"position": 0.0, "torque": 0.5},
                "panda_joint4": {"position": -2.0, "torque": 3.4},
                "panda_joint5": {"position": 0.0, "torque": 0.3},
                "panda_joint6": {"position": 1.5, "torque": 1.2},
                "panda_joint7": {"position": 0.7, "torque": 0.4},
                "joint_0": {"position": 0.0, "torque": 0.0},
                "joint_4": {"position": 0.0, "torque": 0.0},
                "joint_8": {"position": 0.0, "torque": 0.0},
                "joint_12": {"position": 0.0, "torque": 0.0},
            },
        }

    def _simulate_failed_pick_and_place(
        self,
        quality: dict,
        duration: float,
        failure_mode: str,
        place_target: list[float],
    ) -> dict:
        """Simulate a failed pick-and-place attempt."""
        if failure_mode == "transport_drop":
            fail_time = self._rng.uniform(3.5, 5.0)
            logs = [
                f"[{0.0:.2f}s] Initializing pick-and-place sequence",
                f"[{0.3:.2f}s] Phase 1 APPROACH: Moving to pre-grasp position",
                f"[{1.0:.2f}s] Phase 2 DESCEND: Lowering to object",
                f"[{1.5:.2f}s] Phase 3 GRASP: Contact detected, closing fingers",
                f"[{2.5:.2f}s] Phase 4 LIFT: Raising object",
                f"[{3.0:.2f}s] Phase 5 TRANSPORT: Moving to place target",
                f"[{fail_time:.2f}s] WARNING: Object slipping during transport",
                f"[{fail_time + 0.2:.2f}s] WARNING: Contact force decreasing",
                f"[{fail_time + 0.4:.2f}s] ERROR: Object dropped during transport at height 0.02m",
                f"[{fail_time + 0.6:.2f}s] PICK-AND-PLACE FAILED: transport_drop - object lost during lateral movement",
            ]
            drop_x = 0.5 + (place_target[0] - 0.5) * self._rng.uniform(0.2, 0.6)
            drop_y = (place_target[1]) * self._rng.uniform(0.2, 0.6)
            trajectory = [
                {"position": [0.5, 0.0, 0.05], "timestamp": 0.0},
                {"position": [0.5, 0.0, 0.30], "timestamp": 3.0},
                {"position": [drop_x, drop_y, 0.25], "timestamp": fail_time - 0.5},
                {"position": [drop_x, drop_y, 0.02], "timestamp": fail_time},
            ]
            final_pos = [drop_x, drop_y, 0.0]

        else:  # place_miss
            logs = [
                f"[{0.0:.2f}s] Initializing pick-and-place sequence",
                f"[{0.3:.2f}s] Phase 1 APPROACH: Moving to pre-grasp position",
                f"[{1.0:.2f}s] Phase 2 DESCEND: Lowering to object",
                f"[{1.5:.2f}s] Phase 3 GRASP: Contact detected, closing fingers",
                f"[{2.5:.2f}s] Phase 4 LIFT: Raising object",
                f"[{3.5:.2f}s] Phase 5 TRANSPORT: Moving to place target",
                f"[{5.5:.2f}s] Phase 6 PLACE: Lowering object",
                f"[{6.5:.2f}s] Phase 7 RELEASE: Opening fingers",
                f"[{7.0:.2f}s] WARNING: Object placed {self._rng.uniform(0.08, 0.20):.2f}m from target",
                f"[{7.5:.2f}s] PICK-AND-PLACE FAILED: place_miss - object not accurately placed on target",
            ]
            offset = self._rng.uniform(0.08, 0.20)
            angle = self._rng.uniform(0, 6.28)
            import math as _math
            final_pos = [
                place_target[0] + offset * _math.cos(angle),
                place_target[1] + offset * _math.sin(angle),
                place_target[2] + self._rng.uniform(-0.01, 0.01),
            ]
            trajectory = [
                {"position": [0.5, 0.0, 0.05], "timestamp": 0.0},
                {"position": [0.5, 0.0, 0.30], "timestamp": 3.0},
                {"position": [place_target[0], place_target[1], 0.30], "timestamp": 5.0},
                {"position": final_pos, "timestamp": 7.0},
            ]

        num_frames = int(duration * 10)
        frames = [self._generate_frame(i, num_frames, success=False) for i in range(num_frames)]

        return {
            "logs": logs,
            "frames": frames,
            "object_state": {
                "position": final_pos,
                "velocity": [
                    self._rng.uniform(-0.05, 0.05),
                    self._rng.uniform(-0.05, 0.05),
                    self._rng.uniform(-0.2, 0.0) if failure_mode == "transport_drop" else 0.0,
                ],
                "angular_velocity": [
                    self._rng.uniform(-0.3, 0.3),
                    self._rng.uniform(-0.3, 0.3),
                    self._rng.uniform(-0.3, 0.3),
                ],
                "contact_count": 0,
            },
            "object_trajectory": trajectory,
            "contact_forces": [],
            "joint_states": {
                "panda_joint1": {"position": 0.0, "torque": 0.0},
                "panda_joint2": {"position": -0.3, "torque": 1.5},
                "panda_joint4": {"position": -1.8, "torque": 2.0},
                "joint_0": {"position": 0.0, "torque": 0.0},
                "joint_4": {"position": 0.0, "torque": 0.0},
                "joint_8": {"position": 0.0, "torque": 0.0},
                "joint_12": {"position": 0.0, "torque": 0.0},
            },
        }

    def _generate_frame(
        self, frame_idx: int, total_frames: int, success: bool
    ) -> bytes:
        """Generate a minimal 4x4 PNG placeholder frame."""
        progress = frame_idx / max(1, total_frames - 1)

        if success:
            r = int(50 + 50 * (1 - progress))
            g = int(100 + 155 * progress)
            b = 80
        else:
            r = int(100 + 155 * progress)
            g = int(100 * (1 - progress))
            b = 50

        width, height = 4, 4
        raw_rows = []
        for _ in range(height):
            row = b"\x00"
            for _ in range(width):
                row += struct.pack("BBB", r, g, b)
            raw_rows.append(row)

        raw_data = b"".join(raw_rows)
        compressed = zlib.compress(raw_data)

        def _make_chunk(chunk_type: bytes, data: bytes) -> bytes:
            chunk = chunk_type + data
            crc = zlib.crc32(chunk) & 0xFFFFFFFF
            return struct.pack(">I", len(data)) + chunk + struct.pack(">I", crc)

        png = b"\x89PNG\r\n\x1a\n"
        ihdr_data = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
        png += _make_chunk(b"IHDR", ihdr_data)
        png += _make_chunk(b"IDAT", compressed)
        png += _make_chunk(b"IEND", b"")
        return png

    def reset(self):
        self._iteration_count = 0


# ---------------------------------------------------------------------------
# Isaac Sim Manager (real or mock)
# ---------------------------------------------------------------------------


class IsaacSimManager:
    """Manages Isaac Sim simulation lifecycle.

    Supports Franka Panda (7 DOF arm) + Allegro Hand (16 DOF, 4 fingers).
    Falls back to MockSimulationEngine when Isaac Sim is not available.
    """

    # Franka + Allegro USD paths (relative to Nucleus assets root)
    FRANKA_USD = "/Isaac/Robots/Franka/franka_alt_fingers.usd"
    ALLEGRO_USD = "/Isaac/Robots/AllegroHand/allegro_hand.usd"

    # Allegro fingertip link names for contact sensing
    FINGERTIP_LINKS = [
        "link_3_0_tip",   # index
        "link_7_0_tip",   # middle
        "link_11_0_tip",  # ring
        "link_15_0_tip",  # thumb
    ]
    FINGER_NAMES = ["index", "middle", "ring", "thumb"]

    # Physics config
    PHYSICS_DT = 1.0 / 120.0  # 120 Hz physics
    RENDER_EVERY_N = 4  # render every 4 physics steps = 30 fps

    def __init__(self):
        self.world = None
        self.robot_articulation = None
        self.target_object = None
        self.contact_sensors: list = []
        self.camera = None
        self.frames: list = []
        self.logs: list = []
        self._mock_engine = MockSimulationEngine()
        self._assets_root: str | None = None

    async def initialize(self, headless: bool = True):
        if not ISAAC_SIM_AVAILABLE:
            logger.info("Mock initialization (headless=%s)", headless)
            sim_state["initialized"] = True
            sim_state["headless"] = headless
            return True

        try:
            self.world = World(
                stage_units_in_meters=1.0,
                physics_dt=self.PHYSICS_DT,
                rendering_dt=self.PHYSICS_DT * self.RENDER_EVERY_N,
            )
            await self.world.initialize_simulation_context_async()

            # Enable PhysX contact reporting
            stage = self.world.stage
            physx_scene = PhysxSchema.PhysxSceneAPI.Apply(
                stage.GetPrimAtPath("/physicsScene")
            )
            physx_scene.GetEnableGPUDynamicsAttr().Set(True)
            physx_scene.GetBroadphaseTypeAttr().Set("GPU")

            self._assets_root = get_assets_root_path()
            logger.info("Assets root: %s", self._assets_root)

            sim_state["initialized"] = True
            sim_state["headless"] = headless
            logger.info("Isaac Sim world initialized (dt=%.4f)", self.PHYSICS_DT)
            return True
        except Exception as e:
            logger.error("Failed to initialize: %s", e, exc_info=True)
            return False

    async def load_scene(self, config: SceneConfig):
        if not ISAAC_SIM_AVAILABLE:
            sim_state["scene_loaded"] = True
            if config.enable_pick_and_place:
                sim_state["pick_table_position"] = config.pick_table_position
                sim_state["place_tray_position"] = config.place_tray_position
                self.logs.append(
                    f"Scene loaded with pick table at {config.pick_table_position} "
                    f"and place tray at {config.place_tray_position} (mock mode)"
                )
            else:
                self.logs.append("Scene loaded (mock mode)")
            return True

        try:
            self.world.scene.add_default_ground_plane()

            # Add distant light for rendering
            stage = self.world.stage
            light_prim = stage.DefinePrim("/World/DistantLight", "DistantLight")
            light_prim.GetAttribute("inputs:intensity").Set(3000.0)
            light_prim.GetAttribute("inputs:angle").Set(0.53)
            UsdGeom.Xformable(light_prim).AddRotateXYZOp().Set(Gf.Vec3f(-45, 0, 0))

            # Add virtual camera for frame capture
            cam_prim = stage.DefinePrim("/World/Camera", "Camera")
            UsdGeom.Xformable(cam_prim).AddTranslateOp().Set(Gf.Vec3d(1.5, 0.8, 0.8))
            UsdGeom.Xformable(cam_prim).AddRotateXYZOp().Set(Gf.Vec3f(-25, 55, 0))
            cam_prim.GetAttribute("focalLength").Set(24.0)

            sim_state["scene_loaded"] = True
            self.logs.append("Scene loaded with ground plane, lighting, and camera")
            return True
        except Exception as e:
            logger.error("Failed to load scene: %s", e, exc_info=True)
            return False

    async def load_robot(self, config: RobotConfig):
        if not ISAAC_SIM_AVAILABLE:
            sim_state["robot_loaded"] = True
            sim_state["robot_model"] = config.model
            self.logs.append(f"Robot '{config.model}' loaded (mock mode)")
            return True

        try:
            assets_root = self._assets_root or get_assets_root_path()
            stage = self.world.stage

            # --- Load Franka arm ---
            franka_usd = config.usd_path or f"{assets_root}{self.FRANKA_USD}"
            add_reference_to_stage(usd_path=franka_usd, prim_path="/World/Robot")
            self.logs.append(f"Franka arm loaded from {franka_usd}")

            # --- Attach Allegro Hand to Franka end-effector ---
            allegro_usd = f"{assets_root}{self.ALLEGRO_USD}"
            hand_prim_path = "/World/Robot/panda_link8/allegro_hand"
            add_reference_to_stage(usd_path=allegro_usd, prim_path=hand_prim_path)

            # Create fixed joint between panda_link8 and allegro hand base
            hand_prim = stage.GetPrimAtPath(hand_prim_path)
            if hand_prim.IsValid():
                fixed_joint = UsdPhysics.FixedJoint.Define(
                    stage, f"{hand_prim_path}/fixed_joint_to_arm"
                )
                fixed_joint.GetBody0Rel().SetTargets(
                    [Sdf.Path("/World/Robot/panda_link8")]
                )
                fixed_joint.GetBody1Rel().SetTargets(
                    [Sdf.Path(hand_prim_path)]
                )
                self.logs.append("Allegro Hand attached to panda_link8 via fixed joint")
            else:
                logger.warning("Allegro hand prim not found at %s", hand_prim_path)

            # --- Create Articulation for the combined assembly ---
            self.robot_articulation = self.world.scene.add(
                Articulation(
                    prim_path="/World/Robot",
                    name="franka_allegro",
                    position=config.position,
                )
            )

            # --- Add contact sensors to Allegro fingertips ---
            self.contact_sensors = []
            for i, tip_link in enumerate(self.FINGERTIP_LINKS):
                sensor_path = f"{hand_prim_path}/{tip_link}/contact_sensor"
                try:
                    sensor = self.world.scene.add(
                        ContactSensor(
                            prim_path=sensor_path,
                            name=f"contact_{self.FINGER_NAMES[i]}",
                            min_threshold=0.1,
                            max_threshold=100.0,
                            radius=0.005,
                        )
                    )
                    self.contact_sensors.append(
                        {"sensor": sensor, "finger": self.FINGER_NAMES[i]}
                    )
                except Exception as sensor_err:
                    logger.warning(
                        "Contact sensor for %s failed: %s", tip_link, sensor_err
                    )

            self.logs.append(
                f"Contact sensors added: {len(self.contact_sensors)}/{len(self.FINGERTIP_LINKS)}"
            )

            # Let the world pick up new prims
            await self.world.reset_async()

            sim_state["robot_loaded"] = True
            sim_state["robot_model"] = config.model
            self.logs.append(
                f"Robot '{config.model}' loaded: Franka (7 DOF) + Allegro (16 DOF)"
            )
            return True

        except Exception as e:
            logger.error("Failed to load robot: %s", e, exc_info=True)
            return False

    async def load_object(self, config: ObjectConfig):
        if not ISAAC_SIM_AVAILABLE:
            sim_state["object_loaded"] = True
            sim_state["object_path"] = config.cad_file_path
            sim_state["object_position"] = config.position
            self.logs.append(f"Object loaded from {config.cad_file_path} (mock mode)")
            return True

        try:
            cad_path = config.cad_file_path
            prim_path = "/World/TargetObject"

            if cad_path.lower().endswith((".stl", ".obj")):
                self._load_stl_as_mesh_prim(
                    cad_path, prim_path, config.position, config.scale
                )
                self.logs.append(f"STL mesh loaded from {cad_path}")
            else:
                self.world.scene.add(
                    RigidPrim(
                        prim_path=prim_path,
                        name="target_object",
                        usd_path=cad_path,
                        position=config.position,
                        scale=config.scale,
                    )
                )

            self.target_object = RigidPrim(
                prim_path=prim_path, name="target_object"
            )

            await self.world.reset_async()

            sim_state["object_loaded"] = True
            sim_state["object_path"] = cad_path
            sim_state["object_position"] = config.position
            self.logs.append(f"Object loaded at position {config.position}")
            return True

        except Exception as e:
            logger.error("Failed to load object: %s", e, exc_info=True)
            return False

    def _load_stl_as_mesh_prim(self, mesh_path: str, prim_path: str, position, scale):
        """Load an STL/OBJ file and create a UsdGeom.Mesh prim directly on the stage."""
        import trimesh as _trimesh

        mesh = _trimesh.load(mesh_path)
        if hasattr(mesh, "geometry"):
            # Scene with multiple meshes — concatenate
            mesh = _trimesh.util.concatenate(list(mesh.geometry.values()))

        stage = self.world.stage
        mesh_prim = UsdGeom.Mesh.Define(stage, prim_path)

        # Set vertices
        points = [Gf.Vec3f(float(v[0]), float(v[1]), float(v[2])) for v in mesh.vertices]
        mesh_prim.GetPointsAttr().Set(points)

        # Set faces (all triangles)
        face_counts = [3] * len(mesh.faces)
        face_indices = mesh.faces.flatten().tolist()
        mesh_prim.GetFaceVertexCountsAttr().Set(face_counts)
        mesh_prim.GetFaceVertexIndicesAttr().Set(face_indices)

        # Set normals if available
        if mesh.vertex_normals is not None and len(mesh.vertex_normals) > 0:
            normals = [Gf.Vec3f(float(n[0]), float(n[1]), float(n[2])) for n in mesh.vertex_normals]
            mesh_prim.GetNormalsAttr().Set(normals)

        # Transform
        xf = UsdGeom.Xformable(mesh_prim.GetPrim())
        xf.AddTranslateOp().Set(Gf.Vec3d(float(position[0]), float(position[1]), float(position[2])))
        xf.AddScaleOp().Set(Gf.Vec3d(float(scale[0]), float(scale[1]), float(scale[2])))

        # Physics: rigid body + collision + mass
        prim = mesh_prim.GetPrim()
        UsdPhysics.RigidBodyAPI.Apply(prim)
        UsdPhysics.CollisionAPI.Apply(prim)

        # Use mesh collision approximation for performance
        collision_api = PhysxSchema.PhysxCollisionAPI.Apply(prim)

        mass_api = UsdPhysics.MassAPI.Apply(prim)
        mass_api.GetMassAttr().Set(0.5)  # 500g default

        logger.info(
            "STL mesh loaded: %d vertices, %d faces at %s",
            len(mesh.vertices), len(mesh.faces), prim_path,
        )
        return prim

    def _capture_frame(self) -> bytes | None:
        """Capture a single RGB frame from the simulation camera."""
        try:
            from omni.isaac.sensor import Camera as IsaacCamera

            if self.camera is None:
                self.camera = IsaacCamera(
                    prim_path="/World/Camera",
                    resolution=(640, 480),
                )
                self.camera.initialize()

            self.camera.get_current_frame()
            rgba = self.camera.get_rgba()
            if rgba is None:
                return None

            # Convert RGBA numpy array to PNG bytes
            import io
            from PIL import Image

            img = Image.fromarray(rgba[:, :, :3])
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            return buf.getvalue()
        except Exception as e:
            logger.debug("Frame capture failed: %s", e)
            return None

    def _read_contact_forces(self) -> list[dict]:
        """Read contact forces from all fingertip sensors."""
        forces = []
        for cs in self.contact_sensors:
            try:
                reading = cs["sensor"].get_current_frame()
                force_val = float(reading.get("value", 0.0))
                if force_val > 0.05:
                    forces.append({
                        "finger": cs["finger"],
                        "force_n": force_val,
                    })
            except Exception:
                pass
        return forces

    def _read_joint_states(self) -> dict:
        """Read joint positions and torques from the articulation."""
        states = {}
        if self.robot_articulation is None:
            return states

        try:
            # Try tensor API first (requires active physics scene via world.reset())
            if self.robot_articulation.dof_names is not None:
                joint_positions = self.robot_articulation.get_joint_positions()
                joint_efforts = self.robot_articulation.get_applied_joint_efforts()
                dof_names = self.robot_articulation.dof_names
                for i, name in enumerate(dof_names):
                    pos = float(joint_positions[i]) if joint_positions is not None else 0.0
                    torque = float(joint_efforts[i]) if joint_efforts is not None else 0.0
                    states[name] = {"position": pos, "torque": abs(torque)}
            else:
                # Fallback: read joint positions from USD stage directly.
                # This works without tensor API initialization.
                stage = self.world.stage
                robot_prim = stage.GetPrimAtPath("/World/Robot")
                if robot_prim.IsValid():
                    self._collect_joint_states_usd(stage, robot_prim, states)
                    logger.debug("Read %d joints from USD stage", len(states))
        except Exception as e:
            logger.debug("Failed to read joint states: %s", e)

        return states

    def _collect_joint_states_usd(self, stage, root_prim, states: dict):
        """Walk the USD prim hierarchy and collect revolute/prismatic joint states."""
        from pxr import UsdPhysics as _UsdPhysics

        for prim in Usd.PrimRange(root_prim):
            if prim.IsA(_UsdPhysics.RevoluteJoint) or prim.IsA(_UsdPhysics.PrismaticJoint):
                name = prim.GetName()
                # Read current position from the "state:position" attribute
                pos_attr = prim.GetAttribute("state:position")
                pos = float(pos_attr.Get()) if pos_attr and pos_attr.HasValue() else 0.0
                # Read drive force/torque if available
                torque_attr = prim.GetAttribute("state:effort")
                torque = float(torque_attr.Get()) if torque_attr and torque_attr.HasValue() else 0.0
                states[name] = {"position": pos, "torque": abs(torque)}

            # Also check for PhysicsJoint (generic) with drive
            if prim.HasAPI(_UsdPhysics.DriveAPI):
                name = prim.GetName()
                if name not in states:
                    target_attr = prim.GetAttribute("drive:angular:physics:targetPosition")
                    pos = float(target_attr.Get()) if target_attr and target_attr.HasValue() else 0.0
                    states[name] = {"position": pos, "torque": 0.0}

    def _read_object_state(self) -> dict:
        """Read full object state: position, velocity, angular velocity."""
        default_state = {
            "position": [0.0, 0.0, 0.0],
            "velocity": [0.0, 0.0, 0.0],
            "angular_velocity": [0.0, 0.0, 0.0],
            "contact_count": 0,
        }
        if self.target_object is None:
            return default_state

        try:
            pos, _ = self.target_object.get_world_pose()
            vel = self.target_object.get_linear_velocity()
            ang_vel = self.target_object.get_angular_velocity()
            contacts = self._read_contact_forces()

            return {
                "position": pos.tolist() if hasattr(pos, "tolist") else list(pos),
                "velocity": vel.tolist() if hasattr(vel, "tolist") else list(vel),
                "angular_velocity": (
                    ang_vel.tolist() if hasattr(ang_vel, "tolist") else list(ang_vel)
                ),
                "contact_count": len(contacts),
            }
        except Exception as e:
            logger.debug("Failed to read object state: %s", e)
            return default_state

    async def execute_grasp_code(
        self, code: str, timeout: float = 30.0, place_target: list[float] | None = None
    ) -> dict:
        """Execute generated grasping code in the simulation.

        The generated code must define a function:
            run_grasp_simulation(sim_context, robot_prim_path, object_prim_path, place_target=None) -> dict

        This method:
        1. Executes the code to define the function
        2. Calls the function with the simulation context
        3. Reads final object state, contact forces, and joint states
        4. Returns structured results matching GraspValidator expectations
        """
        start_time = time.time()
        self.frames = []
        self.logs = []

        if not ISAAC_SIM_AVAILABLE:
            return self._mock_engine.execute(code, timeout, place_target=place_target)

        try:
            import numpy as _np

            # Build the execution namespace with Isaac Sim APIs
            exec_namespace = {
                "__builtins__": __builtins__,
                "np": _np,
                "numpy": _np,
                "time": __import__("time"),
                "math": __import__("math"),
                "logger": logger,
                # Isaac Sim APIs
                "World": World,
                "Articulation": Articulation,
                "RigidPrim": RigidPrim,
                "XFormPrim": XFormPrim,
            }

            sim_state["running"] = True
            self.logs.append(f"[0.00s] Executing generated grasping code")

            # Step 1: exec the code to define run_grasp_simulation
            exec(code, exec_namespace)

            grasp_fn = exec_namespace.get("run_grasp_simulation")
            if grasp_fn is None:
                raise RuntimeError(
                    "Generated code must define run_grasp_simulation("
                    "sim_context, robot_prim_path, object_prim_path)"
                )

            # Step 2: Call the grasp function
            self.logs.append(f"[{time.time() - start_time:.2f}s] Calling run_grasp_simulation")
            try:
                grasp_result = grasp_fn(
                    self.world,
                    "/World/Robot",
                    "/World/TargetObject",
                    place_target=place_target,
                )
            except TypeError:
                grasp_result = grasp_fn(
                    self.world,
                    "/World/Robot",
                    "/World/TargetObject",
                )
            self.logs.append(
                f"[{time.time() - start_time:.2f}s] run_grasp_simulation returned"
            )

            # Step 3: Run additional physics steps to settle
            settle_steps = int(1.0 / self.PHYSICS_DT)  # 1 second of settling
            for step in range(settle_steps):
                self.world.step(render=(step % self.RENDER_EVERY_N == 0))

                # Capture frames periodically
                if step % (self.RENDER_EVERY_N * 10) == 0:
                    frame = self._capture_frame()
                    if frame:
                        self.frames.append(frame)

            # Step 4: Read final state
            object_state = self._read_object_state()
            contact_forces = self._read_contact_forces()
            joint_states = self._read_joint_states()
            duration = time.time() - start_time

            self.logs.append(
                f"[{duration:.2f}s] Simulation complete. "
                f"Object at z={object_state['position'][2]:.3f}m, "
                f"{len(contact_forces)} contacts"
            )

            # Encode frames as base64
            frames_b64 = [
                base64.b64encode(f).decode("ascii") for f in self.frames
            ]

            sim_state["running"] = False

            return {
                "success": True,
                "duration": duration,
                "object_final_state": object_state,
                "contact_forces": contact_forces,
                "joint_states": joint_states,
                "logs": self.logs,
                "frames": frames_b64,
                "error": None,
            }

        except Exception as e:
            sim_state["running"] = False
            duration = time.time() - start_time
            self.logs.append(f"[{duration:.2f}s] ERROR: {e}")
            logger.error("Grasp code execution failed: %s", e, exc_info=True)

            # Still try to read object state for diagnostics
            object_state = self._read_object_state()

            return {
                "success": False,
                "duration": duration,
                "object_final_state": object_state,
                "contact_forces": [],
                "joint_states": self._read_joint_states(),
                "logs": self.logs,
                "frames": [],
                "error": str(e),
            }

    async def reset(self):
        """Reset the simulation state."""
        if ISAAC_SIM_AVAILABLE and self.world:
            await self.world.reset_async()
        self.robot_articulation = None
        self.target_object = None
        self.contact_sensors = []
        self.camera = None
        self.frames = []
        self.logs = []
        self._mock_engine.reset()
        sim_state["scene_loaded"] = False
        sim_state["robot_loaded"] = False
        sim_state["object_loaded"] = False
        sim_state["running"] = False
        sim_state["robot_model"] = None
        sim_state["object_path"] = None

    # ------------------------------------------------------------------
    # Synchronous methods for main-thread dispatch (standalone mode)
    # ------------------------------------------------------------------

    def _load_scene_sync(self, config: SceneConfig) -> bool:
        """Synchronous scene loading — runs on main thread.

        NOTE: Do NOT call _SIMULATION_APP.update() or world.reset() here.
        The main loop handles updates; calling them from the command queue
        causes 'Cannot run the event loop while another loop is running'.
        """
        try:
            stage = self.world.stage

            # Remove existing scene prims to avoid duplicate xformOps
            for path in ["/World/DistantLight", "/World/Camera",
                         "/World/PickTable", "/World/PlaceTray"]:
                prim = stage.GetPrimAtPath(path)
                if prim.IsValid():
                    stage.RemovePrim(path)

            self.world.scene.add_default_ground_plane()

            light_prim = stage.DefinePrim("/World/DistantLight", "DistantLight")
            light_prim.GetAttribute("inputs:intensity").Set(3000.0)
            light_prim.GetAttribute("inputs:angle").Set(0.53)
            UsdGeom.Xformable(light_prim).AddRotateXYZOp().Set(Gf.Vec3f(-45, 0, 0))

            cam_prim = stage.DefinePrim("/World/Camera", "Camera")
            UsdGeom.Xformable(cam_prim).AddTranslateOp().Set(Gf.Vec3d(1.5, 0.8, 0.8))
            UsdGeom.Xformable(cam_prim).AddRotateXYZOp().Set(Gf.Vec3f(-25, 55, 0))
            cam_prim.GetAttribute("focalLength").Set(24.0)

            if config.enable_pick_and_place:
                self._add_pick_table(stage, config.pick_table_position)
                self._add_place_tray(stage, config.place_tray_position)
                self.logs.append(
                    f"Scene loaded with pick table at {config.pick_table_position} "
                    f"and place tray at {config.place_tray_position}"
                )

            # Let physics pick up new prims via stepping (safe from command queue)
            for _ in range(5):
                self.world.step(render=False)

            sim_state["scene_loaded"] = True
            if config.enable_pick_and_place:
                sim_state["pick_table_position"] = config.pick_table_position
                sim_state["place_tray_position"] = config.place_tray_position
            self.logs.append("Scene loaded with ground plane, lighting, and camera")
            logger.info("Scene loaded successfully (sync)")
            return True
        except Exception as e:
            logger.error("Failed to load scene (sync): %s", e, exc_info=True)
            return False

    def _add_pick_table(self, stage, position: list[float]):
        """Add a pick table (flat box) to the scene."""
        table_path = "/World/PickTable"
        table_prim = stage.DefinePrim(table_path, "Cube")
        xf = UsdGeom.Xformable(table_prim)
        xf.AddTranslateOp().Set(Gf.Vec3d(*position))
        xf.AddScaleOp().Set(Gf.Vec3d(0.4, 0.3, 0.02))
        UsdPhysics.CollisionAPI.Apply(table_prim)
        logger.info("Pick table added at %s", position)

    def _add_place_tray(self, stage, position: list[float]):
        """Add a place tray (box with raised edges) to the scene."""
        tray_path = "/World/PlaceTray"
        base_prim = stage.DefinePrim(f"{tray_path}/Base", "Cube")
        xf = UsdGeom.Xformable(base_prim)
        xf.AddTranslateOp().Set(Gf.Vec3d(position[0], position[1], position[2]))
        xf.AddScaleOp().Set(Gf.Vec3d(0.15, 0.15, 0.01))
        UsdPhysics.CollisionAPI.Apply(base_prim)

        wall_h = 0.03
        walls = [
            ("WallFront", [position[0], position[1] - 0.15, position[2] + wall_h], [0.15, 0.005, wall_h]),
            ("WallBack", [position[0], position[1] + 0.15, position[2] + wall_h], [0.15, 0.005, wall_h]),
            ("WallLeft", [position[0] - 0.15, position[1], position[2] + wall_h], [0.005, 0.15, wall_h]),
            ("WallRight", [position[0] + 0.15, position[1], position[2] + wall_h], [0.005, 0.15, wall_h]),
        ]
        for name, pos, scale in walls:
            wall_prim = stage.DefinePrim(f"{tray_path}/{name}", "Cube")
            wxf = UsdGeom.Xformable(wall_prim)
            wxf.AddTranslateOp().Set(Gf.Vec3d(*pos))
            wxf.AddScaleOp().Set(Gf.Vec3d(*scale))
            UsdPhysics.CollisionAPI.Apply(wall_prim)
        logger.info("Place tray added at %s", position)

    def _load_robot_sync(self, config: RobotConfig) -> bool:
        """Synchronous robot loading — runs on main thread."""
        try:
            assets_root = self._assets_root or get_assets_root_path()
            stage = self.world.stage

            # Remove existing robot prim and scene registry entry
            robot_prim = stage.GetPrimAtPath("/World/Robot")
            if robot_prim.IsValid():
                stage.RemovePrim("/World/Robot")
            # Clear names from World scene registry to avoid "not unique" error.
            # Use registry_only=True — prims already removed above via RemovePrim.
            for name in ["franka_allegro", "contact_thumb", "contact_index",
                         "contact_middle", "contact_ring"]:
                try:
                    self.world.scene.remove_object(name, registry_only=True)
                except Exception:
                    pass

            franka_usd = config.usd_path or f"{assets_root}{self.FRANKA_USD}"
            add_reference_to_stage(usd_path=franka_usd, prim_path="/World/Robot")
            self.logs.append(f"Franka arm loaded from {franka_usd}")

            allegro_usd = f"{assets_root}{self.ALLEGRO_USD}"
            hand_prim_path = "/World/Robot/panda_link8/allegro_hand"
            add_reference_to_stage(usd_path=allegro_usd, prim_path=hand_prim_path)

            hand_prim = stage.GetPrimAtPath(hand_prim_path)
            if hand_prim.IsValid():
                fixed_joint = UsdPhysics.FixedJoint.Define(
                    stage, f"{hand_prim_path}/fixed_joint_to_arm"
                )
                fixed_joint.GetBody0Rel().SetTargets(
                    [Sdf.Path("/World/Robot/panda_link8")]
                )
                fixed_joint.GetBody1Rel().SetTargets(
                    [Sdf.Path(hand_prim_path)]
                )
                self.logs.append("Allegro Hand attached to panda_link8 via fixed joint")

            self.robot_articulation = self.world.scene.add(
                Articulation(
                    prim_path="/World/Robot",
                    name="franka_allegro",
                    position=config.position,
                )
            )

            self.contact_sensors = []
            for i, tip_link in enumerate(self.FINGERTIP_LINKS):
                sensor_path = f"{hand_prim_path}/{tip_link}/contact_sensor"
                try:
                    sensor = self.world.scene.add(
                        ContactSensor(
                            prim_path=sensor_path,
                            name=f"contact_{self.FINGER_NAMES[i]}",
                            min_threshold=0.1,
                            max_threshold=100.0,
                            radius=0.005,
                        )
                    )
                    self.contact_sensors.append(
                        {"sensor": sensor, "finger": self.FINGER_NAMES[i]}
                    )
                except Exception as sensor_err:
                    logger.warning(
                        "Contact sensor for %s failed: %s", tip_link, sensor_err
                    )

            self.logs.append(
                f"Contact sensors added: {len(self.contact_sensors)}/{len(self.FINGERTIP_LINKS)}"
            )

            # Step physics to pick up new prims (NOT world.reset — causes event loop conflict)
            for _ in range(5):
                self.world.step(render=False)

            sim_state["robot_loaded"] = True
            sim_state["robot_model"] = config.model
            self.logs.append(
                f"Robot '{config.model}' loaded: Franka (7 DOF) + Allegro (16 DOF)"
            )
            logger.info("Robot loaded successfully (sync)")
            return True
        except Exception as e:
            logger.error("Failed to load robot (sync): %s", e, exc_info=True)
            return False

    def _load_object_sync(self, config: ObjectConfig) -> bool:
        """Synchronous object loading — runs on main thread."""
        try:
            cad_path = config.cad_file_path
            prim_path = "/World/TargetObject"

            # Remove existing object prim and scene registry entry.
            # Use registry_only=True — prim removed via RemovePrim below.
            stage = self.world.stage
            obj_prim = stage.GetPrimAtPath(prim_path)
            if obj_prim.IsValid():
                stage.RemovePrim(prim_path)
            try:
                self.world.scene.remove_object("target_object", registry_only=True)
            except Exception:
                pass

            if cad_path.lower().endswith((".stl", ".obj")):
                # Load STL/OBJ directly as UsdGeom.Mesh
                self._load_stl_as_mesh_prim(
                    cad_path, prim_path, config.position, config.scale
                )
                self.logs.append(f"STL mesh loaded from {cad_path}")
            else:
                # USD file — use RigidPrim directly
                self.world.scene.add(
                    RigidPrim(
                        prim_path=prim_path,
                        name="target_object",
                        usd_path=cad_path,
                        position=config.position,
                        scale=config.scale,
                    )
                )

            # Wrap the prim for state reading
            self.target_object = RigidPrim(
                prim_path=prim_path, name="target_object"
            )

            # Step physics to pick up new prims (NOT world.reset — causes event loop conflict)
            for _ in range(5):
                self.world.step(render=False)

            sim_state["object_loaded"] = True
            sim_state["object_path"] = cad_path
            sim_state["object_position"] = config.position
            self.logs.append(f"Object loaded at position {config.position}")
            logger.info("Object loaded successfully (sync)")
            return True
        except Exception as e:
            logger.error("Failed to load object (sync): %s", e, exc_info=True)
            return False

    def _execute_grasp_code_sync(
        self, code: str, timeout: float = 30.0, place_target: list[float] | None = None
    ) -> dict:
        """Synchronous grasp code execution — runs on main thread."""
        start_time = time.time()
        self.frames = []
        self.logs = []

        try:
            import numpy as _np

            exec_namespace = {
                "__builtins__": __builtins__,
                "np": _np,
                "numpy": _np,
                "time": __import__("time"),
                "math": __import__("math"),
                "logger": logger,
                "World": World,
                "Articulation": Articulation,
                "RigidPrim": RigidPrim,
                "XFormPrim": XFormPrim,
            }

            sim_state["running"] = True
            self.logs.append("[0.00s] Executing generated grasping code")

            exec(code, exec_namespace)

            grasp_fn = exec_namespace.get("run_grasp_simulation")
            if grasp_fn is None:
                raise RuntimeError(
                    "Generated code must define run_grasp_simulation("
                    "sim_context, robot_prim_path, object_prim_path)"
                )

            self.logs.append(f"[{time.time() - start_time:.2f}s] Calling run_grasp_simulation")
            try:
                grasp_result = grasp_fn(
                    self.world,
                    "/World/Robot",
                    "/World/TargetObject",
                    place_target=place_target,
                )
            except TypeError:
                grasp_result = grasp_fn(
                    self.world,
                    "/World/Robot",
                    "/World/TargetObject",
                )
            self.logs.append(
                f"[{time.time() - start_time:.2f}s] run_grasp_simulation returned"
            )

            settle_steps = int(1.0 / self.PHYSICS_DT)
            for step in range(settle_steps):
                self.world.step(render=(step % self.RENDER_EVERY_N == 0))
                if step % (self.RENDER_EVERY_N * 10) == 0:
                    frame = self._capture_frame()
                    if frame:
                        self.frames.append(frame)

            object_state = self._read_object_state()
            contact_forces = self._read_contact_forces()
            joint_states = self._read_joint_states()
            duration = time.time() - start_time

            self.logs.append(
                f"[{duration:.2f}s] Simulation complete. "
                f"Object at z={object_state['position'][2]:.3f}m, "
                f"{len(contact_forces)} contacts"
            )

            frames_b64 = [
                base64.b64encode(f).decode("ascii") for f in self.frames
            ]

            sim_state["running"] = False

            return {
                "success": True,
                "duration": duration,
                "object_final_state": object_state,
                "contact_forces": contact_forces,
                "joint_states": joint_states,
                "logs": self.logs,
                "frames": frames_b64,
                "error": None,
            }

        except Exception as e:
            sim_state["running"] = False
            duration = time.time() - start_time
            self.logs.append(f"[{duration:.2f}s] ERROR: {e}")
            logger.error("Grasp code execution failed (sync): %s", e, exc_info=True)
            object_state = self._read_object_state()

            return {
                "success": False,
                "duration": duration,
                "object_final_state": object_state,
                "contact_forces": [],
                "joint_states": self._read_joint_states(),
                "logs": self.logs,
                "frames": [],
                "error": str(e),
            }

    def _reset_sync(self):
        """Synchronous reset — runs on main thread.

        Clears scene prims manually instead of world.reset() to avoid
        event loop conflicts with Isaac Sim's async engine.
        """
        if self.world:
            try:
                stage = self.world.stage
                # Remove scene prims (keep /physicsScene and default ground)
                for path in [
                    "/World/Robot", "/World/TargetObject", "/World/DistantLight",
                    "/World/Camera", "/World/PickTable", "/World/PlaceTray",
                ]:
                    prim = stage.GetPrimAtPath(path)
                    if prim.IsValid():
                        stage.RemovePrim(path)
                # Clear World scene registry — use registry_only=True because
                # prims are already removed from stage above.  Without this flag,
                # remove_object() returns early when it can't find the USD prim.
                for name in ["franka_allegro", "target_object"]:
                    try:
                        self.world.scene.remove_object(name, registry_only=True)
                    except Exception:
                        pass
                # Also clear contact sensor names
                for cs in self.contact_sensors:
                    try:
                        self.world.scene.remove_object(cs["sensor"].name, registry_only=True)
                    except Exception:
                        pass

                # Step to process removals
                for _ in range(5):
                    self.world.step(render=False)
            except Exception as e:
                logger.warning("Reset stage cleanup error: %s", e)

        self.robot_articulation = None
        self.target_object = None
        self.contact_sensors = []
        self.camera = None
        self.frames = []
        self.logs = []
        self._mock_engine.reset()
        sim_state["scene_loaded"] = False
        sim_state["robot_loaded"] = False
        sim_state["object_loaded"] = False
        sim_state["running"] = False
        sim_state["robot_model"] = None
        sim_state["object_path"] = None


# Global sim manager
sim_manager = IsaacSimManager()


async def _run_on_main(fn):
    """Dispatch fn to the main thread via command queue, return result.

    In standalone mode (SimulationApp running), Isaac Sim operations must
    execute on the main thread. This helper bridges the async API handler
    to the synchronous main-thread command queue.
    """
    future = app.state.run_on_main(fn)
    return await asyncio.wrap_future(future)


def _is_standalone():
    """True when running with SimulationApp (Isaac Sim available on main thread)."""
    return _SIMULATION_APP is not None and ISAAC_SIM_AVAILABLE


@app.on_event("startup")
async def startup():
    if _SIMULATION_APP is not None:
        logger.info("Startup: SimulationApp active, World already created on main thread")
        sim_state["initialized"] = True
    else:
        await sim_manager.initialize()


@app.get("/health")
async def health():
    return {"status": "ok", "isaac_sim_available": ISAAC_SIM_AVAILABLE, "state": sim_state}


@app.post("/init")
async def init_sim(config: InitRequest = InitRequest()):
    if _is_standalone():
        # World already created in _start_with_isaac_sim
        return {"status": "initialized", "headless": config.headless}
    success = await sim_manager.initialize(headless=config.headless)
    if not success:
        raise HTTPException(500, "Failed to initialize simulation")
    return {"status": "initialized", "headless": config.headless}


@app.post("/scene/load")
async def load_scene(config: SceneConfig = SceneConfig()):
    if _is_standalone():
        success = await _run_on_main(lambda: sim_manager._load_scene_sync(config))
    else:
        success = await sim_manager.load_scene(config)
    if not success:
        raise HTTPException(500, "Failed to load scene")
    return {"status": "scene_loaded"}


@app.post("/robot/load")
async def load_robot(config: RobotConfig):
    if _is_standalone():
        success = await _run_on_main(lambda: sim_manager._load_robot_sync(config))
    else:
        success = await sim_manager.load_robot(config)
    if not success:
        raise HTTPException(500, "Failed to load robot")
    return {"status": "robot_loaded", "model": config.model}


@app.post("/object/load")
async def load_object(config: ObjectConfig):
    if _is_standalone():
        success = await _run_on_main(lambda: sim_manager._load_object_sync(config))
    else:
        success = await sim_manager.load_object(config)
    if not success:
        raise HTTPException(500, "Failed to load object")
    return {"status": "object_loaded"}


@app.post("/execute")
async def execute_code(request: ExecuteRequest):
    if _is_standalone():
        result = await _run_on_main(
            lambda: sim_manager._execute_grasp_code_sync(
                request.code, request.timeout, place_target=request.place_target
            )
        )
    else:
        result = await sim_manager.execute_grasp_code(
            request.code, request.timeout, place_target=request.place_target
        )
    return result


@app.get("/livestream/status")
async def livestream_status():
    """Return WebRTC livestream availability info."""
    webrtc_available = False
    try:
        import omni.kit.livestream.webrtc  # noqa: F401
        webrtc_available = True
    except ImportError:
        pass
    return {
        "webrtc_available": webrtc_available,
        "signaling_port": 8211,
        "media_port": 49100,
        "isaac_sim_available": ISAAC_SIM_AVAILABLE,
    }


@app.post("/reset")
async def reset_sim():
    if _is_standalone():
        await _run_on_main(lambda: sim_manager._reset_sync())
    else:
        await sim_manager.reset()
    return {"status": "reset_complete"}


@app.get("/state")
async def get_state():
    return sim_state


def _start_with_isaac_sim():
    """Start with Isaac Sim on main thread, uvicorn on background thread."""
    global _SIMULATION_APP, ISAAC_SIM_AVAILABLE

    from isaacsim import SimulationApp

    sim_config = {
        "headless": True,
        "renderer": "RayTracedLighting",
        "width": 1280,
        "height": 720,
    }
    _SIMULATION_APP = SimulationApp(sim_config)
    logger.info("SimulationApp initialized (headless=True)")

    # Now re-import Isaac Sim modules that require SimulationApp
    try:
        global World, add_reference_to_stage, XFormPrim, RigidPrim
        global Robot, Articulation, get_assets_root_path, ContactSensor
        global physics_utils, UsdPhysics, PhysxSchema, Gf, Sdf, UsdGeom, Usd, np

        from omni.isaac.core import World as _World
        from omni.isaac.core.utils.stage import add_reference_to_stage as _arts
        from omni.isaac.core.prims import XFormPrim as _XFP, RigidPrim as _RP
        from omni.isaac.core.robots import Robot as _Robot
        from omni.isaac.core.articulations import Articulation as _Art
        from omni.isaac.core.utils.nucleus import get_assets_root_path as _garp
        from omni.isaac.sensor import ContactSensor as _CS
        import omni.isaac.core.utils.physics as _pu
        from pxr import UsdPhysics as _UP, PhysxSchema as _PS, Gf as _Gf
        from pxr import Sdf as _Sdf, UsdGeom as _UG, Usd as _Usd
        import numpy as _np

        World = _World
        add_reference_to_stage = _arts
        XFormPrim = _XFP
        RigidPrim = _RP
        Robot = _Robot
        Articulation = _Art
        get_assets_root_path = _garp
        ContactSensor = _CS
        physics_utils = _pu
        UsdPhysics = _UP
        PhysxSchema = _PS
        Gf = _Gf
        Sdf = _Sdf
        UsdGeom = _UG
        Usd = _Usd
        np = _np

        ISAAC_SIM_AVAILABLE = True
        logger.info("Isaac Sim Python API loaded successfully (standalone mode)")
    except ImportError as e:
        logger.warning("Isaac Sim modules failed to load: %s", e)

    # Command queue: API thread -> main thread
    _cmd_queue: queue.Queue[tuple[callable, Future]] = queue.Queue()

    def run_on_main_thread(fn):
        """Schedule a callable to run on the main thread and return a Future."""
        future = Future()
        _cmd_queue.put((fn, future))
        return future

    app.state.run_on_main = run_on_main_thread

    # Create World on main thread
    logger.info("Creating Isaac Sim World on main thread...")
    sim_manager.world = World(
        stage_units_in_meters=1.0,
        physics_dt=sim_manager.PHYSICS_DT,
        rendering_dt=sim_manager.PHYSICS_DT * sim_manager.RENDER_EVERY_N,
    )
    _SIMULATION_APP.update()
    _SIMULATION_APP.update()

    # Enable PhysX GPU dynamics
    try:
        stage = sim_manager.world.stage
        physx_scene = PhysxSchema.PhysxSceneAPI.Apply(
            stage.GetPrimAtPath("/physicsScene")
        )
        physx_scene.GetEnableGPUDynamicsAttr().Set(True)
        physx_scene.GetBroadphaseTypeAttr().Set("GPU")
        _SIMULATION_APP.update()
        logger.info("PhysX GPU dynamics enabled")
    except Exception as e:
        logger.warning("Could not enable PhysX GPU dynamics: %s", e)

    sim_manager._assets_root = get_assets_root_path()
    sim_state["initialized"] = True
    logger.info("World created, assets root: %s", sim_manager._assets_root)

    # Start uvicorn in a background thread
    server_thread = threading.Thread(
        target=uvicorn.run,
        args=(app,),
        kwargs={"host": "0.0.0.0", "port": 9090, "log_level": "info"},
        daemon=True,
    )
    server_thread.start()
    logger.info("Uvicorn started in background thread")

    # Main thread: process commands + keep SimulationApp alive
    while _SIMULATION_APP.is_running():
        # Process pending commands from API thread
        while not _cmd_queue.empty():
            try:
                fn, future = _cmd_queue.get_nowait()
                try:
                    result = fn()
                    future.set_result(result)
                except Exception as e:
                    future.set_exception(e)
            except queue.Empty:
                break
        _SIMULATION_APP.update()

    _SIMULATION_APP.close()


if __name__ == "__main__":
    try:
        _start_with_isaac_sim()
    except Exception as e:
        logger.info("Isaac Sim standalone not available (%s), running mock-only", e)
        uvicorn.run(app, host="0.0.0.0", port=9090)
