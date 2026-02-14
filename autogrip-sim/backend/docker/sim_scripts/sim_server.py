"""
Isaac Sim Internal Server
Runs inside the Isaac Sim Docker container.
Provides a REST API for the main backend to control simulations.

Response format is aligned with what GraspValidator expects:
- object_final_state: {position, velocity, angular_velocity, contact_count}
- contact_forces: [{finger, force_n}, ...]
- joint_states: {joint_name: {position, torque}, ...}
"""

import base64
import logging
import os
import random
import re
import struct
import time
import zlib

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


class RobotConfig(BaseModel):
    model: str = "unitree_h1"
    usd_path: str | None = None
    position: list[float] = [0.0, 0.0, 0.0]


class ObjectConfig(BaseModel):
    cad_file_path: str
    position: list[float] = [0.0, 0.0, 0.05]
    scale: list[float] = [1.0, 1.0, 1.0]


class ExecuteRequest(BaseModel):
    code: str
    timeout: float = 30.0


class InitRequest(BaseModel):
    headless: bool = True


# Try to import Isaac Sim - falls back to mock if not available
try:
    from omni.isaac.core import World
    from omni.isaac.core.utils.stage import add_reference_to_stage
    from omni.isaac.core.prims import XFormPrim, RigidPrim
    from omni.isaac.core.robots import Robot
    from omni.isaac.core.utils.nucleus import get_assets_root_path
    import omni.isaac.core.utils.physics as physics_utils

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

    def evaluate_code_quality(self, code: str) -> dict:
        """Analyze generated code to estimate its quality."""
        quality = {
            "torque_value": 2.0,
            "grasp_width": 0.08,
            "approach_height": 0.1,
            "has_error_handling": False,
            "has_contact_check": False,
            "has_hold_phase": False,
            "velocity_limit": 1.0,
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

    def execute(self, code: str, timeout: float = 30.0) -> dict:
        """Execute code in the mock simulator and return results.

        Returns results in the format expected by the validator.
        """
        self._iteration_count += 1
        quality = self.evaluate_code_quality(code)
        success_prob = self.compute_success_probability(
            quality, self._iteration_count
        )

        success = self._rng.random() < success_prob
        duration = self._rng.uniform(3.0, 8.0)

        if success:
            sim_data = self._simulate_successful_grasp(quality, duration)
        else:
            failure_mode = self.determine_failure_mode(quality)
            sim_data = self._simulate_failed_grasp(quality, duration, failure_mode)

        frames_b64 = [
            base64.b64encode(frame_bytes).decode("ascii")
            for frame_bytes in sim_data["frames"]
        ]

        return {
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

    def _simulate_successful_grasp(self, quality: dict, duration: float) -> dict:
        logs = [
            f"[{0.0:.2f}s] Initializing grasp sequence",
            f"[{0.3:.2f}s] Moving to pre-grasp position",
            f"[{1.0:.2f}s] Approaching object - clearance: {quality['approach_height']:.3f}m",
            f"[{1.5:.2f}s] Contact detected on left finger (force: {quality['torque_value'] / 0.06 * 1.05:.1f}N)",
            f"[{1.6:.2f}s] Contact detected on right finger (force: {quality['torque_value'] / 0.06 * 0.95:.1f}N)",
            f"[{1.8:.2f}s] Gripper closed - width: {quality['grasp_width']:.4f}m",
            f"[{2.0:.2f}s] Applying grasp torque: {quality['torque_value']:.1f}Nm",
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
                {"finger": "left", "force_n": quality["torque_value"] / 0.06 * 1.05},
                {"finger": "right", "force_n": quality["torque_value"] / 0.06 * 0.95},
            ],
            "joint_states": {
                "joint_0": {"position": 0.0, "torque": 0.0},
                "joint_1": {"position": -0.5, "torque": 2.1},
                "joint_2": {"position": 1.2, "torque": 3.4},
                "gripper_left": {
                    "position": quality["grasp_width"] / 2,
                    "torque": quality["torque_value"],
                },
                "gripper_right": {
                    "position": -quality["grasp_width"] / 2,
                    "torque": quality["torque_value"],
                },
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
                "joint_0": {"position": 0.0, "torque": 0.0},
                "joint_1": {"position": -0.3, "torque": 1.5},
                "joint_2": {"position": 0.8, "torque": 2.0},
                "gripper_left": {"position": 0.0, "torque": 0.0},
                "gripper_right": {"position": 0.0, "torque": 0.0},
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
    """Manages Isaac Sim simulation lifecycle."""

    def __init__(self):
        self.world = None
        self.robot = None
        self.target_object = None
        self.frames: list = []
        self.logs: list = []
        self._mock_engine = MockSimulationEngine()

    async def initialize(self, headless: bool = True):
        if not ISAAC_SIM_AVAILABLE:
            logger.info("Mock initialization (headless=%s)", headless)
            sim_state["initialized"] = True
            sim_state["headless"] = headless
            return True

        try:
            self.world = World(stage_units_in_meters=1.0)
            await self.world.initialize_simulation_context_async()
            sim_state["initialized"] = True
            sim_state["headless"] = headless
            logger.info("Isaac Sim world initialized")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize: {e}")
            return False

    async def load_scene(self, config: SceneConfig):
        if not ISAAC_SIM_AVAILABLE:
            sim_state["scene_loaded"] = True
            self.logs.append("Scene loaded (mock mode)")
            return True

        try:
            self.world.scene.add_default_ground_plane()
            sim_state["scene_loaded"] = True
            self.logs.append("Scene loaded with ground plane and lighting")
            return True
        except Exception as e:
            logger.error(f"Failed to load scene: {e}")
            return False

    async def load_robot(self, config: RobotConfig):
        if not ISAAC_SIM_AVAILABLE:
            sim_state["robot_loaded"] = True
            sim_state["robot_model"] = config.model
            self.logs.append(f"Robot '{config.model}' loaded (mock mode)")
            return True

        try:
            assets_root = get_assets_root_path()
            robot_usd = config.usd_path or f"{assets_root}/Isaac/Robots/Unitree/{config.model}.usd"
            self.robot = self.world.scene.add(
                Robot(
                    prim_path="/World/Robot",
                    name="grasp_robot",
                    usd_path=robot_usd,
                    position=config.position,
                )
            )
            sim_state["robot_loaded"] = True
            sim_state["robot_model"] = config.model
            self.logs.append(f"Robot '{config.model}' loaded at position {config.position}")
            return True
        except Exception as e:
            logger.error(f"Failed to load robot: {e}")
            return False

    async def load_object(self, config: ObjectConfig):
        if not ISAAC_SIM_AVAILABLE:
            sim_state["object_loaded"] = True
            sim_state["object_path"] = config.cad_file_path
            sim_state["object_position"] = config.position
            self.logs.append(f"Object loaded from {config.cad_file_path} (mock mode)")
            return True

        try:
            self.target_object = self.world.scene.add(
                RigidPrim(
                    prim_path="/World/TargetObject",
                    name="target_object",
                    usd_path=config.cad_file_path,
                    position=config.position,
                    scale=config.scale,
                )
            )
            sim_state["object_loaded"] = True
            sim_state["object_path"] = config.cad_file_path
            sim_state["object_position"] = config.position
            self.logs.append(f"Object loaded at position {config.position}")
            return True
        except Exception as e:
            logger.error(f"Failed to load object: {e}")
            return False

    async def execute_grasp_code(self, code: str, timeout: float = 30.0) -> dict:
        """Execute generated grasping code in the simulation."""
        start_time = time.time()
        self.frames = []
        self.logs = []

        if not ISAAC_SIM_AVAILABLE:
            return self._mock_engine.execute(code, timeout)

        try:
            exec_globals = {
                "world": self.world,
                "robot": self.robot,
                "target_object": self.target_object,
                "np": __import__("numpy"),
                "time": __import__("time"),
                "logger": logger,
            }

            sim_state["running"] = True
            exec(code, exec_globals)

            num_steps = int(timeout * 60)
            for step in range(num_steps):
                self.world.step(render=True)

                if self.target_object:
                    pos = self.target_object.get_world_pose()[0].tolist()

                if step % 60 == 0:
                    self.logs.append(f"Step {step}: Object at {pos}")

                elapsed = time.time() - start_time
                if elapsed > timeout:
                    break

            final_pos = [0.0, 0.0, 0.0]
            if self.target_object:
                final_pos = self.target_object.get_world_pose()[0].tolist()

            duration = time.time() - start_time
            sim_state["running"] = False

            return {
                "success": True,
                "duration": duration,
                "object_final_state": {
                    "position": final_pos,
                    "velocity": [0.0, 0.0, 0.0],
                    "angular_velocity": [0.0, 0.0, 0.0],
                    "contact_count": 0,
                },
                "contact_forces": [],
                "joint_states": {},
                "logs": self.logs,
                "frames": [],
                "error": None,
            }

        except Exception as e:
            sim_state["running"] = False
            duration = time.time() - start_time
            return {
                "success": False,
                "duration": duration,
                "object_final_state": {
                    "position": [0.0, 0.0, 0.0],
                    "velocity": [0.0, 0.0, 0.0],
                    "angular_velocity": [0.0, 0.0, 0.0],
                    "contact_count": 0,
                },
                "contact_forces": [],
                "joint_states": {},
                "logs": self.logs + [f"ERROR: {str(e)}"],
                "frames": [],
                "error": str(e),
            }

    async def reset(self):
        """Reset the simulation."""
        if ISAAC_SIM_AVAILABLE and self.world:
            await self.world.reset_async()
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


@app.on_event("startup")
async def startup():
    await sim_manager.initialize()


@app.get("/health")
async def health():
    return {"status": "ok", "isaac_sim_available": ISAAC_SIM_AVAILABLE, "state": sim_state}


@app.post("/init")
async def init_sim(config: InitRequest = InitRequest()):
    success = await sim_manager.initialize(headless=config.headless)
    if not success:
        raise HTTPException(500, "Failed to initialize simulation")
    return {"status": "initialized", "headless": config.headless}


@app.post("/scene/load")
async def load_scene(config: SceneConfig = SceneConfig()):
    success = await sim_manager.load_scene(config)
    if not success:
        raise HTTPException(500, "Failed to load scene")
    return {"status": "scene_loaded"}


@app.post("/robot/load")
async def load_robot(config: RobotConfig):
    success = await sim_manager.load_robot(config)
    if not success:
        raise HTTPException(500, "Failed to load robot")
    return {"status": "robot_loaded", "model": config.model}


@app.post("/object/load")
async def load_object(config: ObjectConfig):
    success = await sim_manager.load_object(config)
    if not success:
        raise HTTPException(500, "Failed to load object")
    return {"status": "object_loaded"}


@app.post("/execute")
async def execute_code(request: ExecuteRequest):
    result = await sim_manager.execute_grasp_code(request.code, request.timeout)
    return result


@app.post("/reset")
async def reset_sim():
    await sim_manager.reset()
    return {"status": "reset_complete"}


@app.get("/state")
async def get_state():
    return sim_state


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9090)
