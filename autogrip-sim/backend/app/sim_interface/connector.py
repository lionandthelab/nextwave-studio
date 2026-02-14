"""Isaac Sim connection manager with mock simulator for development."""

from __future__ import annotations

import base64
import hashlib
import io
import logging
import math
import random
import struct
import time
from dataclasses import dataclass, field

import httpx

from app.config import settings

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes for simulation state
# ---------------------------------------------------------------------------


@dataclass
class Vec3:
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0

    def length(self) -> float:
        return math.sqrt(self.x**2 + self.y**2 + self.z**2)


@dataclass
class ObjectState:
    name: str
    prim_path: str
    position: Vec3 = field(default_factory=Vec3)
    velocity: Vec3 = field(default_factory=Vec3)
    angular_velocity: Vec3 = field(default_factory=Vec3)
    contacts: list[dict] = field(default_factory=list)
    mass: float = 1.0


@dataclass
class RobotState:
    model: str = ""
    prim_path: str = ""
    joint_positions: dict[str, float] = field(default_factory=dict)
    joint_torques: dict[str, float] = field(default_factory=dict)
    gripper_opening: float = 0.1
    end_effector_position: Vec3 = field(default_factory=Vec3)


@dataclass
class SimulationContext:
    """Tracks the full state of a simulation session."""

    running: bool = False
    headless: bool = True
    time_step: float = 1.0 / 120.0
    elapsed_time: float = 0.0
    frame_count: int = 0
    gravity: float = -9.81
    objects: dict[str, ObjectState] = field(default_factory=dict)
    robot: RobotState | None = None
    ground_plane: bool = False
    frames: list[bytes] = field(default_factory=list)
    logs: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# MockSimulator - realistic simulation behavior for development
# ---------------------------------------------------------------------------


class MockSimulator:
    """Simulates Isaac Sim behavior for development/testing.

    Models approximate physics to provide realistic success/failure patterns.
    Early iterations tend to fail; as code quality improves (detected via
    torque values and grasp width), success probability increases.
    """

    def __init__(self):
        self._rng = random.Random()
        self._iteration_count = 0

    def evaluate_code_quality(self, code: str) -> dict:
        """Analyze generated code to estimate its quality.

        Checks for key parameters that affect grasping success.
        """
        quality = {
            "torque_value": 2.0,
            "grasp_width": 0.08,
            "approach_height": 0.1,
            "has_error_handling": False,
            "has_contact_check": False,
            "has_hold_phase": False,
            "velocity_limit": 1.0,
        }

        # Extract torque values from code
        import re

        torque_matches = re.findall(
            r"torque\s*[=:]\s*(\d+\.?\d*)", code, re.IGNORECASE
        )
        if torque_matches:
            quality["torque_value"] = max(float(t) for t in torque_matches)

        # Extract grasp/grip width
        width_matches = re.findall(
            r"(?:grasp|grip)[\s_]*width\s*[=:]\s*(\d+\.?\d*)", code, re.IGNORECASE
        )
        if width_matches:
            quality["grasp_width"] = float(width_matches[-1])

        # Check for approach height
        height_matches = re.findall(
            r"(?:approach|pre.?grasp)[\s_]*height\s*[=:]\s*(\d+\.?\d*)",
            code,
            re.IGNORECASE,
        )
        if height_matches:
            quality["approach_height"] = float(height_matches[-1])

        # Check for error handling
        quality["has_error_handling"] = "try" in code and "except" in code

        # Check for contact verification
        quality["has_contact_check"] = any(
            kw in code.lower()
            for kw in ("contact", "force_sensor", "get_contact", "is_touching")
        )

        # Check for hold phase
        quality["has_hold_phase"] = any(
            kw in code.lower() for kw in ("hold", "maintain", "sleep(5", "wait(5")
        )

        return quality

    def compute_success_probability(
        self, code_quality: dict, iteration: int
    ) -> float:
        """Compute probability of successful grasp based on code quality."""
        base_prob = 0.05  # Very low base probability

        # Torque contribution: higher torque (up to a point) helps
        torque = code_quality["torque_value"]
        if torque < 1.0:
            torque_bonus = 0.0
        elif torque < 5.0:
            torque_bonus = torque * 0.08
        elif torque < 15.0:
            torque_bonus = 0.4
        else:
            torque_bonus = 0.3  # Too much torque can cause issues

        # Grasp width contribution
        width = code_quality["grasp_width"]
        if 0.03 < width < 0.12:
            width_bonus = 0.15
        else:
            width_bonus = 0.0

        # Feature bonuses
        error_bonus = 0.05 if code_quality["has_error_handling"] else 0.0
        contact_bonus = 0.1 if code_quality["has_contact_check"] else 0.0
        hold_bonus = 0.1 if code_quality["has_hold_phase"] else 0.0

        # Iteration bonus: slight improvement over time (learning)
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
        issues = []

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
            # Random failure
            return self._rng.choice(["slip", "no_contact", "collision"])

        # Weighted random selection
        total = sum(w for _, w in issues)
        r = self._rng.random() * total
        cumulative = 0.0
        for mode, weight in issues:
            cumulative += weight
            if r <= cumulative:
                return mode
        return issues[-1][0]

    async def simulate_execution(
        self, code: str, context: SimulationContext
    ) -> dict:
        """Execute code in the mock simulator and return results.

        Args:
            code: The grasping Python code to evaluate.
            context: Current simulation context.

        Returns:
            Execution results dict.
        """
        self._iteration_count += 1
        quality = self.evaluate_code_quality(code)
        success_prob = self.compute_success_probability(
            quality, self._iteration_count
        )

        success = self._rng.random() < success_prob
        duration = self._rng.uniform(3.0, 8.0)

        logs = []
        frames = []

        if success:
            sim_data = self._simulate_successful_grasp(context, quality, duration)
        else:
            failure_mode = self.determine_failure_mode(quality)
            sim_data = self._simulate_failed_grasp(
                context, quality, duration, failure_mode
            )

        logs = sim_data["logs"]
        frames = sim_data["frames"]

        # Update context
        context.elapsed_time += duration
        context.frame_count += len(frames)
        context.frames.extend(frames)
        context.logs.extend(logs)

        return {
            "success": success,
            "duration": duration,
            "frames": frames,
            "logs": logs,
            "object_final_state": sim_data["object_state"],
            "contact_forces": sim_data["contact_forces"],
            "joint_states": sim_data["joint_states"],
        }

    def _simulate_successful_grasp(
        self, context: SimulationContext, quality: dict, duration: float
    ) -> dict:
        """Generate data for a successful grasp sequence."""
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

        num_frames = int(duration * 10)  # 10 FPS for captures
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
        self,
        context: SimulationContext,
        quality: dict,
        duration: float,
        failure_mode: str,
    ) -> dict:
        """Generate data for a failed grasp sequence."""
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
        """Generate a minimal PNG-like placeholder frame.

        Creates a small 4x4 pixel PNG with color indicating simulation state:
        - Green tones for successful grasps
        - Red tones for failures
        """
        progress = frame_idx / max(1, total_frames - 1)

        if success:
            r = int(50 + 50 * (1 - progress))
            g = int(100 + 155 * progress)
            b = 80
        else:
            r = int(100 + 155 * progress)
            g = int(100 * (1 - progress))
            b = 50

        # Build a minimal valid 4x4 PNG in memory
        width, height = 4, 4
        raw_rows = []
        for _ in range(height):
            row = b"\x00"  # filter byte
            for _ in range(width):
                row += struct.pack("BBB", r, g, b)
            raw_rows.append(row)

        import zlib

        raw_data = b"".join(raw_rows)
        compressed = zlib.compress(raw_data)

        def _make_chunk(chunk_type: bytes, data: bytes) -> bytes:
            import struct as _st
            chunk = chunk_type + data
            crc = zlib.crc32(chunk) & 0xFFFFFFFF
            return _st.pack(">I", len(data)) + chunk + _st.pack(">I", crc)

        png = b"\x89PNG\r\n\x1a\n"
        # IHDR
        ihdr_data = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
        png += _make_chunk(b"IHDR", ihdr_data)
        # IDAT
        png += _make_chunk(b"IDAT", compressed)
        # IEND
        png += _make_chunk(b"IEND", b"")

        return png

    def reset(self):
        """Reset the simulator state for a new session."""
        self._iteration_count = 0


# ---------------------------------------------------------------------------
# IsaacSimConnector - communicates with sim_server via HTTP
# ---------------------------------------------------------------------------


class IsaacSimConnector:
    """Manages connection to Isaac Sim server via HTTP REST API.

    Communicates with the sim_server running inside the Isaac Sim Docker
    container (or in mock mode for development/testing).
    """

    def __init__(self, http_client: httpx.AsyncClient | None = None):
        self._context: SimulationContext | None = None
        self._http_client = http_client
        self._owns_client = http_client is None

    async def _get_client(self) -> httpx.AsyncClient:
        """Return the HTTP client, creating one if needed."""
        if self._http_client is None:
            self._http_client = httpx.AsyncClient(
                base_url=settings.isaac_sim_endpoint,
                timeout=60.0,
            )
        return self._http_client

    @property
    def context(self) -> SimulationContext | None:
        return self._context

    async def start_simulation(self, headless: bool = True) -> bool:
        """Start a new simulation session.

        Sends /init to the sim_server to initialise the simulation engine.

        Args:
            headless: Whether to run without GUI.

        Returns:
            True if the simulation started successfully.
        """
        client = await self._get_client()
        resp = await client.post("/init", json={"headless": headless})
        resp.raise_for_status()

        self._context = SimulationContext(running=True, headless=headless)
        self._context.logs.append("Simulation session started")

        logger.info("Simulation started (headless=%s)", headless)
        return True

    async def load_scene(self) -> bool:
        """Set up the basic simulation scene (ground plane, lighting, gravity)."""
        if not self._context:
            raise RuntimeError("Simulation not started")

        client = await self._get_client()
        resp = await client.post("/scene/load", json={})
        resp.raise_for_status()

        self._context.ground_plane = True
        self._context.logs.append(
            "Scene loaded: ground plane, default lighting, gravity=-9.81"
        )
        logger.info("Scene loaded")
        return True

    async def load_robot(self, robot_model: str) -> bool:
        """Load a robot USD model into the scene.

        Args:
            robot_model: Robot model identifier (e.g. 'unitree_h1').

        Returns:
            True if robot loaded successfully.
        """
        if not self._context:
            raise RuntimeError("Simulation not started")

        client = await self._get_client()
        resp = await client.post("/robot/load", json={"model": robot_model})
        resp.raise_for_status()

        prim_path = f"/World/{robot_model}"
        self._context.robot = RobotState(
            model=robot_model,
            prim_path=prim_path,
            joint_positions={
                "joint_0": 0.0,
                "joint_1": 0.0,
                "joint_2": 0.0,
                "joint_3": 0.0,
                "joint_4": 0.0,
                "joint_5": 0.0,
                "gripper_left": 0.05,
                "gripper_right": -0.05,
            },
            gripper_opening=0.1,
            end_effector_position=Vec3(0.0, 0.0, 0.5),
        )
        self._context.logs.append(f"Robot loaded: {robot_model} at {prim_path}")
        logger.info("Robot loaded: %s", robot_model)
        return True

    async def load_object(
        self, cad_file_path: str, position: tuple = (0.5, 0.0, 0.05)
    ) -> bool:
        """Load a CAD object into the simulation scene.

        Args:
            cad_file_path: Path to the CAD file.
            position: (x, y, z) spawn position.

        Returns:
            True if object loaded successfully.
        """
        if not self._context:
            raise RuntimeError("Simulation not started")

        client = await self._get_client()
        resp = await client.post("/object/load", json={
            "cad_file_path": cad_file_path,
            "position": list(position),
        })
        resp.raise_for_status()

        obj_name = f"object_{hashlib.md5(cad_file_path.encode()).hexdigest()[:8]}"
        prim_path = f"/World/{obj_name}"

        obj_state = ObjectState(
            name=obj_name,
            prim_path=prim_path,
            position=Vec3(*position),
            mass=1.0,
        )
        self._context.objects[obj_name] = obj_state
        self._context.logs.append(
            f"Object loaded: {cad_file_path} at {prim_path} pos={position}"
        )
        logger.info("Object loaded: %s at %s", cad_file_path, position)
        return True

    async def execute_code(self, code: str) -> dict:
        """Execute generated grasping code in the simulation.

        Sends the code to the sim_server for execution, decodes base64
        frames from the response, and updates the local context.

        Args:
            code: Python code string to execute.

        Returns:
            Execution results with success, duration, frames, and logs.
        """
        if not self._context:
            raise RuntimeError("Simulation not started")

        self._context.logs.append("Executing generated grasping code...")

        client = await self._get_client()
        resp = await client.post("/execute", json={"code": code})
        resp.raise_for_status()
        result = resp.json()

        # Decode base64 frames to PNG bytes
        frames_b64 = result.get("frames", [])
        frames = [base64.b64decode(f) for f in frames_b64]

        # Update local context
        duration = result.get("duration", 0.0)
        self._context.elapsed_time += duration
        self._context.frame_count += len(frames)
        self._context.frames.extend(frames)
        self._context.logs.extend(result.get("logs", []))

        return {
            "success": result["success"],
            "duration": result["duration"],
            "frames": frames,
            "logs": result.get("logs", []),
            "object_final_state": result["object_final_state"],
            "contact_forces": result["contact_forces"],
            "joint_states": result["joint_states"],
        }

    async def capture_frames(self) -> list[bytes]:
        """Return all captured simulation frames as PNG bytes."""
        if not self._context:
            return []
        return list(self._context.frames)

    async def stop_simulation(self):
        """Stop and clean up the simulation session."""
        if self._context:
            try:
                client = await self._get_client()
                await client.post("/reset")
            except Exception:
                logger.warning("Failed to reset sim server during stop")

            self._context.running = False
            self._context.logs.append("Simulation session stopped")
            logger.info("Simulation stopped (elapsed=%.1fs)", self._context.elapsed_time)
        self._context = None

    async def get_object_state(self) -> dict:
        """Return the current state of the first loaded object.

        Returns:
            Dict with position, velocity, angular_velocity, and contacts.
        """
        if not self._context or not self._context.objects:
            return {}

        obj = next(iter(self._context.objects.values()))
        return {
            "name": obj.name,
            "position": [obj.position.x, obj.position.y, obj.position.z],
            "velocity": [obj.velocity.x, obj.velocity.y, obj.velocity.z],
            "angular_velocity": [
                obj.angular_velocity.x,
                obj.angular_velocity.y,
                obj.angular_velocity.z,
            ],
            "contacts": obj.contacts,
            "mass": obj.mass,
        }

    async def close(self):
        """Close the HTTP client if owned by this connector."""
        if self._owns_client and self._http_client is not None:
            await self._http_client.aclose()
            self._http_client = None
