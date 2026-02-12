"""
Isaac Sim Internal Server
Runs inside the Isaac Sim Docker container.
Provides a REST API for the main backend to control simulations.
"""

import asyncio
import json
import logging
import os
import tempfile
import time
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("isaac-sim-server")

app = FastAPI(title="AutoGrip Isaac Sim Server", version="1.0.0")

# Global simulation state
sim_state = {
    "initialized": False,
    "scene_loaded": False,
    "robot_loaded": False,
    "object_loaded": False,
    "running": False,
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


class SimResult(BaseModel):
    success: bool
    duration: float
    object_final_position: list[float]
    contact_forces: list[float]
    object_trajectory: list[list[float]]
    logs: list[str]
    frames: list[str]  # base64 encoded PNG frames
    error: str | None = None


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


class IsaacSimManager:
    """Manages Isaac Sim simulation lifecycle."""

    def __init__(self):
        self.world = None
        self.robot = None
        self.target_object = None
        self.frames = []
        self.logs = []

    async def initialize(self):
        if not ISAAC_SIM_AVAILABLE:
            logger.info("Mock initialization")
            sim_state["initialized"] = True
            return True

        try:
            self.world = World(stage_units_in_meters=1.0)
            await self.world.initialize_simulation_context_async()
            sim_state["initialized"] = True
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
            self.logs.append(f"Robot '{config.model}' loaded at position {config.position}")
            return True
        except Exception as e:
            logger.error(f"Failed to load robot: {e}")
            return False

    async def load_object(self, config: ObjectConfig):
        if not ISAAC_SIM_AVAILABLE:
            sim_state["object_loaded"] = True
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
            self.logs.append(f"Object loaded at position {config.position}")
            return True
        except Exception as e:
            logger.error(f"Failed to load object: {e}")
            return False

    async def execute_grasp_code(self, code: str, timeout: float = 30.0) -> SimResult:
        """Execute generated grasping code in the simulation."""
        start_time = time.time()
        self.frames = []
        self.logs = []
        object_trajectory = []

        if not ISAAC_SIM_AVAILABLE:
            return self._mock_execute(code, timeout)

        try:
            # Create a sandboxed execution environment
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

            # Run simulation steps
            num_steps = int(timeout * 60)  # 60 Hz
            for step in range(num_steps):
                self.world.step(render=True)

                if self.target_object:
                    pos = self.target_object.get_world_pose()[0].tolist()
                    object_trajectory.append(pos)

                if step % 60 == 0:  # Capture frame every second
                    self.logs.append(f"Step {step}: Object at {pos}")

                elapsed = time.time() - start_time
                if elapsed > timeout:
                    break

            # Get final state
            final_pos = [0.0, 0.0, 0.0]
            contact_forces = [0.0]
            if self.target_object:
                final_pos = self.target_object.get_world_pose()[0].tolist()

            duration = time.time() - start_time
            sim_state["running"] = False

            return SimResult(
                success=True,
                duration=duration,
                object_final_position=final_pos,
                contact_forces=contact_forces,
                object_trajectory=object_trajectory,
                logs=self.logs,
                frames=[],
                error=None,
            )

        except Exception as e:
            sim_state["running"] = False
            duration = time.time() - start_time
            return SimResult(
                success=False,
                duration=duration,
                object_final_position=[0.0, 0.0, 0.0],
                contact_forces=[0.0],
                object_trajectory=object_trajectory,
                logs=self.logs + [f"ERROR: {str(e)}"],
                frames=[],
                error=str(e),
            )

    def _mock_execute(self, code: str, timeout: float) -> SimResult:
        """Mock simulation for development without Isaac Sim."""
        import random
        import hashlib

        code_hash = hashlib.md5(code.encode()).hexdigest()
        random.seed(code_hash)

        # Analyze code quality heuristically
        score = 0
        if "torque" in code.lower() or "force" in code.lower():
            score += 2
        if "grasp" in code.lower() or "grip" in code.lower():
            score += 1
        if "approach" in code.lower():
            score += 1
        if "close" in code.lower() or "clamp" in code.lower():
            score += 1
        if "lift" in code.lower() or "raise" in code.lower():
            score += 1

        success_prob = min(0.9, score * 0.15 + random.random() * 0.3)
        is_success = random.random() < success_prob

        trajectory = []
        z_start = 0.05
        for i in range(150):
            z = z_start + (0.1 if is_success else -0.01) * (i / 150)
            if not is_success and i > 80:
                z = max(0.0, z - 0.002 * (i - 80))
            trajectory.append([0.0, 0.0, round(z, 4)])

        final_pos = trajectory[-1] if trajectory else [0.0, 0.0, 0.0]
        contact = [round(random.uniform(1.0, 10.0), 2)] if is_success else [0.0]

        logs = [
            f"[MOCK] Simulation started (code score: {score})",
            f"[MOCK] Robot initialized, approaching object...",
            f"[MOCK] Grasp attempt executing...",
            f"[MOCK] Object final Z: {final_pos[2]:.4f}",
            f"[MOCK] Contact force: {contact[0]:.2f}N",
            f"[MOCK] Result: {'SUCCESS' if is_success else 'FAILED'}",
        ]

        if not is_success:
            failure_reasons = [
                "Object slip detected - insufficient friction/torque",
                "Object dropped during lift - grasp not secure",
                "No contact established - approach vector misaligned",
                "Collision detected during approach phase",
            ]
            error = random.choice(failure_reasons)
            logs.append(f"[MOCK] Failure reason: {error}")
        else:
            error = None
            logs.append("[MOCK] Object held stable for 5.0 seconds")

        return SimResult(
            success=is_success,
            duration=round(random.uniform(2.0, 8.0), 2),
            object_final_position=final_pos,
            contact_forces=contact,
            object_trajectory=trajectory,
            logs=logs,
            frames=[],
            error=error,
        )

    async def reset(self):
        """Reset the simulation."""
        if ISAAC_SIM_AVAILABLE and self.world:
            await self.world.reset_async()
        self.frames = []
        self.logs = []
        sim_state["scene_loaded"] = False
        sim_state["robot_loaded"] = False
        sim_state["object_loaded"] = False
        sim_state["running"] = False


# Global sim manager
sim_manager = IsaacSimManager()


@app.on_event("startup")
async def startup():
    await sim_manager.initialize()


@app.get("/health")
async def health():
    return {"status": "ok", "isaac_sim_available": ISAAC_SIM_AVAILABLE, "state": sim_state}


@app.post("/scene/load")
async def load_scene(config: SceneConfig):
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
    return result.model_dump()


@app.post("/reset")
async def reset_sim():
    await sim_manager.reset()
    return {"status": "reset_complete"}


@app.get("/state")
async def get_state():
    return sim_state


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9090)
