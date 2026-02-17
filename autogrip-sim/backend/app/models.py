"""Pydantic models for request/response schemas."""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class UploadResponse(BaseModel):
    id: str
    filename: str
    file_type: str = Field(description="'cad' or 'manual'")
    size_bytes: int


class CADMetadata(BaseModel):
    filename: str
    format: str
    dimensions: dict[str, float] = Field(
        description="Bounding box dimensions with keys x, y, z"
    )
    volume: Optional[float] = None
    center_of_mass: Optional[list[float]] = Field(
        default=None, description="[x, y, z] center of mass"
    )


class SessionCreate(BaseModel):
    cad_file_id: str
    manual_file_id: Optional[str] = None
    robot_model: str = "franka_allegro"


class SessionResponse(BaseModel):
    session_id: str
    status: str
    created_at: str


class GenerateRequest(BaseModel):
    session_id: str


class SimulationResult(BaseModel):
    iteration: int
    success: bool
    checks: dict[str, bool] = Field(
        description="Check names mapped to pass/fail"
    )
    error_log: Optional[str] = None
    code_diff: Optional[str] = None
    place_target: Optional[list[float]] = Field(
        default=None, description="[x, y, z] target position for pick-and-place mode"
    )
    place_accuracy: Optional[float] = Field(
        default=None, description="Distance (m) from object final position to place_target"
    )
    replay_data: Optional[dict] = Field(
        default=None,
        description="Simulation replay data for 3D viewer animation: phases, trajectory, contacts",
    )


class LoopStatus(BaseModel):
    session_id: str
    current_iteration: int
    max_iterations: int
    status: str = Field(
        description="One of: running, success, failed, stopped"
    )
    results: list[SimulationResult] = Field(default_factory=list)
    final_code: Optional[str] = None
