"""RAG-based code generation engine for robot grasping code."""

from __future__ import annotations

import json
import logging
import re
import uuid

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

from app.config import settings
from app.core.parser import ManualParser

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompt templates -- optimized for structured output and code quality
# ---------------------------------------------------------------------------

# System-level instruction shared across all code-generation prompts.
# Separated from the user prompt so the LLM treats it as a persistent role.
SYSTEM_PROMPT = """\
You are an expert robotics engineer specializing in NVIDIA Isaac Sim grasping \
simulations with a Franka Panda arm and Allegro Hand (4-finger dexterous hand).

Robot Configuration:
- ARM: Franka Panda (7 DOF) -- joints: panda_joint1 through panda_joint7.
  Position control for arm joints. Workspace radius ~0.855 m. Payload 3 kg.
- HAND: Allegro Hand (16 DOF) -- 4 fingers x 4 joints each.
  Index: joint_0..joint_3, Middle: joint_4..joint_7,
  Ring: joint_8..joint_11, Thumb: joint_12..joint_15.
  Torque control for finger joints (max ~0.7 Nm per joint).
- Combined prim: /World/Robot  (23 DOF total: indices 0-6 arm, 7-22 fingers).

You write production-quality Python code that:
- Uses the `omni.isaac.core` API correctly (Articulation, RigidPrim, \
XFormPrim, World, SimulationContext).
- Sets the physics time step to 1/120 s.
- Uses position control for Franka arm joints (indices 0-6).
- Uses torque control for Allegro finger joints (indices 7-22).
- Coordinates arm positioning THEN finger grasping (not simultaneous).
- Includes robust error handling: contact-loss detection, joint-limit clamping, \
and timeout guards.
- Never imports modules that are unavailable inside Isaac Sim's embedded Python.
- Always returns results through the prescribed function signature.

When generating code, follow this phased grasp sequence:
  Phase 1 - APPROACH: Move Franka arm to pre-grasp pose above object.
  Phase 2 - DESCEND:  Lower end-effector to grasp height around object COM.
  Phase 3 - GRASP:    Close all 4 Allegro fingers with controlled torque ramp.
  Phase 4 - LIFT:     Raise the object using Franka arm while maintaining grip.
  Phase 5 - HOLD:     Maintain finger torques for the required duration.

Output ONLY valid Python code. Do NOT include markdown fences, explanations, \
or commentary outside the code itself."""

PICK_AND_PLACE_SYSTEM_PROMPT = """\
You are an expert robotics engineer specializing in NVIDIA Isaac Sim \
pick-and-place simulations with a Franka Panda arm and Allegro Hand.

Robot Configuration:
- ARM: Franka Panda (7 DOF) -- joints: panda_joint1 through panda_joint7.
- HAND: Allegro Hand (16 DOF) -- 4 fingers x 4 joints each.
- Combined prim: /World/Robot  (23 DOF total).

You write production-quality Python for an 8-phase pick-and-place sequence:
  Phase 1 - APPROACH:  Move Franka arm to pre-grasp pose above object.
  Phase 2 - DESCEND:   Lower end-effector to grasp height around object COM.
  Phase 3 - GRASP:     Close all 4 Allegro fingers with controlled torque ramp.
  Phase 4 - LIFT:      Raise the object using Franka arm while maintaining grip.
  Phase 5 - TRANSPORT: Move arm laterally to place_target position.
  Phase 6 - PLACE:     Lower object into the tray at place_target.
  Phase 7 - RELEASE:   Open fingers gradually (torque ramp down).
  Phase 8 - RETRACT:   Raise arm away from placed object.

The function signature MUST be:
  run_grasp_simulation(sim_context, robot_prim_path, object_prim_path, \
place_target=None) -> dict

Returns dict with: success (bool), duration (float), logs (list[str]), \
object_trajectory (list[dict]).

Output ONLY valid Python code. Do NOT include markdown fences."""

INITIAL_CODE_PROMPT = """\
## Robot Manual Context
The following are relevant excerpts from the robot's manual.  Use these to \
determine correct joint names, control API calls, torque limits, and motion \
planning parameters.
{manual_context}

## Object Information
- Filename: {object_filename}
- Bounding-box dimensions: X={dim_x:.4f} m, Y={dim_y:.4f} m, Z={dim_z:.4f} m
- Volume: {volume} m^3
- Center of mass: {center_of_mass}
- Estimated mass (plastic, 1200 kg/m^3): {estimated_mass:.3f} kg

## Robot Model
- Model: {robot_model}
- Joint names (from manual): {joint_names}
- Control functions (from manual): {control_functions}

## Task Parameters
- Object spawn position: (0.5, 0.0, {object_height:.4f})
- Minimum gripper torque for reliable grasp: {min_torque} Nm
- Lift height: 0.3 m
- Hold duration: 5 s

## Required Function Signature
The script MUST define exactly one entry-point function:

```python
def run_grasp_simulation(
    sim_context,
    robot_prim_path: str,
    object_prim_path: str,
) -> dict:
    \"\"\"Execute the grasping sequence.

    Args:
        sim_context: The Isaac Sim SimulationContext instance.
        robot_prim_path: USD prim path to the robot (e.g. "/World/Robot").
        object_prim_path: USD prim path to the target object.

    Returns:
        dict with keys:
            success  (bool)  - True if object held for full duration.
            duration (float) - Wall-clock seconds elapsed.
            logs     (list[str]) - Ordered log messages for each phase.
    \"\"\"
```

## Grasp Sequence (implement each phase)
1. **APPROACH** -- Move end-effector to pre-grasp pose \
(directly above object, clearance = object_height + 0.10 m).  \
Open gripper to width >= max(dim_x, dim_y) * 1.2.
2. **DESCEND** -- Lower end-effector to grasp height = object center-of-mass Z.  \
Use smooth interpolation over >= 60 sim steps.
3. **GRASP** -- Ramp gripper torque from 20% to 100% of target over 30 steps.  \
Verify contact force > 0.5 N after closure.  If no contact, abort with \
success=False.
4. **LIFT** -- Raise by 0.3 m over >= 120 sim steps.  Monitor contact each step; \
if lost, abort.
5. **HOLD** -- Maintain grip for 5 s of sim time.  Check contact every 60 steps.

## Few-Shot Example (simplified pattern)
Below is a minimal reference showing the expected code structure.  Adapt joint \
names, torques, and positions to the actual robot and object.

```python
import time
import numpy as np
from omni.isaac.core.articulations import Articulation
from omni.isaac.core.prims import RigidPrim

def run_grasp_simulation(sim_context, robot_prim_path: str, object_prim_path: str) -> dict:
    logs = []
    start = time.time()
    dt = 1.0 / 120.0

    # -- Initialise Franka + Allegro articulation (23 DOF)
    robot = Articulation(prim_path=robot_prim_path)
    robot.initialize()
    num_dof = robot.num_dof  # 7 arm + 16 hand = 23
    arm_idx = list(range(7))        # panda_joint1-7
    finger_idx = list(range(7, 23)) # Allegro joint_0-15

    # Get object position for approach planning
    obj = RigidPrim(prim_path=object_prim_path)
    obj_pos, _ = obj.get_world_pose()

    # -- Phase 1: APPROACH (move Franka arm to pre-grasp pose)
    logs.append("Phase 1: moving Franka arm to pre-grasp pose")
    pre_grasp = np.zeros(num_dof)
    pre_grasp[0:7] = [0.0, -0.5, 0.0, -2.0, 0.0, 2.0, 0.785]
    pre_grasp[7:23] = 0.0  # fingers open
    robot.set_joint_position_targets(pre_grasp)
    for _ in range(180):
        sim_context.step(render=False)

    # -- Phase 2: DESCEND (adjust arm to reach around object)
    logs.append("Phase 2: descending to grasp height")
    descend = pre_grasp.copy()
    # Adjust arm joints to lower end-effector to object height
    descend[1] = -0.3   # shoulder pitch
    descend[3] = -2.2   # elbow
    robot.set_joint_position_targets(descend)
    for _ in range(120):
        sim_context.step(render=False)

    # -- Phase 3: GRASP (torque ramp on all 4 Allegro fingers)
    logs.append("Phase 3: closing Allegro fingers")
    target_torque = 0.5  # Nm per finger joint (max ~0.7)
    for step in range(60):
        fraction = (step + 1) / 60.0
        efforts = np.zeros(num_dof)
        efforts[finger_idx] = target_torque * fraction
        robot.set_joint_efforts(efforts)
        sim_context.step(render=False)

    # -- verify contact via finger position convergence
    pos_after = robot.get_joint_positions()
    finger_moved = np.any(np.abs(pos_after[7:23]) > 0.05)
    if not finger_moved:
        logs.append("ERROR: no contact detected after grasp closure")
        return {{"success": False, "duration": time.time() - start, "logs": logs}}

    # -- Phase 4: LIFT (raise via Franka arm while maintaining finger torques)
    logs.append("Phase 4: lifting object")
    lift = descend.copy()
    lift[1] = -0.6   # raise shoulder
    lift[3] = -1.8   # adjust elbow
    for _ in range(180):
        efforts = np.zeros(num_dof)
        efforts[finger_idx] = target_torque
        robot.set_joint_efforts(efforts)
        robot.set_joint_position_targets(lift)
        sim_context.step(render=False)

    # -- Phase 5: HOLD
    logs.append("Phase 5: holding for 5 s")
    hold_steps = int(5.0 / dt)
    for _ in range(hold_steps):
        efforts = np.zeros(num_dof)
        efforts[finger_idx] = target_torque
        robot.set_joint_efforts(efforts)
        sim_context.step(render=False)

    logs.append("Grasp sequence completed successfully")
    return {{"success": True, "duration": time.time() - start, "logs": logs}}
```

## Constraints
- Use `omni.isaac.core` APIs only.
- All joint commands MUST respect limits from the manual.
- Do NOT use `time.sleep`; advance simulation with `sim_context.step()`.
- Physics time step: 1/120 s.
- Clamp all torque values to the manual-specified maximums.

Generate the complete implementation now."""

CORRECTION_PROMPT = """\
You are fixing Isaac Sim grasping code that failed during simulation.

## Current Code (to be corrected)
```python
{current_code}
```

## Simulation Error Log
```
{error_log}
```

## Attempt {iteration} of {max_iterations}
{escalation_note}

## Object Information
- Bounding-box: X={dim_x:.4f} m, Y={dim_y:.4f} m, Z={dim_z:.4f} m
- Volume: {volume} m^3
- Estimated mass: {estimated_mass:.3f} kg

## Diagnosed Error Category: **{error_type}**

## Correction Strategy
{correction_strategy}

## Rules
1. Keep the function signature: \
`run_grasp_simulation(sim_context, robot_prim_path, object_prim_path) -> dict`
2. Make MINIMAL, TARGETED changes -- do not rewrite parts that work.
3. Add an inline comment `# FIX(iter {iteration}): <reason>` at every changed line.
4. Preserve all existing logging (logs.append calls).
5. If the error is in physics parameters (torque, speed, height), adjust the \
   value by the percentage suggested in the correction strategy -- do not guess.

Generate ONLY the corrected Python code."""

FEASIBILITY_PROMPT = """\
Assess whether the following grasping task is physically feasible.

## Object
- Dimensions: X={dim_x:.4f} m, Y={dim_y:.4f} m, Z={dim_z:.4f} m
- Volume: {volume} m^3
- Estimated mass: {estimated_mass} kg (volume * 1200 kg/m^3)

## Robot
- Model: {robot_model}
- Max payload: {max_payload} kg
- Max grip force: {max_grip_force} N
- Gripper max opening: {max_opening} m
- Joint count: {joint_count}

## Assessment Criteria
1. Gripper opening vs. smallest graspable object dimension.
2. Object mass vs. payload capacity (include 20% safety margin).
3. Required grip force to hold object against gravity (F >= m*g / mu, mu=0.4).
4. Reachability -- object at (0.5, 0, object_height) within typical workspace.

Respond with ONLY this JSON (no markdown, no extra text):
{{"feasible": <bool>, "confidence": <0.0-1.0>, "reason": "<one sentence>", \
"warnings": ["<issue 1>", ...], "recommended_torque": <float Nm>}}"""

# Prompt used to validate generated code before sending it to simulation.
# This catches common mistakes (missing imports, wrong signatures, unsafe ops).
CODE_VALIDATION_PROMPT = """\
Review the following Isaac Sim grasping code for correctness and safety.

```python
{code}
```

Check for these issues:
1. SIGNATURE -- Does it define `run_grasp_simulation(sim_context, \
robot_prim_path: str, object_prim_path: str) -> dict`?
2. RETURN VALUE -- Does every code path return a dict with keys: \
success (bool), duration (float), logs (list[str])?
3. IMPORTS -- Are all imports available inside Isaac Sim's Python environment? \
   Allowed: omni.isaac.core.*, numpy, math, time.  Disallowed: torch, scipy, \
   tensorflow, external HTTP libraries.
4. SAFETY -- No `os.system`, `subprocess`, `eval`, `exec`, or file writes.
5. PHYSICS -- Is `sim_context.step()` used (not `time.sleep`) to advance sim?
6. JOINT LIMITS -- Are torque/position values clamped or bounded?

Respond with ONLY this JSON (no markdown, no extra text):
{{"valid": <bool>, "issues": ["<issue description>", ...], \
"suggested_fix": "<brief fix or empty string>"}}"""

# ---------------------------------------------------------------------------
# Error classification and correction strategies
# ---------------------------------------------------------------------------

# Expanded keyword sets for more accurate error classification
_ERROR_KEYWORDS: dict[str, list[str]] = {
    "slip": [
        "slip", "dropped", "lost grip", "fell", "sliding",
        "insufficient friction", "grip lost", "object fell",
    ],
    "collision": [
        "collision", "collide", "penetration", "overlap",
        "self-collision", "hit", "crash", "intersect",
    ],
    "joint_limit": [
        "joint limit", "joint_limit", "out of range", "limit exceeded",
        "position limit", "velocity limit", "beyond range",
    ],
    "no_contact": [
        "no contact", "no_contact", "miss", "not touching",
        "zero contact", "no force", "failed to reach",
    ],
    "timeout": [
        "timeout", "timed out", "too slow", "exceeded",
        "max steps", "deadline", "hung",
    ],
    "overforce": [
        "overforce", "exceeded force limit", "exceeded safe limit",
        "force exceeded", "torque exceeded", "excessive force",
        "risk of damage", "emergency stop",
    ],
    "unstable_grasp": [
        "unstable_grasp", "unstable grasp", "rotating during lift",
        "angular velocity", "object shifting", "not centered",
        "asymmetric", "object tilting",
    ],
    "workspace_violation": [
        "workspace", "unreachable", "out of reach", "ik failed",
        "inverse kinematics", "beyond range", "cannot reach",
        "outside workspace",
    ],
    "import_error": [
        "importerror", "modulenotfounderror", "no module named",
        "cannot import",
    ],
    "attribute_error": [
        "attributeerror", "has no attribute", "undefined",
    ],
    "place_miss": [
        "place_miss", "place miss", "not accurately placed",
        "missed target", "placement error", "off target",
        "outside tray", "missed tray",
    ],
    "transport_drop": [
        "transport_drop", "transport drop", "dropped during transport",
        "lost during lateral", "object lost during",
        "fell during transport", "transport failed",
    ],
}

# Correction strategies with quantified adjustments
CORRECTION_STRATEGIES: dict[str, str] = {
    "slip": (
        "The object slipped from the gripper.\n"
        "1. INCREASE gripper torque by 40% from current value.\n"
        "2. NARROW initial grasp width by 10% to create tighter contact.\n"
        "3. ADD a torque-ramp phase: ramp from 30% to 100% over 30 sim steps "
        "before the lift phase.\n"
        "4. INSERT a contact-force check after closure: if measured force < 0.5 N, "
        "re-close with 50% more torque before proceeding.\n"
        "5. REDUCE lift speed by 30% (increase lift step count by 40%)."
    ),
    "collision": (
        "The robot collided with the object or environment.\n"
        "1. INCREASE pre-grasp clearance by 0.05 m (add to Z of approach pose).\n"
        "2. SWITCH to a top-down approach trajectory if currently using lateral.\n"
        "3. ADD at least one intermediate waypoint between current pose and "
        "pre-grasp pose to create a collision-free arc.\n"
        "4. REDUCE approach velocity by 30% in the last 0.1 m of descent.\n"
        "5. ENSURE gripper is fully open before descent begins."
    ),
    "joint_limit": (
        "A joint exceeded its limits.\n"
        "1. CLAMP all joint position targets to [lower_limit + 5%, "
        "upper_limit - 5%] using np.clip.\n"
        "2. REDUCE joint velocity commands by 20%.\n"
        "3. CHECK that the target grasp pose is reachable -- if not, "
        "try grasping along the shorter object axis.\n"
        "4. ADD joint-limit validation before every set_joint_position_targets call."
    ),
    "no_contact": (
        "The gripper did not make contact with the object.\n"
        "1. LOWER the grasp target Z by 0.02 m (closer to table surface).\n"
        "2. WIDEN initial gripper opening by 20%.\n"
        "3. SHIFT grasp X/Y to match the object center of mass exactly.\n"
        "4. INCREASE the gripper closure range -- ensure fingers travel at least "
        "the full object width.\n"
        "5. ADD a secondary contact attempt: if first grasp fails, open and "
        "retry 0.01 m lower."
    ),
    "timeout": (
        "The simulation timed out.\n"
        "1. REDUCE total waypoints to at most 3 (pre-grasp, grasp, lift).\n"
        "2. INCREASE joint velocity limits by 25%.\n"
        "3. REMOVE any time.sleep() calls -- use only sim_context.step().\n"
        "4. CUT hold duration to 3 s if currently > 5 s.\n"
        "5. USE direct joint position targets instead of trajectory interpolation."
    ),
    "overforce": (
        "The hand applied excessive force, risking damage.\n"
        "1. REDUCE Allegro finger torque by 40-50% from current value.\n"
        "2. ADD a force feedback check: if contact force > 20N per finger, "
        "reduce torque immediately.\n"
        "3. CLAMP maximum torque to 0.7 Nm for each Allegro finger joint.\n"
        "4. USE a gentler torque ramp: ramp from 10% to 60% of max over 50 steps.\n"
        "5. ENSURE the force check runs every sim step during the grasp phase."
    ),
    "unstable_grasp": (
        "The object was grasped but became unstable during lift.\n"
        "1. SHIFT the grasp point to align with the object's center of mass.\n"
        "2. APPLY symmetric forces on all gripper fingers.\n"
        "3. REDUCE lift speed by 40% (increase lift step count).\n"
        "4. ADD angular velocity monitoring during lift: if > 0.5 rad/s, pause "
        "and re-stabilize grip before continuing.\n"
        "5. INCREASE gripper torque by 15% during lift to maintain firm hold."
    ),
    "workspace_violation": (
        "The target position is outside the Franka Panda's reachable workspace.\n"
        "1. CHECK that the object is within 0.10-0.85m from the robot base.\n"
        "2. ADJUST the approach to use a different arm configuration "
        "(elbow-up vs elbow-down).\n"
        "3. MOVE the grasp target to the nearest reachable point on the object.\n"
        "4. VERIFY that the lift target (current_z + 0.3m) does not exceed "
        "the workspace ceiling (0.80m above base).\n"
        "5. If the object is too far, consider a two-step approach: first slide "
        "the object closer, then grasp."
    ),
    "import_error": (
        "An import failed inside Isaac Sim.\n"
        "1. REMOVE the failing import and replace with the Isaac Sim equivalent.\n"
        "2. Allowed imports: omni.isaac.core.*, numpy, math, time.\n"
        "3. Do NOT use torch, scipy, tensorflow, or any pip-only package.\n"
        "4. For matrix operations, use numpy instead of scipy.spatial."
    ),
    "attribute_error": (
        "An attribute or method does not exist on an object.\n"
        "1. CHECK the Isaac Sim API: Articulation uses set_joint_position_targets, "
        "set_joint_velocity_targets, set_joint_efforts, get_joint_positions.\n"
        "2. VERIFY prim paths: robot at /World/Robot, object at /World/TargetObject.\n"
        "3. ENSURE robot.initialize() is called before any control commands.\n"
        "4. If using deprecated API (ArticulationView), replace with Articulation."
    ),
    "place_miss": (
        "The object was not placed accurately on the target tray.\n"
        "1. VERIFY the TRANSPORT phase moves the arm to exactly the place_target "
        "XY coordinates before lowering.\n"
        "2. LOWER the object more slowly during the PLACE phase -- reduce descent "
        "speed by 40%.\n"
        "3. OPEN fingers more gradually during RELEASE -- ramp torque down over "
        "30 steps instead of instant open.\n"
        "4. ADD a final positioning check: read end-effector XY and correct "
        "before release.\n"
        "5. REDUCE release height -- lower the object to within 2cm of the tray "
        "surface before opening fingers."
    ),
    "transport_drop": (
        "The object was dropped during the TRANSPORT phase (lateral movement).\n"
        "1. INCREASE finger torque by 20% during transport to compensate for "
        "lateral acceleration.\n"
        "2. REDUCE lateral movement speed by 30% -- use more sim steps.\n"
        "3. ADD a contact-force check during transport: if force drops below "
        "threshold, pause and re-grip.\n"
        "4. MAINTAIN the lift height during transport -- do not lower the arm "
        "until directly above the tray.\n"
        "5. SPLIT the transport into smaller waypoints (at least 3) instead of "
        "a single lateral move."
    ),
    "unknown": (
        "An unspecified error occurred.  Apply general hardening:\n"
        "1. VERIFY all joint names and USD prim paths are correct.\n"
        "2. ENSURE physics time step is 1/120 s.\n"
        "3. WRAP each phase in try/except, logging the error and returning "
        "success=False on failure.\n"
        "4. ADD robot.initialize() if missing.\n"
        "5. CHECK that sim_context.step() is used instead of time.sleep()."
    ),
}


class GraspCodeGenerator:
    """RAG-based engine that generates and iteratively corrects robot grasping code."""

    def __init__(self):
        self._llm = ChatOpenAI(
            model=settings.llm_model,
            temperature=settings.llm_temperature,
            max_tokens=settings.llm_max_tokens,
            api_key=settings.openai_api_key,
        )
        self._embeddings = OpenAIEmbeddings(api_key=settings.openai_api_key)
        self._parser = ManualParser()
        self._stores: dict[str, Chroma] = {}

    # ------------------------------------------------------------------
    # Manual ingestion & RAG retrieval
    # ------------------------------------------------------------------

    def ingest_manual(self, file_path: str) -> str:
        """Parse a robot manual PDF and store embeddings in ChromaDB.

        Args:
            file_path: Path to the PDF manual.

        Returns:
            Collection ID for the stored embeddings.
        """
        manual_data = self._parser.parse(file_path)
        collection_id = uuid.uuid4().hex[:12]

        documents = []
        for i, chunk in enumerate(manual_data.chunks):
            documents.append(
                Document(
                    page_content=chunk,
                    metadata={
                        "source": file_path,
                        "chunk_index": i,
                        "collection_id": collection_id,
                        "joint_names": ", ".join(manual_data.joint_names),
                        "control_functions": ", ".join(manual_data.control_functions),
                    },
                )
            )

        store = Chroma.from_documents(
            documents=documents,
            embedding=self._embeddings,
            collection_name=f"manual_{collection_id}",
            persist_directory=settings.chroma_persist_dir,
        )
        self._stores[collection_id] = store

        logger.info(
            "Ingested manual: %s -> collection %s (%d chunks)",
            file_path,
            collection_id,
            len(documents),
        )
        return collection_id

    def _get_store(self, collection_id: str) -> Chroma:
        """Retrieve or reconnect to a ChromaDB collection."""
        if collection_id in self._stores:
            return self._stores[collection_id]

        store = Chroma(
            collection_name=f"manual_{collection_id}",
            embedding_function=self._embeddings,
            persist_directory=settings.chroma_persist_dir,
        )
        self._stores[collection_id] = store
        return store

    def _query_manual(self, collection_id: str, query: str, k: int = 5) -> list[str]:
        """Query the vector store for relevant manual sections."""
        store = self._get_store(collection_id)
        docs = store.similarity_search(query, k=k)
        return [doc.page_content for doc in docs]

    def _query_manual_for_task(
        self,
        collection_id: str,
        cad_metadata: dict,
        robot_model: str,
    ) -> str:
        """Build task-specific RAG queries and return deduplicated context.

        Instead of generic keyword queries, this constructs queries that are
        grounded in the actual object dimensions and robot model so the
        retrieved manual sections are maximally relevant.
        """
        dims = cad_metadata.get("dimensions", {})
        max_dim = max(dims.get("x", 0.1), dims.get("y", 0.1), dims.get("z", 0.1))

        # Task-specific queries grounded in actual parameters
        queries = [
            f"{robot_model} gripper finger control torque force limits",
            f"{robot_model} joint names position limits articulation",
            f"grasping objects approximately {max_dim:.3f} m wide pick and place",
            f"{robot_model} end effector pose control inverse kinematics",
            f"Isaac Sim ArticulationController set_joint_position_targets",
            f"{robot_model} gripper opening width maximum aperture",
        ]

        manual_sections: list[str] = []
        for q in queries:
            sections = self._query_manual(collection_id, q, k=3)
            manual_sections.extend(sections)

        # Deduplicate while preserving order
        seen: set[str] = set()
        unique: list[str] = []
        for s in manual_sections:
            if s not in seen:
                seen.add(s)
                unique.append(s)

        # Limit to top-10 most relevant unique sections
        return "\n---\n".join(unique[:10])

    def _get_manual_metadata(self, collection_id: str) -> dict:
        """Extract joint names and control functions from stored metadata."""
        store = self._get_store(collection_id)
        results = store.get(limit=1, include=["metadatas"])
        if results and results["metadatas"]:
            meta = results["metadatas"][0]
            return {
                "joint_names": meta.get("joint_names", ""),
                "control_functions": meta.get("control_functions", ""),
            }
        return {"joint_names": "", "control_functions": ""}

    # ------------------------------------------------------------------
    # Code generation
    # ------------------------------------------------------------------

    async def generate_initial_code(
        self,
        cad_metadata: dict,
        robot_model: str,
        manual_collection_id: str,
    ) -> str:
        """Generate initial grasping code using RAG.

        Args:
            cad_metadata: Object metadata including dimensions, volume, center_of_mass.
            robot_model: Name/identifier of the robot model.
            manual_collection_id: ChromaDB collection ID for the robot manual.

        Returns:
            Generated Python code as a string.
        """
        # Optimised RAG retrieval with task-specific queries
        manual_context = self._query_manual_for_task(
            manual_collection_id, cad_metadata, robot_model,
        )

        meta = self._get_manual_metadata(manual_collection_id)
        dims = cad_metadata.get("dimensions", {})
        dim_z = dims.get("z", 0.1)

        volume = cad_metadata.get("volume") or (
            dims.get("x", 0.1) * dims.get("y", 0.1) * dim_z
        )
        estimated_mass = volume * 1200 if volume else 0.5
        min_torque = max(1.0, estimated_mass * 9.81 * 0.5)

        prompt = INITIAL_CODE_PROMPT.format(
            manual_context=manual_context,
            object_filename=cad_metadata.get("filename", "object"),
            dim_x=dims.get("x", 0.1),
            dim_y=dims.get("y", 0.1),
            dim_z=dim_z,
            volume=volume or "unknown",
            center_of_mass=cad_metadata.get("center_of_mass", "[0, 0, 0]"),
            estimated_mass=estimated_mass,
            robot_model=robot_model,
            joint_names=meta.get("joint_names", "not specified"),
            control_functions=meta.get("control_functions", "not specified"),
            object_height=dim_z / 2 + 0.01,
            min_torque=f"{min_torque:.1f}",
        )

        logger.info("Generating initial grasping code for robot=%s", robot_model)
        response = await self._llm.ainvoke(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ]
        )
        code = self._extract_code(response.content)

        # Validate the generated code before returning
        validation = await self._validate_code(code)
        if not validation["valid"]:
            logger.warning(
                "Generated code failed validation: %s -- requesting fix",
                validation["issues"],
            )
            code = await self._fix_validation_issues(code, validation)

        return code

    # ------------------------------------------------------------------
    # Code correction
    # ------------------------------------------------------------------

    async def correct_code(
        self,
        current_code: str,
        error_log: str,
        iteration: int,
        cad_metadata: dict,
    ) -> str:
        """Generate corrected code based on simulation error feedback.

        Args:
            current_code: The code that failed.
            error_log: Error/failure log from the simulation.
            iteration: Current correction iteration number.
            cad_metadata: Object metadata.

        Returns:
            Corrected Python code as a string.
        """
        error_type = self._classify_error(error_log)
        strategy = CORRECTION_STRATEGIES.get(
            error_type, CORRECTION_STRATEGIES["unknown"]
        )

        dims = cad_metadata.get("dimensions", {})
        volume = cad_metadata.get("volume") or (
            dims.get("x", 0.1) * dims.get("y", 0.1) * dims.get("z", 0.1)
        )
        estimated_mass = (volume * 1200) if volume else 0.5

        # Escalation note: later iterations get progressively more aggressive
        if iteration <= 3:
            escalation_note = "Early attempt -- make conservative, targeted fixes."
        elif iteration <= 10:
            escalation_note = (
                "Mid-stage attempt -- apply the correction strategy fully "
                "and consider alternative grasp approaches (e.g. different axis)."
            )
        else:
            escalation_note = (
                "Late-stage attempt -- make aggressive changes. Consider "
                "simplifying the entire approach: use direct joint position "
                "commands, reduce waypoints to 3, and increase all safety margins "
                "by 50%."
            )

        prompt = CORRECTION_PROMPT.format(
            current_code=current_code,
            error_log=error_log,
            iteration=iteration,
            max_iterations=settings.max_loop_iterations,
            dim_x=dims.get("x", 0.1),
            dim_y=dims.get("y", 0.1),
            dim_z=dims.get("z", 0.1),
            volume=volume or "unknown",
            estimated_mass=estimated_mass,
            error_type=error_type,
            correction_strategy=strategy,
            escalation_note=escalation_note,
        )

        logger.info(
            "Correcting code: iteration=%d, error_type=%s", iteration, error_type
        )
        response = await self._llm.ainvoke(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ]
        )
        code = self._extract_code(response.content)
        return code

    # ------------------------------------------------------------------
    # Feasibility assessment
    # ------------------------------------------------------------------

    async def assess_feasibility(
        self, cad_metadata: dict, robot_specs: dict
    ) -> dict:
        """Assess whether the grasping task is physically feasible.

        Args:
            cad_metadata: Object metadata with dimensions and volume.
            robot_specs: Robot capabilities (payload, grip force, opening, etc.).

        Returns:
            Dict with feasible (bool), reason (str), confidence (float).
        """
        dims = cad_metadata.get("dimensions", {})
        volume = cad_metadata.get("volume") or (
            dims.get("x", 0.1) * dims.get("y", 0.1) * dims.get("z", 0.1)
        )
        estimated_mass = volume * 1200 if volume else 0.5

        prompt = FEASIBILITY_PROMPT.format(
            dim_x=dims.get("x", 0.1),
            dim_y=dims.get("y", 0.1),
            dim_z=dims.get("z", 0.1),
            volume=volume or "unknown",
            estimated_mass=f"{estimated_mass:.2f}",
            robot_model=robot_specs.get("model", "unknown"),
            max_payload=robot_specs.get("max_payload", 5.0),
            max_grip_force=robot_specs.get("max_grip_force", 100.0),
            max_opening=robot_specs.get("max_opening", 0.15),
            joint_count=robot_specs.get("joint_count", 6),
        )

        response = await self._llm.ainvoke(prompt)
        result = self._parse_json_response(
            response.content,
            fallback={
                "feasible": True,
                "confidence": 0.5,
                "reason": "Could not parse LLM response; assuming feasible.",
            },
        )

        return {
            "feasible": result.get("feasible", True),
            "reason": result.get("reason", ""),
            "confidence": result.get("confidence", 0.5),
            "warnings": result.get("warnings", []),
            "recommended_torque": result.get("recommended_torque"),
        }

    # ------------------------------------------------------------------
    # Code validation
    # ------------------------------------------------------------------

    async def _validate_code(self, code: str) -> dict:
        """Run a lightweight LLM-based validation pass on generated code.

        Returns dict with 'valid' (bool), 'issues' (list[str]),
        and 'suggested_fix' (str).
        """
        prompt = CODE_VALIDATION_PROMPT.format(code=code)
        try:
            response = await self._llm.ainvoke(prompt)
            result = self._parse_json_response(
                response.content,
                fallback={"valid": True, "issues": [], "suggested_fix": ""},
            )
            return {
                "valid": result.get("valid", True),
                "issues": result.get("issues", []),
                "suggested_fix": result.get("suggested_fix", ""),
            }
        except Exception as exc:
            logger.warning("Code validation LLM call failed: %s", exc)
            return {"valid": True, "issues": [], "suggested_fix": ""}

    async def _fix_validation_issues(self, code: str, validation: dict) -> str:
        """Ask the LLM to fix specific validation issues in the code."""
        issues_text = "\n".join(f"- {issue}" for issue in validation["issues"])
        fix_prompt = (
            f"The following code has validation issues. Fix ONLY these issues "
            f"and return the corrected code.\n\n"
            f"## Issues\n{issues_text}\n\n"
            f"## Suggested fix\n{validation.get('suggested_fix', 'N/A')}\n\n"
            f"## Code\n```python\n{code}\n```\n\n"
            f"Return ONLY the corrected Python code."
        )
        try:
            response = await self._llm.ainvoke(
                [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": fix_prompt},
                ]
            )
            return self._extract_code(response.content)
        except Exception as exc:
            logger.warning("Validation fix LLM call failed: %s -- using original", exc)
            return code

    # ------------------------------------------------------------------
    # Error classification
    # ------------------------------------------------------------------

    @staticmethod
    def _classify_error(error_log: str) -> str:
        """Classify the simulation error using keyword matching.

        Uses expanded keyword sets and returns the category with the most
        keyword matches for better accuracy when logs contain mixed signals.
        """
        log_lower = error_log.lower()
        scores: dict[str, int] = {}
        for category, keywords in _ERROR_KEYWORDS.items():
            scores[category] = sum(1 for kw in keywords if kw in log_lower)

        best = max(scores, key=scores.get)  # type: ignore[arg-type]
        if scores[best] == 0:
            return "unknown"
        return best

    # ------------------------------------------------------------------
    # Response parsing helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_code(response_text: str) -> str:
        """Extract Python code from an LLM response, stripping markdown fences.

        Handles multiple code blocks by joining them, and tolerates extra text
        before/after fences.
        """
        text = response_text.strip()

        # Collect all fenced code blocks
        blocks: list[str] = []
        pattern = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.DOTALL)
        for m in pattern.finditer(text):
            block = m.group(1).strip()
            if block:
                blocks.append(block)

        if blocks:
            return "\n\n".join(blocks)

        # Fallback: legacy split-based extraction for single fence
        if "```python" in text:
            parts = text.split("```python", 1)
            code = parts[1].rsplit("```", 1)[0]
            return code.strip()
        if "```" in text:
            parts = text.split("```", 1)
            code = parts[1].rsplit("```", 1)[0]
            lines = code.split("\n", 1)
            if lines[0].strip() in ("python", "py", ""):
                return lines[1].strip() if len(lines) > 1 else ""
            return code.strip()

        # No fences -- assume the entire response is code
        return text

    @staticmethod
    def _parse_json_response(response_text: str, fallback: dict) -> dict:
        """Parse a JSON response from the LLM, tolerating markdown fences."""
        text = response_text.strip()

        # Try direct parse first
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Try extracting from markdown code block
        json_pattern = re.compile(r"```(?:json)?\s*\n(.*?)```", re.DOTALL)
        m = json_pattern.search(text)
        if m:
            try:
                return json.loads(m.group(1).strip())
            except json.JSONDecodeError:
                pass

        # Try finding a JSON object in the text
        brace_pattern = re.compile(r"\{.*\}", re.DOTALL)
        m = brace_pattern.search(text)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                pass

        logger.warning("Failed to parse JSON from LLM response")
        return fallback


# ---------------------------------------------------------------------------
# Module-level convenience functions (used by api/v1/generate.py)
# ---------------------------------------------------------------------------

_generator: GraspCodeGenerator | None = None
_manual_collections: dict[str, str] = {}  # manual_path -> collection_id


def _get_generator() -> GraspCodeGenerator:
    global _generator
    if _generator is None:
        _generator = GraspCodeGenerator()
    return _generator


async def generate_code(
    cad_metadata: dict,
    robot_model: str,
    manual_path: str | None = None,
    mode: str = "grasp_only",
    place_target: list[float] | None = None,
) -> str:
    """Generate initial grasping code.

    If a manual path is provided and hasn't been ingested yet, it will be
    ingested first. If no manual is provided, a default code template is
    generated using only the CAD metadata.
    """
    gen = _get_generator()

    if manual_path:
        if manual_path not in _manual_collections:
            try:
                collection_id = gen.ingest_manual(manual_path)
                _manual_collections[manual_path] = collection_id
            except Exception as exc:
                logger.warning("Failed to ingest manual: %s", exc)
                return _generate_default_code(cad_metadata, robot_model, mode=mode, place_target=place_target)

        collection_id = _manual_collections[manual_path]
        return await gen.generate_initial_code(cad_metadata, robot_model, collection_id)

    return _generate_default_code(cad_metadata, robot_model, mode=mode, place_target=place_target)


async def refine_code(
    current_code: str,
    error_log: str,
    cad_metadata: dict,
    robot_model: str,
    iteration: int | None = None,
    mode: str = "grasp_only",
    place_target: list[float] | None = None,
) -> str:
    """Refine existing code based on simulation error feedback.

    Args:
        current_code: The code that failed.
        error_log: Error/failure log from the simulation.
        cad_metadata: Object metadata.
        robot_model: Robot model name (unused here but kept for API compat).
        iteration: Explicit iteration number.  Falls back to a hash-derived
            value for backward compatibility.
        mode: "grasp_only" or "pick_and_place".
        place_target: [x, y, z] target position for pick-and-place.
    """
    gen = _get_generator()
    if iteration is None:
        iteration = len(current_code) % 20 + 1
    return await gen.correct_code(current_code, error_log, iteration, cad_metadata)


def _generate_default_code(
    cad_metadata: dict,
    robot_model: str,
    mode: str = "grasp_only",
    place_target: list[float] | None = None,
) -> str:
    """Generate a sensible default grasping script without RAG context."""
    dims = cad_metadata.get("dimensions", {})
    dim_x = dims.get("x", 0.05)
    dim_y = dims.get("y", 0.05)
    dim_z = dims.get("z", 0.05)
    max_dim = max(dim_x, dim_y, dim_z)
    # Allegro finger torque: scale with object mass estimate, clamp to 0.7 Nm max
    torque = min(0.7, max(0.2, max_dim * 5))
    approach_height = dim_z + 0.1

    if mode == "pick_and_place" and place_target is not None:
        return _generate_pick_and_place_code(torque, place_target)

    return f'''\
"""Auto-generated grasping code for Franka Panda + Allegro Hand."""
import time
import numpy as np
from omni.isaac.core.articulations import Articulation
from omni.isaac.core.prims import RigidPrim

def run_grasp_simulation(sim_context, robot_prim_path: str, object_prim_path: str) -> dict:
    """Execute a pick-and-place grasping sequence with Franka + Allegro."""
    logs = []
    start_time = time.time()
    dt = 1.0 / 120.0

    finger_torque = {torque:.3f}  # Nm per Allegro finger joint (max 0.7)
    hold_duration = 5.0

    try:
        # Initialise robot (23 DOF: 7 arm + 16 fingers)
        robot = Articulation(prim_path=robot_prim_path)
        robot.initialize()
        num_dof = robot.num_dof
        arm_idx = list(range(7))
        finger_idx = list(range(7, 23))

        # Get object position
        obj = RigidPrim(prim_path=object_prim_path)
        obj_pos, _ = obj.get_world_pose()

        # Phase 1: APPROACH -- move Franka arm to pre-grasp pose
        logs.append("Phase 1: moving Franka arm to pre-grasp pose")
        pre_grasp = np.zeros(num_dof)
        pre_grasp[0:7] = [0.0, -0.5, 0.0, -2.0, 0.0, 2.0, 0.785]
        pre_grasp[7:23] = 0.0  # fingers open
        robot.set_joint_position_targets(pre_grasp)
        for _ in range(180):
            sim_context.step(render=False)

        # Phase 2: DESCEND -- lower end-effector to object height
        logs.append("Phase 2: descending to grasp height")
        descend = pre_grasp.copy()
        descend[1] = -0.3
        descend[3] = -2.2
        robot.set_joint_position_targets(descend)
        for _ in range(120):
            sim_context.step(render=False)

        # Phase 3: GRASP -- ramp torque on all 4 Allegro fingers
        logs.append(f"Phase 3: closing Allegro fingers (torque={{finger_torque:.2f}} Nm)")
        for step in range(60):
            fraction = (step + 1) / 60.0
            efforts = np.zeros(num_dof)
            efforts[finger_idx] = finger_torque * fraction
            robot.set_joint_efforts(efforts)
            sim_context.step(render=False)

        # Verify contact via finger position change
        pos_after = robot.get_joint_positions()
        finger_moved = np.any(np.abs(pos_after[7:23]) > 0.05)
        if not finger_moved:
            logs.append("ERROR: no contact detected after grasp closure")
            return {{"success": False, "duration": time.time() - start_time, "logs": logs}}

        # Phase 4: LIFT -- raise arm while maintaining finger torques
        logs.append("Phase 4: lifting object")
        lift = descend.copy()
        lift[1] = -0.6
        lift[3] = -1.8
        for _ in range(180):
            efforts = np.zeros(num_dof)
            efforts[finger_idx] = finger_torque
            robot.set_joint_efforts(efforts)
            robot.set_joint_position_targets(lift)
            sim_context.step(render=False)

        # Phase 5: HOLD
        hold_steps = int(hold_duration / dt)
        logs.append(f"Phase 5: holding for {{hold_duration}} s ({{hold_steps}} steps)")
        for _ in range(hold_steps):
            efforts = np.zeros(num_dof)
            efforts[finger_idx] = finger_torque
            robot.set_joint_efforts(efforts)
            sim_context.step(render=False)

        logs.append("Grasp sequence completed successfully")
        return {{"success": True, "duration": time.time() - start_time, "logs": logs}}

    except Exception as e:
        logs.append(f"ERROR: {{str(e)}}")
        return {{"success": False, "duration": time.time() - start_time, "logs": logs}}
'''


def _generate_pick_and_place_code(torque: float, place_target: list[float]) -> str:
    """Generate an 8-phase pick-and-place script."""
    pt_x, pt_y, pt_z = place_target[0], place_target[1], place_target[2]
    return f'''\
"""Auto-generated pick-and-place code for Franka Panda + Allegro Hand."""
import time
import numpy as np
from omni.isaac.core.articulations import Articulation
from omni.isaac.core.prims import RigidPrim

def run_grasp_simulation(sim_context, robot_prim_path: str, object_prim_path: str, place_target=None) -> dict:
    """Execute an 8-phase pick-and-place sequence with Franka + Allegro."""
    logs = []
    trajectory = []
    start_time = time.time()
    dt = 1.0 / 120.0

    finger_torque = {torque:.3f}
    if place_target is None:
        place_target = [{pt_x}, {pt_y}, {pt_z}]

    try:
        robot = Articulation(prim_path=robot_prim_path)
        robot.initialize()
        num_dof = robot.num_dof
        finger_idx = list(range(7, 23))

        obj = RigidPrim(prim_path=object_prim_path)
        obj_pos, _ = obj.get_world_pose()
        trajectory.append({{"position": obj_pos.tolist(), "timestamp": 0.0}})

        # Phase 1: APPROACH
        logs.append("Phase 1 APPROACH: moving to pre-grasp pose")
        pre_grasp = np.zeros(num_dof)
        pre_grasp[0:7] = [0.0, -0.5, 0.0, -2.0, 0.0, 2.0, 0.785]
        robot.set_joint_position_targets(pre_grasp)
        for _ in range(180):
            sim_context.step(render=False)

        # Phase 2: DESCEND
        logs.append("Phase 2 DESCEND: lowering to grasp height")
        descend = pre_grasp.copy()
        descend[1] = -0.3
        descend[3] = -2.2
        robot.set_joint_position_targets(descend)
        for _ in range(120):
            sim_context.step(render=False)

        # Phase 3: GRASP
        logs.append("Phase 3 GRASP: closing fingers")
        for step in range(60):
            fraction = (step + 1) / 60.0
            efforts = np.zeros(num_dof)
            efforts[finger_idx] = finger_torque * fraction
            robot.set_joint_efforts(efforts)
            sim_context.step(render=False)

        pos_after = robot.get_joint_positions()
        finger_moved = np.any(np.abs(pos_after[7:23]) > 0.05)
        if not finger_moved:
            logs.append("ERROR: no contact detected after grasp closure")
            return {{"success": False, "duration": time.time() - start_time, "logs": logs, "object_trajectory": trajectory}}

        # Phase 4: LIFT
        logs.append("Phase 4 LIFT: raising object")
        lift = descend.copy()
        lift[1] = -0.6
        lift[3] = -1.8
        for _ in range(180):
            efforts = np.zeros(num_dof)
            efforts[finger_idx] = finger_torque
            robot.set_joint_efforts(efforts)
            robot.set_joint_position_targets(lift)
            sim_context.step(render=False)
        obj_pos, _ = obj.get_world_pose()
        trajectory.append({{"position": obj_pos.tolist(), "timestamp": time.time() - start_time}})

        # Phase 5: TRANSPORT
        logs.append(f"Phase 5 TRANSPORT: moving to place target ({{place_target[0]:.2f}}, {{place_target[1]:.2f}})")
        transport = lift.copy()
        transport[0] = 0.3  # adjust base joint for lateral reach
        for step in range(240):
            efforts = np.zeros(num_dof)
            efforts[finger_idx] = finger_torque * 1.1  # extra grip during transport
            robot.set_joint_efforts(efforts)
            robot.set_joint_position_targets(transport)
            sim_context.step(render=False)
            if step % 60 == 0:
                obj_pos, _ = obj.get_world_pose()
                trajectory.append({{"position": obj_pos.tolist(), "timestamp": time.time() - start_time}})

        # Phase 6: PLACE
        logs.append("Phase 6 PLACE: lowering into tray")
        place_pose = transport.copy()
        place_pose[1] = -0.2
        place_pose[3] = -2.3
        for _ in range(180):
            efforts = np.zeros(num_dof)
            efforts[finger_idx] = finger_torque
            robot.set_joint_efforts(efforts)
            robot.set_joint_position_targets(place_pose)
            sim_context.step(render=False)
        obj_pos, _ = obj.get_world_pose()
        trajectory.append({{"position": obj_pos.tolist(), "timestamp": time.time() - start_time}})

        # Phase 7: RELEASE
        logs.append("Phase 7 RELEASE: opening fingers gradually")
        for step in range(30):
            fraction = 1.0 - (step + 1) / 30.0
            efforts = np.zeros(num_dof)
            efforts[finger_idx] = finger_torque * fraction
            robot.set_joint_efforts(efforts)
            sim_context.step(render=False)
        # Fully open
        robot.set_joint_efforts(np.zeros(num_dof))
        for _ in range(60):
            sim_context.step(render=False)

        # Phase 8: RETRACT
        logs.append("Phase 8 RETRACT: raising arm away")
        retract = place_pose.copy()
        retract[1] = -0.6
        retract[3] = -1.5
        robot.set_joint_position_targets(retract)
        for _ in range(120):
            sim_context.step(render=False)

        obj_pos, _ = obj.get_world_pose()
        trajectory.append({{"position": obj_pos.tolist(), "timestamp": time.time() - start_time}})
        logs.append("Pick-and-place sequence completed successfully")
        return {{"success": True, "duration": time.time() - start_time, "logs": logs, "object_trajectory": trajectory}}

    except Exception as e:
        logs.append(f"ERROR: {{str(e)}}")
        return {{"success": False, "duration": time.time() - start_time, "logs": logs, "object_trajectory": trajectory}}
'''
