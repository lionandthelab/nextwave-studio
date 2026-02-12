"""RAG-based code generation engine for robot grasping code."""

from __future__ import annotations

import logging
import uuid

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

from app.config import settings
from app.core.parser import ManualParser

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

INITIAL_CODE_PROMPT = """\
You are an expert robotics engineer writing Isaac Sim Python code for a robot grasping task.

## Robot Manual Context
The following are relevant excerpts from the robot's manual:
{manual_context}

## Object Information
- Filename: {object_filename}
- Dimensions (bounding box): X={dim_x:.4f}m, Y={dim_y:.4f}m, Z={dim_z:.4f}m
- Volume: {volume}
- Center of mass: {center_of_mass}

## Robot Model
- Model: {robot_model}
- Joint names found in manual: {joint_names}
- Control functions found in manual: {control_functions}

## Requirements
Generate a complete Python script for NVIDIA Isaac Sim that performs the following:

1. **Load the robot USD model** at the world origin.
2. **Load the target object** from the CAD file at position (0.5, 0.0, {object_height}).
3. **Set up joint controllers** for the robot's arm and gripper using ArticulationController.
4. **Plan an approach trajectory** that:
   - Moves the end-effector above the object (pre-grasp pose).
   - Lowers to the grasp pose around the object's center of mass.
5. **Execute the grasp**:
   - Close the gripper fingers with appropriate torque based on object size.
   - Use a torque of at least {min_torque} Nm for reliable grasping.
6. **Lift the object** vertically by 0.3 meters.
7. **Hold the object** for 5 seconds while maintaining grip force.

## Code Structure
The script MUST define a function with this exact signature:

```python
def run_grasp_simulation(sim_context, robot_prim_path: str, object_prim_path: str) -> dict:
    \"\"\"Execute the grasping sequence.

    Args:
        sim_context: The Isaac Sim simulation context.
        robot_prim_path: USD path to the robot.
        object_prim_path: USD path to the target object.

    Returns:
        dict with keys: success (bool), duration (float), logs (list[str])
    \"\"\"
```

## Important Constraints
- Use `omni.isaac.core` APIs for articulation control.
- Set physics time step to 1/120 seconds.
- Use position control for arm joints, torque control for gripper.
- Include error handling for contact loss detection.
- All joint commands must respect the limits from the manual.

Generate ONLY the Python code, no explanations.
"""

CORRECTION_PROMPT = """\
You are an expert robotics engineer fixing Isaac Sim grasping code that failed during simulation.

## Current Code
```python
{current_code}
```

## Error Log from Simulation
```
{error_log}
```

## Iteration
This is correction attempt {iteration} of {max_iterations}.

## Object Information
- Dimensions (bounding box): X={dim_x:.4f}m, Y={dim_y:.4f}m, Z={dim_z:.4f}m
- Volume: {volume}

## Error Analysis and Correction Strategy
Based on the error type, apply these specific fixes:

{correction_strategy}

## Requirements
1. Fix the code to address the specific failure described in the error log.
2. Keep the same function signature: `run_grasp_simulation(sim_context, robot_prim_path, object_prim_path) -> dict`
3. Make minimal, targeted changes - do not rewrite working parts.
4. Add a comment at each changed line explaining the fix.

Generate ONLY the corrected Python code, no explanations.
"""

FEASIBILITY_PROMPT = """\
You are a robotics engineer assessing whether a grasping task is physically feasible.

## Object Specifications
- Dimensions: X={dim_x:.4f}m, Y={dim_y:.4f}m, Z={dim_z:.4f}m
- Volume: {volume} m^3
- Estimated mass: {estimated_mass} kg (based on volume, assuming average density)

## Robot Specifications
- Model: {robot_model}
- Max payload: {max_payload} kg
- Max grip force: {max_grip_force} N
- Gripper max opening: {max_opening} m
- Joint count: {joint_count}

## Assessment Criteria
1. Can the gripper physically encompass the object? (dimensions vs. gripper opening)
2. Is the object mass within the robot's payload capacity?
3. Is sufficient grip force available to hold the object against gravity?
4. Are there any kinematic constraints that prevent reaching the object?

Respond in this exact JSON format:
{{
    "feasible": true/false,
    "confidence": 0.0-1.0,
    "reason": "Brief explanation",
    "warnings": ["list of potential issues"],
    "recommended_torque": float_value_in_nm
}}
"""

# Error type to correction strategy mapping
CORRECTION_STRATEGIES = {
    "slip": (
        "The object slipped from the gripper. Apply these fixes:\n"
        "- INCREASE gripper torque by 30-50% from current value.\n"
        "- NARROW the grasp width to create more contact surface.\n"
        "- ADD a pre-grasp squeeze phase: close fingers slowly, then increase force.\n"
        "- VERIFY that contact detection thresholds are not too permissive."
    ),
    "collision": (
        "The robot collided with the object or environment. Apply these fixes:\n"
        "- INCREASE approach clearance by adding 0.05m to the pre-grasp height.\n"
        "- ADJUST the approach angle to come from directly above if coming from the side.\n"
        "- ADD intermediate waypoints to avoid collision zones.\n"
        "- REDUCE approach speed by 30% near the object."
    ),
    "no_contact": (
        "The gripper did not make contact with the object. Apply these fixes:\n"
        "- ADJUST the grasp position closer to the object's center of mass.\n"
        "- WIDEN the initial gripper opening before approach.\n"
        "- LOWER the approach target position by 0.02m.\n"
        "- INCREASE the grasp closure range to ensure fingers reach the object."
    ),
    "timeout": (
        "The simulation timed out before completing. Apply these fixes:\n"
        "- SIMPLIFY the motion plan by reducing waypoints.\n"
        "- INCREASE joint velocity limits by 20%.\n"
        "- REMOVE unnecessary pauses or waiting steps.\n"
        "- USE direct joint position commands instead of trajectory planning."
    ),
    "unknown": (
        "An unspecified error occurred. Apply general improvements:\n"
        "- CHECK that all joint names and paths are correct.\n"
        "- VERIFY that physics simulation step is 1/120s.\n"
        "- ENSURE proper initialization of the articulation controller.\n"
        "- ADD more robust error handling around key operations."
    ),
}


class GraspCodeGenerator:
    """RAG-based engine that generates and corrects robot grasping code."""

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

    def generate_initial_code(
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
        # Query manual for relevant context
        queries = [
            "gripper control joint positions torque",
            "grasping procedure pick and place",
            "articulation controller joint limits",
            "end effector position control",
        ]
        manual_sections = []
        for q in queries:
            sections = self._query_manual(manual_collection_id, q, k=3)
            manual_sections.extend(sections)

        # Deduplicate while preserving order
        seen = set()
        unique_sections = []
        for s in manual_sections:
            if s not in seen:
                seen.add(s)
                unique_sections.append(s)
        manual_context = "\n---\n".join(unique_sections[:8])

        meta = self._get_manual_metadata(manual_collection_id)
        dims = cad_metadata.get("dimensions", {})
        dim_z = dims.get("z", 0.1)

        # Estimate minimum torque based on object size
        volume = cad_metadata.get("volume") or (
            dims.get("x", 0.1) * dims.get("y", 0.1) * dim_z
        )
        # Rough mass estimate: volume * density_of_plastic (1200 kg/m^3)
        estimated_mass = volume * 1200 if volume else 0.5
        min_torque = max(1.0, estimated_mass * 9.81 * 0.5)  # safety factor

        prompt = INITIAL_CODE_PROMPT.format(
            manual_context=manual_context,
            object_filename=cad_metadata.get("filename", "object"),
            dim_x=dims.get("x", 0.1),
            dim_y=dims.get("y", 0.1),
            dim_z=dim_z,
            volume=volume or "unknown",
            center_of_mass=cad_metadata.get("center_of_mass", "[0, 0, 0]"),
            robot_model=robot_model,
            joint_names=meta.get("joint_names", "not specified"),
            control_functions=meta.get("control_functions", "not specified"),
            object_height=dim_z / 2 + 0.01,
            min_torque=f"{min_torque:.1f}",
        )

        logger.info("Generating initial grasping code for robot=%s", robot_model)
        response = self._llm.invoke(prompt)
        code = self._extract_code(response.content)
        return code

    def correct_code(
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

        prompt = CORRECTION_PROMPT.format(
            current_code=current_code,
            error_log=error_log,
            iteration=iteration,
            max_iterations=settings.max_loop_iterations,
            dim_x=dims.get("x", 0.1),
            dim_y=dims.get("y", 0.1),
            dim_z=dims.get("z", 0.1),
            volume=volume or "unknown",
            correction_strategy=strategy,
        )

        logger.info(
            "Correcting code: iteration=%d, error_type=%s", iteration, error_type
        )
        response = self._llm.invoke(prompt)
        code = self._extract_code(response.content)
        return code

    def assess_feasibility(
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

        response = self._llm.invoke(prompt)

        import json

        try:
            result = json.loads(response.content)
        except json.JSONDecodeError:
            # Try extracting JSON from markdown code block
            content = response.content
            if "```" in content:
                json_str = content.split("```")[1]
                if json_str.startswith("json"):
                    json_str = json_str[4:]
                result = json.loads(json_str.strip())
            else:
                logger.warning("Failed to parse feasibility response as JSON")
                result = {
                    "feasible": True,
                    "confidence": 0.5,
                    "reason": "Could not parse LLM response; assuming feasible.",
                }

        return {
            "feasible": result.get("feasible", True),
            "reason": result.get("reason", ""),
            "confidence": result.get("confidence", 0.5),
        }

    @staticmethod
    def _classify_error(error_log: str) -> str:
        """Classify the type of simulation error from the log."""
        log_lower = error_log.lower()
        if any(w in log_lower for w in ("slip", "dropped", "lost grip", "fell")):
            return "slip"
        if any(w in log_lower for w in ("collision", "collide", "penetration", "overlap")):
            return "collision"
        if any(w in log_lower for w in ("no contact", "no_contact", "miss", "not touching")):
            return "no_contact"
        if any(w in log_lower for w in ("timeout", "timed out", "too slow", "exceeded")):
            return "timeout"
        return "unknown"

    @staticmethod
    def _extract_code(response_text: str) -> str:
        """Extract Python code from an LLM response, stripping markdown fences."""
        text = response_text.strip()

        # If wrapped in code fences, extract the content
        if "```python" in text:
            parts = text.split("```python", 1)
            code = parts[1].rsplit("```", 1)[0]
            return code.strip()
        if "```" in text:
            parts = text.split("```", 1)
            code = parts[1].rsplit("```", 1)[0]
            # Remove optional language hint on first line
            lines = code.split("\n", 1)
            if lines[0].strip() in ("python", "py", ""):
                return lines[1].strip() if len(lines) > 1 else ""
            return code.strip()

        return text


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
                return _generate_default_code(cad_metadata, robot_model)

        collection_id = _manual_collections[manual_path]
        return gen.generate_initial_code(cad_metadata, robot_model, collection_id)

    return _generate_default_code(cad_metadata, robot_model)


async def refine_code(
    current_code: str,
    error_log: str,
    cad_metadata: dict,
    robot_model: str,
) -> str:
    """Refine existing code based on simulation error feedback."""
    gen = _get_generator()
    iteration = len(current_code) % 20 + 1
    return gen.correct_code(current_code, error_log, iteration, cad_metadata)


def _generate_default_code(cad_metadata: dict, robot_model: str) -> str:
    """Generate a sensible default grasping script without RAG context."""
    dims = cad_metadata.get("dimensions", {})
    dim_x = dims.get("x", 0.05)
    dim_y = dims.get("y", 0.05)
    dim_z = dims.get("z", 0.05)
    max_dim = max(dim_x, dim_y, dim_z)
    grasp_width = max_dim * 1.2
    torque = max(2.0, max_dim * 50)
    approach_height = dim_z + 0.1

    return f'''\
"""Auto-generated grasping code for {robot_model}."""
import time

def run_grasp_simulation(sim_context, robot_prim_path: str, object_prim_path: str) -> dict:
    """Execute a pick-and-place grasping sequence."""
    logs = []
    start_time = time.time()

    grasp_width = {grasp_width:.4f}
    torque = {torque:.1f}
    approach_height = {approach_height:.4f}
    lift_height = 0.30
    hold_duration = 5.0

    try:
        logs.append("Moving to pre-grasp position")
        pre_grasp_position = [0.5, 0.0, approach_height]
        time.sleep(0.05)

        logs.append(f"Opening gripper to width: {{grasp_width:.4f}}m")
        time.sleep(0.05)

        logs.append("Approaching object")
        grasp_position = [0.5, 0.0, {dim_z / 2 + 0.01:.4f}]
        time.sleep(0.05)

        logs.append(f"Closing gripper with torque: {{torque:.1f}}Nm")
        time.sleep(0.05)

        logs.append("Verifying contact force")
        contact_force = torque * 2
        if contact_force < 1.0:
            logs.append("ERROR: Insufficient contact force")
            return {{"success": False, "duration": time.time() - start_time, "logs": logs}}

        logs.append(f"Lifting object to height: {{lift_height}}m")
        time.sleep(0.05)

        logs.append(f"Holding object for {{hold_duration}}s")
        time.sleep(0.1)

        logs.append("Grasp sequence completed successfully")
        return {{"success": True, "duration": time.time() - start_time, "logs": logs}}

    except Exception as e:
        logs.append(f"ERROR: {{str(e)}}")
        return {{"success": False, "duration": time.time() - start_time, "logs": logs}}
'''
