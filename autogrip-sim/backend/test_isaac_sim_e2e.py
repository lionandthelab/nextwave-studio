"""Quick end-to-end test: backend → Isaac Sim container via HTTP."""

import asyncio
import sys

from app.sim_interface.runner import run_simulation, validate_result


SIMPLE_GRASP_CODE = """
def run_grasp_simulation(sim_context, robot_prim_path, object_prim_path, place_target=None):
    import time

    # Step physics to let object settle
    for i in range(240):  # 2 seconds at 120Hz
        sim_context.step(render=False)

    return {"status": "ok"}
"""


async def main():
    print("=== Testing backend → Isaac Sim pipeline ===")
    print()

    # Test 1: Grasp-only mode
    print("[1] Running grasp-only simulation...")
    result = await run_simulation(
        code=SIMPLE_GRASP_CODE,
        cad_path="/autogrip-sim/cad_files/2890ae3bd1fa4f87bf5957544ba1b256.stl",
        robot_model="franka_allegro",
    )
    print(f"    Success: {result.get('success')}")
    print(f"    Duration: {result.get('duration', 0):.2f}s")
    pos = result.get("object_final_state", {}).get("position", [0, 0, 0])
    print(f"    Object position: [{pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f}]")
    print(f"    Joint count: {len(result.get('joint_states', {}))}")
    print(f"    Frames: {len(result.get('frames', []))}")
    print()

    # Validate
    print("[2] Validating result...")
    checks, error_log = await validate_result(result)
    print(f"    Checks: {checks}")
    if error_log:
        print(f"    Errors: {error_log[:200]}")
    else:
        print("    All checks passed!")
    print()

    # Test 2: Pick-and-place mode
    print("[3] Running pick-and-place simulation...")
    result2 = await run_simulation(
        code=SIMPLE_GRASP_CODE,
        cad_path="/autogrip-sim/cad_files/2890ae3bd1fa4f87bf5957544ba1b256.stl",
        robot_model="franka_allegro",
        place_target=[0.5, 0.4, 0.0],
    )
    print(f"    Success: {result2.get('success')}")
    pos2 = result2.get("object_final_state", {}).get("position", [0, 0, 0])
    print(f"    Object position: [{pos2[0]:.4f}, {pos2[1]:.4f}, {pos2[2]:.4f}]")
    print()

    print("=== Done ===")
    return 0 if result.get("success") else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
