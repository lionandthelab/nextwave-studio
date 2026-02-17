"""End-to-end HTTP API test for pick-and-place pipeline."""
import requests
import time
import json
import sys

BASE = "http://localhost:8009"

# Step 1: List presets
print("=== Step 1: List presets ===")
res = requests.get(f"{BASE}/api/v1/upload/presets/list")
print(f"Status: {res.status_code}")
presets = res.json()
for p in presets:
    print(f"  {p['name']}: {p['label']} ({p['size_bytes']} bytes)")

# Step 2: Load the box_6cm preset
print("\n=== Step 2: Load preset box_6cm ===")
res = requests.post(f"{BASE}/api/v1/upload/preset/box_6cm")
print(f"Status: {res.status_code}")
upload = res.json()
print(f"  file_id: {upload['id']}")
print(f"  filename: {upload['filename']}")
cad_file_id = upload["id"]

# Step 3: Check file metadata
print("\n=== Step 3: Check CAD metadata ===")
res = requests.get(f"{BASE}/api/v1/upload/{cad_file_id}")
meta = res.json()
print(f"  dimensions: {meta.get('cad_metadata', {}).get('dimensions')}")

# Step 4: Start PnP generation
print("\n=== Step 4: Start PnP generation (100 iterations max) ===")
res = requests.post(f"{BASE}/api/v1/generate/start", json={
    "cad_file_id": cad_file_id,
    "mode": "pick_and_place",
    "place_target": [0.5, 0.4, 0.05],
    "max_iterations": 100,
    "success_threshold": 3,
})
print(f"Status: {res.status_code}")
if res.status_code != 200:
    print(f"  Error: {res.text}")
    sys.exit(1)
loop_status = res.json()
session_id = loop_status["session_id"]
print(f"  session_id: {session_id}")
print(f"  status: {loop_status['status']}")

# Step 5: Poll for completion
print("\n=== Step 5: Polling for results ===")
for poll in range(200):
    time.sleep(2)
    res = requests.get(f"{BASE}/api/v1/generate/status/{session_id}")
    if res.status_code != 200:
        print(f"  Poll error: {res.status_code}")
        continue
    status = res.json()
    st = status["status"]
    it = status.get("current_iteration", 0)
    results = status.get("results", [])

    if results:
        last = results[-1]
        checks = last.get("checks", {})
        passed = sum(1 for v in checks.values() if v)
        total = len(checks)
        print(f"  Iter {it}: {st} | checks: {passed}/{total} | success: {last.get('success')}")
        if not last.get("success") and last.get("error_log"):
            err_line = last["error_log"].split("\n")[0]
            print(f"    -> {err_line[:120]}")

    if st in ("success", "failed", "stopped"):
        print(f"\n=== FINAL: {st} after {it} iterations ===")
        if results:
            successes = sum(1 for r in results if r.get("success"))
            failures = len(results) - successes
            print(f"  Total successes: {successes}/{len(results)}")
            print(f"  Total failures: {failures}/{len(results)}")
            last = results[-1]
            print(f"  Last checks: {json.dumps(last.get('checks', {}), indent=2)}")
            if last.get("place_accuracy") is not None:
                print(f"  Place accuracy: {last['place_accuracy']:.4f}m")
        break
else:
    print("Timeout waiting for completion")
    sys.exit(1)

if st == "success":
    print("\nPICK-AND-PLACE PIPELINE: SUCCESS")
    sys.exit(0)
else:
    print(f"\nPICK-AND-PLACE PIPELINE: {st}")
    sys.exit(1)
