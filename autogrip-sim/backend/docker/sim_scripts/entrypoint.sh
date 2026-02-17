#!/bin/bash
# AutoGrip-Sim Entrypoint
# Starts the FastAPI sim_server alongside Isaac Sim (or standalone in mock mode)

set -e

echo "[AutoGrip] Starting sim_server on port 9090..."

# Use Isaac Sim's bundled Python if available, otherwise system python
if [ -f /isaac-sim/python.sh ]; then
    exec /isaac-sim/python.sh /autogrip-sim/sim_scripts/sim_server.py
else
    exec python3 /autogrip-sim/sim_scripts/sim_server.py
fi
