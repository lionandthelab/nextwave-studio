#!/bin/bash
# AutoGrip-Sim Entrypoint
# Starts Xvfb + FastAPI sim_server alongside Isaac Sim (or standalone in mock mode)

set -e

# Start Xvfb virtual display for headless GPU rendering + WebRTC streaming
echo "[AutoGrip] Starting Xvfb virtual display..."
Xvfb :1 -screen 0 1280x720x24 +extension GLX &
export DISPLAY=:1
sleep 2

echo "[AutoGrip] Starting sim_server on port 9090..."

# Use Isaac Sim's bundled Python if available, otherwise system python
if [ -f /isaac-sim/python.sh ]; then
    exec /isaac-sim/python.sh /autogrip-sim/sim_scripts/sim_server.py
else
    exec python3 /autogrip-sim/sim_scripts/sim_server.py
fi
