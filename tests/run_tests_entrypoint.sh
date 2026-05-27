#!/bin/bash
set -e

# Give Xvfb a moment to initialize
sleep 2

# Activate Python environment
source /workspace/venv/bin/activate

export DEPTHAI_TELEMETRY_API_KEY='phc_zZmy6ywwrAmtdhXyZcy4fTmpdJxnh7QcR55XvZoZCmqz'
export DEPTHAI_TELEMETRY_URL='https://b.luxonis.com'

# Run your tests with passed arguments (e.g., rvc2 or rvc4)
cd /workspace/tests
echo "Running tests with args: $@"
python3 run_tests.py "--$@"
