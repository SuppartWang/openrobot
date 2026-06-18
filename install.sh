#!/usr/bin/env bash
# OpenRobot development environment installer.
# Sets up a Python virtual environment and installs dependencies for the
# OpenRobotDemo simulation stack. Run this from the repository root.

set -e

REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
VENV_DIR="${REPO_ROOT}/.venv"
PYTHON="${PYTHON:-python3}"

echo "============================================================"
echo " OpenRobot Development Environment Setup"
echo "============================================================"
echo ""

# Check Python version
PYTHON_VERSION=$($PYTHON -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
echo "Detected Python ${PYTHON_VERSION}"

# Create virtual environment
if [ -d "$VENV_DIR" ]; then
    echo "Virtual environment already exists at ${VENV_DIR}"
else
    echo "Creating virtual environment at ${VENV_DIR}..."
    $PYTHON -m venv "$VENV_DIR"
fi

# Activate virtual environment
source "${VENV_DIR}/bin/activate"

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install OpenRobotDemo dependencies (simulation-only by default)
echo "Installing OpenRobotDemo simulation dependencies..."
pip install -r "${REPO_ROOT}/OpenRobotDemo/requirements.txt"

# Optional: install root-level dependencies (includes full perception/cognition stack)
if [ "${INSTALL_FULL:-0}" = "1" ]; then
    echo "Installing full project dependencies..."
    pip install -r "${REPO_ROOT}/requirements.txt"
fi

echo ""
echo "============================================================"
echo " Installation complete!"
echo "============================================================"
echo ""
echo "Activate the environment with:"
echo "  source ${VENV_DIR}/bin/activate"
echo ""
echo "Run the minimal simulation demo:"
echo "  cd OpenRobotDemo && python examples/sim_pick_place.py"
echo ""
echo "Run the full-stack simulation demo (requires API keys for LLM/VLM):"
echo "  cd OpenRobotDemo && python scripts/demo_simulation_full_stack.py"
echo ""
