# OpenRobot simulation environment
# Provides a containerized Ubuntu 22.04 + Python 3.10 stack for running
# the MuJoCo-based OpenRobotDemo without real hardware.
#
# Build:
#   docker build -t openrobot-sim .
#
# Run the minimal pick-and-place demo:
#   docker run --rm -v $(pwd)/OpenRobotDemo/data:/app/OpenRobotDemo/data openrobot-sim
#
# Run interactively:
#   docker run --rm -it -v $(pwd)/OpenRobotDemo/data:/app/OpenRobotDemo/data openrobot-sim bash

FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3.10-venv \
    python3-pip \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements first for better layer caching
COPY OpenRobotDemo/requirements.txt ./OpenRobotDemo/requirements.txt

# Create virtual environment and install Python dependencies
RUN python3.10 -m venv /app/.venv && \
    /app/.venv/bin/pip install --upgrade pip && \
    /app/.venv/bin/pip install -r /app/OpenRobotDemo/requirements.txt

# Copy project source
COPY OpenRobotDemo/ ./OpenRobotDemo/
COPY openrobot_core/ ./openrobot_core/
COPY sim/ ./sim/
COPY README.md ./

# Activate venv by default
ENV PATH="/app/.venv/bin:${PATH}"

WORKDIR /app/OpenRobotDemo

# Default command: run the minimal simulation demo
CMD ["python", "examples/sim_pick_place.py"]
