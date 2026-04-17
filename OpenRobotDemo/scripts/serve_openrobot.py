"""
OpenRobotDemo server entry point with pluggable input channels.

Supports:
    --channel cli   : Interactive command-line input
    --channel http  : HTTP API (POST /task, GET /status, GET /health)
    --channel voice : Voice input placeholder (implement ASR to activate)

Example:
    python scripts/serve_openrobot.py --channel cli
    python scripts/serve_openrobot.py --channel http --http-port 8080
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

_project_root = os.path.join(os.path.dirname(__file__), "..")
load_dotenv(Path(_project_root) / ".env")
sys.path.insert(0, _project_root)
sys.path.insert(0, os.path.join(os.path.dirname(_project_root), "openrobot_core"))

from openrobot_demo.channels import (
    CLIChannel,
    HTTPChannel,
    VoiceChannel,
    register_channel,
    create_channel,
)
from openrobot_demo.persistence.db import init_database, EpisodeRecorder
from openrobot_demo.runtime.queue import RobotQueue
from scripts.demo_simulation_full_stack import run_simulation

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="OpenRobotDemo Channel Server")
    parser.add_argument(
        "--channel",
        choices=["cli", "http", "voice"],
        default="cli",
        help="Input channel to use",
    )
    parser.add_argument(
        "--http-port",
        type=int,
        default=8080,
        help="HTTP server port (only used with --channel http)",
    )
    args = parser.parse_args()

    # Register built-in channels
    register_channel("cli", CLIChannel)
    register_channel("http", HTTPChannel)
    register_channel("voice", VoiceChannel)

    # Init persistence and queue
    db = init_database()
    queue = RobotQueue(db=db, max_retries=2, base_retry_delay=2.0)

    def task_fn(instruction: str, recorder: EpisodeRecorder):
        """The actual robot task executed by the queue."""
        return run_simulation(recorder=recorder, instruction=instruction)

    def on_message(instruction: str) -> str:
        """Enqueue a new task and return an acknowledgment."""
        episode_id = queue.enqueue(instruction, task_fn)
        return json.dumps(
            {
                "episode_id": episode_id,
                "status": "queued",
                "message": f"Task queued. Query GET /status?episode_id={episode_id}",
            },
            ensure_ascii=False,
        )

    # Instantiate channel
    if args.channel == "http":
        channel = create_channel("http", port=args.http_port)
        channel.bind_status_lookup(queue.get_status)
    elif args.channel == "voice":
        channel = create_channel("voice")
    else:
        channel = create_channel("cli")

    channel.start(on_message)

    print("\n✅ OpenRobotDemo server is running. Press Ctrl+C to stop.\n")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n🛑 Shutting down...")
    finally:
        channel.stop()
        queue.shutdown(wait=True, timeout=30.0)
        db.close()
        print("👋 Server stopped.")


if __name__ == "__main__":
    main()
