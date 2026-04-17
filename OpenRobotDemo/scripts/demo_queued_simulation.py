"""
Run the MuJoCo simulation demo through RobotQueue.

This demonstrates the Phase 1 integration:
- SQLite persistence for episodes, steps, and state snapshots
- Serial task execution via RobotQueue
"""

import os
import sys
import time
import argparse
from pathlib import Path

from dotenv import load_dotenv

_project_root = os.path.join(os.path.dirname(__file__), "..")
load_dotenv(Path(_project_root) / ".env")
sys.path.insert(0, _project_root)
sys.path.insert(0, os.path.join(os.path.dirname(_project_root), "openrobot_core"))

from openrobot_demo.persistence.db import init_database
from openrobot_demo.runtime.queue import RobotQueue
from scripts.demo_simulation_full_stack import run_simulation
from openrobot_demo.persistence.db import EpisodeRecorder


def main():
    parser = argparse.ArgumentParser(description="Queued MuJoCo simulation demo")
    parser.add_argument(
        "--instruction",
        type=str,
        default="捡起黄色方块，而后扔向远处.",
        help="Task instruction",
    )
    parser.add_argument(
        "--no-block",
        action="store_true",
        help="Enqueue and exit without waiting for completion",
    )
    args = parser.parse_args()

    db = init_database()
    queue = RobotQueue(db=db, max_retries=2, base_retry_delay=2.0)

    def _task(instr: str, recorder: EpisodeRecorder):
        return run_simulation(recorder=recorder, instruction=instr)

    episode_id = queue.enqueue(args.instruction, _task)
    print(f"\n📋 Task enqueued: {episode_id}")

    if not args.no_block:
        print("⏳ Waiting for task to complete...")
        while True:
            status = queue.get_status(episode_id)
            if status is None:
                break
            if status["status"] in ("completed", "failed"):
                print(f"\n🏁 Task {episode_id} finished with status: {status['status']}")
                if status.get("error"):
                    print(f"   Error: {status['error']}")
                break
            time.sleep(0.5)
        queue.shutdown(wait=True)
        
        # Print a summary from the database
        episode = db.get_episode(episode_id)
        steps = db.get_episode_steps(episode_id)
        print(f"\n📊 Episode summary:")
        print(f"   Instruction: {episode['instruction']}")
        print(f"   Status:      {episode['status']}")
        print(f"   Steps:       {len(steps)}")
        for s in steps:
            skill = s.get("skill_name") or "plan"
            ok = "✅" if s.get("success") else "❌"
            print(f"   {ok} Step {s['step_idx']}: {skill} — {s.get('message', '')}")
    else:
        print("Task is running in the background. Use the database to inspect progress.")


if __name__ == "__main__":
    main()
