"""VLA Policy Executor: end-to-end vision-language-action skill.

This skill serves as a bridge to a fine-tuned VLA model (e.g. pi05) that
outputs an action sequence given an RGB image and a language instruction.
In the current MVP the model inference stub is left for the user to fill in.
"""

import logging
from typing import Any, Dict, Optional

import numpy as np

from openrobot_demo.skills.base import SkillInterface

logger = logging.getLogger(__name__)


class VLAPolicyExecutor(SkillInterface):
    """
    Execute a single robotic skill using a fine-tuned VLA policy.

    Typical usage in ReAct:
        {"skill": "vla_policy_executor", "args": {"instruction": "夹布料", "rgb_frame": "rgb_frame"}}

    The skill will:
    1. Encode the RGB image
    2. (TODO) Load and run the pi05 VLA model
    3. Decode the predicted action chunk (joint positions or end-effector poses)
    4. Execute the trajectory via the external arm adapter
    """

    name = "vla_policy_executor"

    def __init__(self, external_arm=None):
        self._arm = external_arm
        self._model = None
        self._model_loaded = False

    def _load_model(self) -> bool:
        """Load the fine-tuned pi05 VLA model. TODO: implement actual loading."""
        if self._model_loaded:
            return True
        logger.warning(
            "[VLAPolicyExecutor] VLA model loading is not implemented yet. "
            "Please fill in _load_model() with your pi05 checkpoint path and inference code."
        )
        # Example:
        # from pi05_policy import Pi05Policy
        # self._model = Pi05Policy(checkpoint_path="/path/to/checkpoint.pt")
        # self._model_loaded = True
        return False

    def _infer(self, instruction: str, rgb_frame: np.ndarray) -> Optional[np.ndarray]:
        """
        Run VLA inference.

        Args:
            instruction: language instruction (e.g. "夹布料")
            rgb_frame:   RGB image array (H, W, 3)

        Returns:
            Action chunk of shape (T, A) where T is horizon and A is action dim.
            Actions can be joint positions (6-DOF or 7-DOF) or end-effector deltas.
        """
        if not self._load_model():
            return None

        logger.info("[VLAPolicyExecutor] Running inference for: %s", instruction)
        # TODO: implement actual inference
        # Example:
        # action_chunk = self._model.predict(instruction, rgb_frame)
        # return action_chunk
        return None

    def execute(
        self,
        instruction: str,
        rgb_frame: np.ndarray,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Execute the VLA policy.

        Args:
            instruction: natural language task description
            rgb_frame:   current RGB image
            **kwargs:    optional extras (e.g. "world_model" snapshot)

        Returns:
            {"success": bool, "message": str, "actions": list}
        """
        if rgb_frame is None or not isinstance(rgb_frame, np.ndarray):
            return {
                "success": False,
                "message": "Missing or invalid rgb_frame for VLA inference.",
            }

        action_chunk = self._infer(instruction, rgb_frame)
        if action_chunk is None:
            return {
                "success": False,
                "message": "VLA model not loaded or inference returned no actions.",
            }

        # Execute trajectory if arm adapter is provided
        executed = 0
        if self._arm is not None:
            for t, action in enumerate(action_chunk):
                try:
                    # Assume action is joint positions; truncate/pad to match arm DOF
                    self._arm.joint_control(action.tolist() if hasattr(action, "tolist") else list(action))
                    executed += 1
                except Exception as exc:
                    logger.exception("[VLAPolicyExecutor] Failed at action step %d", t)
                    return {
                        "success": False,
                        "message": f"Trajectory execution failed at step {t}: {exc}",
                        "executed_steps": executed,
                        "actions": action_chunk.tolist() if hasattr(action_chunk, "tolist") else list(action_chunk),
                    }
        else:
            logger.info("[VLAPolicyExecutor] No arm adapter attached; returning actions without execution.")

        return {
            "success": True,
            "message": f"VLA policy executed {executed} action steps." if self._arm else "VLA inference completed (no arm attached).",
            "executed_steps": executed,
            "actions": action_chunk.tolist() if hasattr(action_chunk, "tolist") else list(action_chunk),
        }
