"""VLA Policy Executor: end-to-end vision-language-action skill.

This skill serves as a bridge to a fine-tuned VLA model (e.g. pi05, OpenVLA,
Octo, etc.) that outputs an action sequence given an RGB image and a language
instruction.

Architecture:
    1. Encode image + text instruction
    2. Load VLA model (lazy singleton)
    3. Run inference → action chunk (T × action_dim)
    4. Execute trajectory via the external arm adapter

To integrate your own model:
    - Subclass VLAPolicyExecutor and override _load_model() and _infer()
    - Or set the VLA_MODEL_CLASS environment variable to a callable path
"""

import logging
import os
from typing import Any, Dict, Optional, Callable

import numpy as np

from openrobot_demo.skills.base import SkillInterface, SkillSchema, ParamSchema, ResultSchema

logger = logging.getLogger(__name__)


class VLAPolicyExecutor(SkillInterface):
    """Execute a single robotic skill using a fine-tuned VLA policy."""

    name = "vla_policy_executor"

    @property
    def schema(self) -> SkillSchema:
        return SkillSchema(
            description="Execute an end-to-end vision-language-action (VLA) policy. Given an RGB image and a language instruction, the model predicts a trajectory of actions and executes them.",
            parameters=[
                ParamSchema(name="instruction", type="str", description="Natural language task description, e.g. 'grasp the red cube'.", required=True),
                ParamSchema(name="rgb_frame", type="ndarray", description="Current RGB image array (H, W, 3).", required=True),
                ParamSchema(name="execute", type="bool", description="Whether to actually execute the trajectory or just return it.", required=False, default=True),
            ],
            returns=[
                ResultSchema(name="success", type="bool", description="Whether VLA execution succeeded."),
                ResultSchema(name="message", type="str", description="Status message."),
                ResultSchema(name="executed_steps", type="int", description="Number of action steps actually executed."),
                ResultSchema(name="actions", type="list", description="Predicted action trajectory as a list of action vectors."),
            ],
            dependencies=["camera", "arm", "vla_model"],
            preconditions=["VLA model must be loaded"],
            postconditions=["arm has executed the predicted trajectory (or a subset)"],
        )

    def __init__(self, external_arm=None, model_path: Optional[str] = None):
        self._arm = external_arm
        self._model_path = model_path or os.getenv("VLA_MODEL_PATH", "")
        self._model = None
        self._model_loaded = False
        self._model_factory: Optional[Callable] = None

    # ------------------------------------------------------------------
    # Model loading hooks (override these for your model)
    # ------------------------------------------------------------------
    def set_model_factory(self, factory: Callable):
        """Set a custom model factory: factory() -> model object."""
        self._model_factory = factory

    def _load_model(self) -> bool:
        """Load the fine-tuned VLA model.

        Override this method or call set_model_factory() to integrate your model.
        Default implementation tries VLA_MODEL_CLASS env var, then falls back to stub.
        """
        if self._model_loaded:
            return True

        # Try factory first
        if self._model_factory is not None:
            try:
                self._model = self._model_factory()
                self._model_loaded = True
                logger.info("[VLAPolicyExecutor] Model loaded via custom factory.")
                return True
            except Exception as exc:
                logger.error("[VLAPolicyExecutor] Custom factory failed: %s", exc)

        # Try environment-based dynamic loading
        model_class_path = os.getenv("VLA_MODEL_CLASS", "")
        if model_class_path:
            try:
                self._model = _import_dynamic(model_class_path, model_path=self._model_path)
                self._model_loaded = True
                logger.info("[VLAPolicyExecutor] Model loaded from %s", model_class_path)
                return True
            except Exception as exc:
                logger.error("[VLAPolicyExecutor] Dynamic import failed: %s", exc)

        logger.warning(
            "[VLAPolicyExecutor] No VLA model configured. "
            "Set VLA_MODEL_CLASS env var or call set_model_factory(). "
            "Currently running in stub mode (returns mock actions)."
        )
        return False

    def _infer(self, instruction: str, rgb_frame: np.ndarray) -> Optional[np.ndarray]:
        """Run VLA inference.

        Args:
            instruction: language instruction
            rgb_frame: RGB image array (H, W, 3)

        Returns:
            Action chunk of shape (T, A) where T is horizon and A is action dim.
            Actions can be joint positions or end-effector deltas.
        """
        if not self._load_model():
            # Stub: return a simple mock trajectory
            logger.info("[VLAPolicyExecutor] Running stub inference for: %s", instruction)
            T, A = 10, 7
            return np.zeros((T, A), dtype=np.float32)

        # If custom model is loaded, try to call its predict method
        if hasattr(self._model, "predict"):
            try:
                return self._model.predict(instruction=instruction, image=rgb_frame)
            except Exception as exc:
                logger.error("[VLAPolicyExecutor] Model predict failed: %s", exc)
                return None

        if hasattr(self._model, "infer"):
            try:
                return self._model.infer(instruction=instruction, image=rgb_frame)
            except Exception as exc:
                logger.error("[VLAPolicyExecutor] Model infer failed: %s", exc)
                return None

        logger.error("[VLAPolicyExecutor] Loaded model has no predict() or infer() method.")
        return None

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------
    def execute(
        self,
        instruction: str,
        rgb_frame: np.ndarray,
        execute: bool = True,
        **kwargs,
    ) -> Dict[str, Any]:
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

        actions_list = action_chunk.tolist() if hasattr(action_chunk, "tolist") else list(action_chunk)

        if not execute:
            return {
                "success": True,
                "message": f"VLA inference completed. {len(actions_list)} action steps predicted (not executed).",
                "executed_steps": 0,
                "actions": actions_list,
            }

        # Execute trajectory if arm adapter is provided
        executed = 0
        if self._arm is not None:
            for t, action in enumerate(action_chunk):
                try:
                    action_list = action.tolist() if hasattr(action, "tolist") else list(action)
                    # Try generic Action interface first
                    if hasattr(self._arm, "command"):
                        from openrobot_demo.hardware.robot_interface import Action
                        self._arm.command(
                            Action(action_type="joint_position", values=np.array(action_list))
                        )
                    else:
                        # Fallback to legacy joint_control
                        self._arm.joint_control(action_list)
                    executed += 1
                except Exception as exc:
                    logger.exception("[VLAPolicyExecutor] Failed at action step %d", t)
                    return {
                        "success": False,
                        "message": f"Trajectory execution failed at step {t}: {exc}",
                        "executed_steps": executed,
                        "actions": actions_list,
                    }
        else:
            logger.info("[VLAPolicyExecutor] No arm adapter attached; returning actions without execution.")

        return {
            "success": True,
            "message": f"VLA policy executed {executed} action steps." if self._arm else "VLA inference completed (no arm attached).",
            "executed_steps": executed,
            "actions": actions_list,
        }


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------
def _import_dynamic(dotted_path: str, **kwargs):
    """Dynamically import a class/function from a dotted path."""
    module_path, _, class_name = dotted_path.rpartition(".")
    mod = __import__(module_path, fromlist=[class_name])
    cls = getattr(mod, class_name)
    return cls(**kwargs)
