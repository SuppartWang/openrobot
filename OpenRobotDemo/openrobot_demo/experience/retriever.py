"""ExperienceRetriever: fuzzy matching for relevant experiences."""

import logging
from typing import List, Optional

from openrobot_demo.experience.library import ExperienceLibrary
from openrobot_demo.experience.schema import Experience, GripperConfig

logger = logging.getLogger(__name__)


class ExperienceRetriever:
    """
    Retrieve the most relevant experience(s) for a given task context.

    Uses a multi-stage retrieval strategy:
    1. Exact structured query (task_intent, object_type, action_type)
    2. Fuzzy keyword fallback (broader LIKE matching)
    3. Default/generic fallback (same action_type, any object)
    """

    def __init__(self, library: ExperienceLibrary):
        self.library = library

    def retrieve(
        self,
        task_intent: str,
        target_object_type: str,
        action_type: str,
        gripper_config: GripperConfig = GripperConfig.PARALLEL_2_FINGER,
        arm_count: int = 1,
        top_k: int = 3,
    ) -> List[Experience]:
        """
        Retrieve top-k relevant experiences for the given context.

        Returns empty list if no experiences match.
        """
        # Stage 1: exact-ish match
        results = self.library.query(
            task_intent=task_intent,
            target_object_type=target_object_type,
            action_type=action_type,
            gripper_config=gripper_config.value,
            success_only=True,
            limit=top_k,
        )
        if results:
            logger.info(
                "[ExperienceRetriever] Stage-1 hit: %d experiences for '%s | %s'",
                len(results),
                task_intent,
                action_type,
            )
            for r in results:
                self.library.increment_use(r.experience_id)
            return results

        # Stage 2: broader keyword match (task intent only)
        results = self.library.query(
            task_intent=task_intent,
            action_type=action_type,
            success_only=True,
            limit=top_k,
        )
        if results:
            logger.info(
                "[ExperienceRetriever] Stage-2 hit: %d experiences for '%s' (broader)",
                len(results),
                task_intent,
            )
            for r in results:
                self.library.increment_use(r.experience_id)
            return results

        # Stage 3: action-only fallback
        results = self.library.query(
            action_type=action_type,
            success_only=True,
            limit=top_k,
        )
        if results:
            logger.info(
                "[ExperienceRetriever] Stage-3 hit: %d generic experiences for action='%s'",
                len(results),
                action_type,
            )
            for r in results:
                self.library.increment_use(r.experience_id)
            return results

        logger.warning(
            "[ExperienceRetriever] No experience found for '%s | %s | %s'",
            task_intent,
            target_object_type,
            action_type,
        )
        return []

    def retrieve_best(
        self,
        task_intent: str,
        target_object_type: str,
        action_type: str,
        gripper_config: GripperConfig = GripperConfig.PARALLEL_2_FINGER,
        arm_count: int = 1,
    ) -> Optional[Experience]:
        """Return the single best-matching experience, or None."""
        results = self.retrieve(task_intent, target_object_type, action_type, gripper_config, arm_count, top_k=1)
        return results[0] if results else None
