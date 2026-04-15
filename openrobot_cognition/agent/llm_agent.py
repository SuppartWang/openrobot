"""Layer 4: LLM Agent for high-level task planning and reasoning."""

import os
import json
from typing import List, Dict, Any, Optional
from openai import OpenAI


class LLMAgent:
    """
    A lightweight LLM-based agent for embodied robotics.
    Uses a ReAct-style prompt to translate natural language into subtask plans.
    """

    def __init__(self, model: str = "gpt-4o-mini", api_key: Optional[str] = None, base_url: Optional[str] = None):
        self.model = model
        self.client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"), base_url=base_url)
        self.system_prompt = (
            "You are the high-level cognitive module of an embodied robot. "
            "Your job is to decompose a user instruction into a short sequence of executable subtasks.\n\n"
            "Available actions:\n"
            "- move_to(x, y, z): move gripper to target position\n"
            "- grasp(object_id): close gripper to grasp object\n"
            "- release(): open gripper\n"
            "- push(object_id, dx, dy, dz): push object in direction\n"
            "- observe(): explicitly request visual observation (optional)\n\n"
            "Respond ONLY with a JSON list of steps, e.g.:\n"
            '[{"action": "move_to", "args": [0.5, 0.1, 0.5]}, {"action": "grasp", "args": ["cube"]}]'
        )

    def plan(self, instruction: str, scene_context: Optional[str] = None) -> List[Dict[str, Any]]:
        user_content = f"Instruction: {instruction}"
        if scene_context:
            user_content += f"\nScene context: {scene_context}"

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_content},
            ],
            temperature=0.2,
            max_tokens=512,
        )

        raw = response.choices[0].message.content
        # Strip markdown code fences if present
        raw = raw.strip()
        if raw.startswith("```json"):
            raw = raw[7:]
        if raw.startswith("```"):
            raw = raw[3:]
        if raw.endswith("```"):
            raw = raw[:-3]
        raw = raw.strip()

        try:
            plan = json.loads(raw)
            if not isinstance(plan, list):
                plan = [plan]
        except json.JSONDecodeError:
            # Fallback: wrap raw text as a single reasoning step
            plan = [{"action": "reasoning", "content": raw}]

        return plan
