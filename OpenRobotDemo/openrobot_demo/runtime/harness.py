"""Harness: workflow orchestration engine for robot skill execution.

Supports:
- Sequence: execute nodes in order
- Parallel: execute nodes concurrently
- Condition: if/then/else branching
- Loop: while/for iteration
- Pipeline: data-flow composition (output of A → input of B)
- Try/Catch: error handling and recovery

All execution flows are expressed as a DAG of Nodes, executed by a HarnessRunner.
"""

from __future__ import annotations

import logging
import time
import threading
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Union

from openrobot_demo.agent.skill_router import SkillRouter

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Execution Context
# ------------------------------------------------------------------
class ExecutionContext:
    """Shared state container passed through the execution graph."""

    def __init__(self, initial_vars: Optional[Dict[str, Any]] = None):
        self.vars: Dict[str, Any] = initial_vars or {}
        self.history: List[Dict[str, Any]] = []
        self.errors: List[str] = []
        self._lock = threading.RLock()

    def get(self, key: str, default: Any = None) -> Any:
        with self._lock:
            return self.vars.get(key, default)

    def set(self, key: str, value: Any):
        with self._lock:
            self.vars[key] = value

    def update(self, values: Dict[str, Any]):
        with self._lock:
            self.vars.update(values)

    def record(self, node_name: str, result: Dict[str, Any]):
        with self._lock:
            self.history.append({"node": node_name, "result": result, "timestamp": time.time()})

    def add_error(self, msg: str):
        with self._lock:
            self.errors.append(msg)

    def resolve(self, value: Any) -> Any:
        """Recursively resolve variable references like ${var_name}."""
        if isinstance(value, str) and value.startswith("${") and value.endswith("}"):
            var_name = value[2:-1]
            return self.get(var_name, value)
        if isinstance(value, dict):
            return {k: self.resolve(v) for k, v in value.items()}
        if isinstance(value, list):
            return [self.resolve(v) for v in value]
        return value


# ------------------------------------------------------------------
# Node Results
# ------------------------------------------------------------------
class NodeStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILURE = "failure"
    SKIPPED = "skipped"


@dataclass
class NodeResult:
    status: NodeStatus
    output: Any = None
    error: Optional[str] = None
    execution_time_ms: float = 0.0


# ------------------------------------------------------------------
# Base Node
# ------------------------------------------------------------------
class Node(ABC):
    """Abstract node in the execution graph."""

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def run(self, ctx: ExecutionContext, router: SkillRouter) -> NodeResult:
        ...

    def _resolve_args(self, ctx: ExecutionContext, args: Dict[str, Any]) -> Dict[str, Any]:
        return {k: ctx.resolve(v) for k, v in args.items()}


# ------------------------------------------------------------------
# Skill Node
# ------------------------------------------------------------------
class SkillNode(Node):
    """Execute a single skill via SkillRouter."""

    def __init__(self, name: str, skill_name: str, args: Dict[str, Any],
                 output_key: Optional[str] = None):
        super().__init__(name)
        self.skill_name = skill_name
        self.args = args
        self.output_key = output_key  # If set, store result in ctx under this key

    def run(self, ctx: ExecutionContext, router: SkillRouter) -> NodeResult:
        resolved = self._resolve_args(ctx, self.args)
        logger.info("[Harness] SkillNode '%s' → skill '%s'", self.name, self.skill_name)
        start = time.time()

        try:
            result = router.execute_single(self.skill_name, resolved)
            elapsed = (time.time() - start) * 1000
            ctx.record(self.name, result)

            if self.output_key and isinstance(result, dict):
                ctx.set(self.output_key, result)

            status = NodeStatus.SUCCESS if result.get("success") else NodeStatus.FAILURE
            return NodeResult(status=status, output=result, execution_time_ms=elapsed)

        except Exception as exc:
            elapsed = (time.time() - start) * 1000
            ctx.add_error(str(exc))
            logger.exception("[Harness] SkillNode '%s' failed", self.name)
            return NodeResult(status=NodeStatus.FAILURE, error=str(exc), execution_time_ms=elapsed)


# ------------------------------------------------------------------
# Sequence Node
# ------------------------------------------------------------------
class SequenceNode(Node):
    """Execute children in order, stop on first failure."""

    def __init__(self, name: str, children: List[Node], stop_on_error: bool = True):
        super().__init__(name)
        self.children = children
        self.stop_on_error = stop_on_error

    def run(self, ctx: ExecutionContext, router: SkillRouter) -> NodeResult:
        logger.info("[Harness] SequenceNode '%s' with %d children", self.name, len(self.children))
        outputs = []
        start = time.time()

        for child in self.children:
            result = child.run(ctx, router)
            outputs.append(result)
            if result.status == NodeStatus.FAILURE and self.stop_on_error:
                elapsed = (time.time() - start) * 1000
                return NodeResult(
                    status=NodeStatus.FAILURE,
                    output=outputs,
                    error=f"Sequence stopped at '{child.name}': {result.error}",
                    execution_time_ms=elapsed,
                )

        elapsed = (time.time() - start) * 1000
        return NodeResult(status=NodeStatus.SUCCESS, output=outputs, execution_time_ms=elapsed)


# ------------------------------------------------------------------
# Parallel Node
# ------------------------------------------------------------------
class ParallelNode(Node):
    """Execute children concurrently using a thread pool."""

    def __init__(self, name: str, children: List[Node], max_workers: int = 4,
                 require_all: bool = True):
        super().__init__(name)
        self.children = children
        self.max_workers = max_workers
        self.require_all = require_all  # If True, all must succeed

    def run(self, ctx: ExecutionContext, router: SkillRouter) -> NodeResult:
        logger.info("[Harness] ParallelNode '%s' with %d children", self.name, len(self.children))
        start = time.time()
        results = {}

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_child = {
                executor.submit(child.run, ctx, router): child
                for child in self.children
            }
            for future in as_completed(future_to_child):
                child = future_to_child[future]
                try:
                    result = future.result()
                except Exception as exc:
                    result = NodeResult(status=NodeStatus.FAILURE, error=str(exc))
                results[child.name] = result

        elapsed = (time.time() - start) * 1000
        any_failed = any(r.status == NodeStatus.FAILURE for r in results.values())

        if any_failed and self.require_all:
            errors = [f"{name}: {r.error}" for name, r in results.items() if r.status == NodeStatus.FAILURE]
            return NodeResult(
                status=NodeStatus.FAILURE,
                output=results,
                error=f"Parallel execution failed: {'; '.join(errors)}",
                execution_time_ms=elapsed,
            )

        return NodeResult(status=NodeStatus.SUCCESS, output=results, execution_time_ms=elapsed)


# ------------------------------------------------------------------
# Condition Node
# ------------------------------------------------------------------
class ConditionNode(Node):
    """If/then/else branching based on a predicate."""

    def __init__(self, name: str,
                 predicate: Callable[[ExecutionContext], bool],
                 then_node: Node,
                 else_node: Optional[Node] = None):
        super().__init__(name)
        self.predicate = predicate
        self.then_node = then_node
        self.else_node = else_node

    def run(self, ctx: ExecutionContext, router: SkillRouter) -> NodeResult:
        condition = self.predicate(ctx)
        logger.info("[Harness] ConditionNode '%s' → %s", self.name, "then" if condition else "else")

        if condition:
            return self.then_node.run(ctx, router)
        elif self.else_node:
            return self.else_node.run(ctx, router)
        else:
            return NodeResult(status=NodeStatus.SKIPPED, output=None)


# ------------------------------------------------------------------
# Loop Node
# ------------------------------------------------------------------
class LoopNode(Node):
    """While-loop: repeat child while predicate is true."""

    def __init__(self, name: str,
                 predicate: Callable[[ExecutionContext], bool],
                 body: Node,
                 max_iterations: int = 100):
        super().__init__(name)
        self.predicate = predicate
        self.body = body
        self.max_iterations = max_iterations

    def run(self, ctx: ExecutionContext, router: SkillRouter) -> NodeResult:
        logger.info("[Harness] LoopNode '%s' starting", self.name)
        iterations = 0
        outputs = []
        start = time.time()

        while self.predicate(ctx) and iterations < self.max_iterations:
            result = self.body.run(ctx, router)
            outputs.append(result)
            iterations += 1

            if result.status == NodeStatus.FAILURE:
                elapsed = (time.time() - start) * 1000
                return NodeResult(
                    status=NodeStatus.FAILURE,
                    output=outputs,
                    error=f"Loop body failed at iteration {iterations}: {result.error}",
                    execution_time_ms=elapsed,
                )

        elapsed = (time.time() - start) * 1000
        logger.info("[Harness] LoopNode '%s' completed after %d iterations", self.name, iterations)
        return NodeResult(status=NodeStatus.SUCCESS, output=outputs, execution_time_ms=elapsed)


# ------------------------------------------------------------------
# Pipeline Node
# ------------------------------------------------------------------
class PipelineNode(Node):
    """Data-flow pipeline: output of node N becomes input of node N+1.

    Usage: define a mapping of which output field feeds which input parameter.
    """

    def __init__(self, name: str, stages: List[Node],
                 mappings: Optional[List[Dict[str, str]]] = None):
        """
        Args:
            stages: List of nodes to execute in sequence.
            mappings: List of dicts, one per stage (except first).
                      Each dict maps {input_param: output_field_from_prev}.
                      Use ${output.<field>} to reference previous output.
        """
        super().__init__(name)
        self.stages = stages
        self.mappings = mappings or []

    def run(self, ctx: ExecutionContext, router: SkillRouter) -> NodeResult:
        logger.info("[Harness] PipelineNode '%s' with %d stages", self.name, len(self.stages))
        prev_output = None
        outputs = []
        start = time.time()

        for i, stage in enumerate(self.stages):
            # If this is not the first stage and we have a mapping, inject prev output
            if i > 0 and prev_output is not None and i - 1 < len(self.mappings):
                mapping = self.mappings[i - 1]
                if isinstance(stage, SkillNode):
                    for input_key, output_expr in mapping.items():
                        if output_expr.startswith("${output.") and output_expr.endswith("}"):
                            field = output_expr[9:-1]  # extract field name
                            if isinstance(prev_output, dict) and field in prev_output:
                                stage.args[input_key] = prev_output[field]

            result = stage.run(ctx, router)
            outputs.append(result)
            prev_output = result.output

            if result.status == NodeStatus.FAILURE:
                elapsed = (time.time() - start) * 1000
                return NodeResult(
                    status=NodeStatus.FAILURE,
                    output=outputs,
                    error=f"Pipeline failed at stage '{stage.name}': {result.error}",
                    execution_time_ms=elapsed,
                )

        elapsed = (time.time() - start) * 1000
        return NodeResult(status=NodeStatus.SUCCESS, output=outputs, execution_time_ms=elapsed)


# ------------------------------------------------------------------
# Try/Catch Node
# ------------------------------------------------------------------
class TryCatchNode(Node):
    """Try a node, catch errors and run recovery."""

    def __init__(self, name: str, try_node: Node,
                 catch_node: Optional[Node] = None,
                 finally_node: Optional[Node] = None):
        super().__init__(name)
        self.try_node = try_node
        self.catch_node = catch_node
        self.finally_node = finally_node

    def run(self, ctx: ExecutionContext, router: SkillRouter) -> NodeResult:
        logger.info("[Harness] TryCatchNode '%s'", self.name)
        start = time.time()

        try:
            result = self.try_node.run(ctx, router)
            if result.status == NodeStatus.FAILURE:
                raise RuntimeError(result.error or "Try node failed")
        except Exception as exc:
            ctx.add_error(str(exc))
            logger.warning("[Harness] TryCatchNode '%s' caught: %s", self.name, exc)
            if self.catch_node:
                result = self.catch_node.run(ctx, router)
            else:
                elapsed = (time.time() - start) * 1000
                return NodeResult(status=NodeStatus.FAILURE, error=str(exc), execution_time_ms=elapsed)
        finally:
            if self.finally_node:
                self.finally_node.run(ctx, router)

        elapsed = (time.time() - start) * 1000
        return NodeResult(status=result.status, output=result.output, execution_time_ms=elapsed)


# ------------------------------------------------------------------
# Harness Runner
# ------------------------------------------------------------------
class HarnessRunner:
    """Execute a Node graph against a SkillRouter."""

    def __init__(self, router: SkillRouter):
        self.router = router

    def run(self, root: Node, initial_vars: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Execute the graph and return a summary."""
        ctx = ExecutionContext(initial_vars)
        logger.info("[HarnessRunner] Starting execution of '%s'", root.name)
        start = time.time()

        result = root.run(ctx, self.router)

        elapsed = (time.time() - start) * 1000
        summary = {
            "success": result.status == NodeStatus.SUCCESS,
            "root": root.name,
            "status": result.status.value,
            "execution_time_ms": elapsed,
            "final_vars": dict(ctx.vars),
            "history": ctx.history,
            "errors": ctx.errors,
            "output": result.output,
        }

        if result.error:
            summary["error"] = result.error

        logger.info("[HarnessRunner] Execution finished: success=%s, time=%.1fms, errors=%d",
                    summary["success"], elapsed, len(ctx.errors))
        return summary

    def run_from_dict(self, spec: Dict[str, Any], initial_vars: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Build a Node graph from a dict spec and execute it.

        Spec format:
        {
            "type": "sequence",
            "name": "my_plan",
            "children": [
                {"type": "skill", "name": "cap", "skill": "camera_capture", "args": {"return_depth": true}},
                {"type": "skill", "name": "detect", "skill": "vision_3d_estimator", "args": {"rgb_frame": "${rgb_frame}", "target_name": "cube"}, "output_key": "detection"},
                {"type": "parallel", "name": "sense", "children": [...]},
                {"type": "condition", "name": "check", "predicate": "${detection.success}", "then": {...}, "else": {...}},
            ]
        }
        """
        root = self._build_node(spec)
        return self.run(root, initial_vars)

    def _build_node(self, spec: Dict[str, Any]) -> Node:
        node_type = spec.get("type", "skill")
        name = spec.get("name", "unnamed")

        if node_type == "skill":
            return SkillNode(
                name=name,
                skill_name=spec["skill"],
                args=spec.get("args", {}),
                output_key=spec.get("output_key"),
            )

        if node_type == "sequence":
            children = [self._build_node(c) for c in spec.get("children", [])]
            return SequenceNode(name, children, stop_on_error=spec.get("stop_on_error", True))

        if node_type == "parallel":
            children = [self._build_node(c) for c in spec.get("children", [])]
            return ParallelNode(name, children,
                                max_workers=spec.get("max_workers", 4),
                                require_all=spec.get("require_all", True))

        if node_type == "condition":
            # For now, predicate is a simple variable check expressed as string
            pred_expr = spec.get("predicate", "")
            then_node = self._build_node(spec["then"])
            else_node = self._build_node(spec["else"]) if "else" in spec else None

            def predicate(ctx: ExecutionContext) -> bool:
                return bool(ctx.resolve(pred_expr))

            return ConditionNode(name, predicate, then_node, else_node)

        if node_type == "loop":
            pred_expr = spec.get("predicate", "")
            body = self._build_node(spec["body"])
            max_iter = spec.get("max_iterations", 100)

            def predicate(ctx: ExecutionContext) -> bool:
                return bool(ctx.resolve(pred_expr))

            return LoopNode(name, predicate, body, max_iter)

        if node_type == "pipeline":
            stages = [self._build_node(s) for s in spec.get("stages", [])]
            mappings = spec.get("mappings")
            return PipelineNode(name, stages, mappings)

        if node_type == "trycatch":
            try_node = self._build_node(spec["try"])
            catch_node = self._build_node(spec["catch"]) if "catch" in spec else None
            finally_node = self._build_node(spec["finally"]) if "finally" in spec else None
            return TryCatchNode(name, try_node, catch_node, finally_node)

        raise ValueError(f"Unknown node type: {node_type}")
