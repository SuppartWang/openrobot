"""RobotDashboard: 三栏式实机可视化界面

Layout (3 columns, each ~1/3 width):
  Left  : Perception (RGB, Depth, arm joint positions)
  Center: Planning reasoning (top) + instruction input (bottom)
  Right : Motion execution status (current action, history, goal tree)

Usage:
    from openrobot_demo.ui.dashboard import RobotDashboard
    dash = RobotDashboard(agent, sensors, dual_arm)
    dash.run()
"""

from __future__ import annotations

import logging
import queue
import threading
import time
import tkinter as tk
from tkinter import ttk, scrolledtext
from typing import Any, Callable, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

try:
    from PIL import Image, ImageTk
    _PIL_AVAILABLE = True
except ImportError:
    _PIL_AVAILABLE = False
    logger.warning("PIL not available; image panels will show text placeholders.")


class RobotDashboard:
    """Tkinter-based three-column dashboard for OpenRobotDemo."""

    def __init__(
        self,
        agent=None,
        sensors: Optional[List] = None,
        dual_arm=None,
        width: int = 1400,
        height: int = 900,
    ):
        self.agent = agent
        self.sensors = sensors or []
        self.dual_arm = dual_arm

        # Thread-safe queue for observer events -> main thread
        self._event_queue: queue.Queue = queue.Queue()
        self._running = False
        self._agent_thread: Optional[threading.Thread] = None

        # Cached data for display
        self._latest_rgb: Optional[np.ndarray] = None
        self._latest_depth: Optional[np.ndarray] = None
        self._latest_joints: Dict[str, List[float]] = {}
        self._current_instruction: str = ""
        self._planning_log: List[str] = []
        self._execution_log: List[str] = []
        self._goal_tree_text: str = ""
        self._current_action: str = "空闲"
        self._step_count: int = 0

        # Tk root
        self.root = tk.Tk()
        self.root.title("OpenRobotDemo — 实机可视化控制台")
        self.root.geometry(f"{width}x{height}")
        self.root.configure(bg="#1e1e1e")
        self._build_ui()

        # Register as observer if agent provided
        if self.agent is not None:
            self.agent.add_observer(self._on_agent_event)

    # ------------------------------------------------------------------
    # Font helper
    # ------------------------------------------------------------------
    @staticmethod
    def _detect_cjk_font() -> str:
        """Return a CJK-capable font name available on the system."""
        import tkinter.font as tkfont
        families = set(tkfont.families())
        candidates = [
            "Noto Sans CJK SC",
            "Noto Sans Mono CJK SC",
            "WenQuanYi Micro Hei",
            "WenQuanYi Zen Hei",
            "AR PL UKai CN",
            "AR PL UMing CN",
            "Droid Sans Fallback",
            "Microsoft YaHei",
            "SimHei",
            "SimSun",
            "DejaVu Sans",
        ]
        for name in candidates:
            if name in families:
                return name
        for f in families:
            if "Noto" in f and "CJK" in f:
                return f
        return "DejaVu Sans"

    # ------------------------------------------------------------------
    # UI Construction
    # ------------------------------------------------------------------
    def _build_ui(self):
        # Detect available CJK font
        _cjk_font = self._detect_cjk_font()
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TFrame", background="#1e1e1e")
        style.configure("TLabel", background="#1e1e1e", foreground="#e0e0e0", font=(_cjk_font, 10))
        style.configure("Header.TLabel", font=(_cjk_font, 12, "bold"), foreground="#4fc3f7")
        style.configure("TButton", font=(_cjk_font, 10))
        style.configure("TEntry", font=(_cjk_font, 10))

        # Main 3-column grid
        self.root.grid_columnconfigure(0, weight=1, uniform="col")
        self.root.grid_columnconfigure(1, weight=1, uniform="col")
        self.root.grid_columnconfigure(2, weight=1, uniform="col")
        self.root.grid_rowconfigure(0, weight=1)

        # ---- Left Column: Perception ----
        left = ttk.Frame(self.root, padding=8)
        left.grid(row=0, column=0, sticky="nsew")
        left.grid_rowconfigure(1, weight=1)
        left.grid_rowconfigure(3, weight=1)
        left.grid_columnconfigure(0, weight=1)

        ttk.Label(left, text="[摄] RGB 图像", style="Header.TLabel").grid(row=0, column=0, sticky="w", pady=(0, 4))
        self.rgb_canvas = tk.Canvas(left, bg="#2a2a2a", highlightthickness=1, highlightbackground="#444")
        self.rgb_canvas.grid(row=1, column=0, sticky="nsew", pady=(0, 8))

        ttk.Label(left, text="[深] 深度图", style="Header.TLabel").grid(row=2, column=0, sticky="w", pady=(0, 4))
        self.depth_canvas = tk.Canvas(left, bg="#2a2a2a", highlightthickness=1, highlightbackground="#444")
        self.depth_canvas.grid(row=3, column=0, sticky="nsew", pady=(0, 8))

        ttk.Label(left, text="[臂] 机械臂状态", style="Header.TLabel").grid(row=4, column=0, sticky="w", pady=(0, 4))
        self.arm_text = scrolledtext.ScrolledText(
            left, wrap=tk.WORD, height=6, bg="#252525", fg="#e0e0e0",
            font=("Consolas", 9), insertbackground="white"
        )
        self.arm_text.grid(row=5, column=0, sticky="nsew")
        self.arm_text.config(state=tk.DISABLED)

        # ---- Center Column: Planning + Input ----
        center = ttk.Frame(self.root, padding=8)
        center.grid(row=0, column=1, sticky="nsew")
        center.grid_rowconfigure(0, weight=3)
        center.grid_rowconfigure(1, weight=0)
        center.grid_columnconfigure(0, weight=1)

        # Top: planning reasoning
        ttk.Label(center, text="[思] 规划思考过程", style="Header.TLabel").grid(row=0, column=0, sticky="nw", pady=(0, 4))
        self.planning_text = scrolledtext.ScrolledText(
            center, wrap=tk.WORD, bg="#252525", fg="#e0e0e0",
            font=("Microsoft YaHei", 10), insertbackground="white"
        )
        self.planning_text.grid(row=0, column=0, sticky="nsew", pady=(24, 8))
        self.planning_text.config(state=tk.DISABLED)

        # Bottom: instruction input
        input_frame = ttk.Frame(center)
        input_frame.grid(row=1, column=0, sticky="sew", pady=(8, 0))
        input_frame.grid_columnconfigure(0, weight=1)

        ttk.Label(input_frame, text="[令] 指令输入", style="Header.TLabel").grid(row=0, column=0, sticky="w", pady=(0, 4))
        self.instruction_entry = ttk.Entry(input_frame, font=("Microsoft YaHei", 11))
        self.instruction_entry.grid(row=1, column=0, sticky="ew", pady=(0, 6))
        self.instruction_entry.insert(0, "将筒状布料提起，套在铝合金支撑板上，等待检测后再取下来")
        self.instruction_entry.bind("<Return>", lambda e: self._on_run())

        btn_frame = ttk.Frame(input_frame)
        btn_frame.grid(row=2, column=0, sticky="ew")
        self.run_btn = ttk.Button(btn_frame, text="[>] 执行任务", command=self._on_run)
        self.run_btn.pack(side=tk.LEFT, padx=(0, 8))
        self.stop_btn = ttk.Button(btn_frame, text="[X] 停止", command=self._on_stop)
        self.stop_btn.pack(side=tk.LEFT, padx=(0, 8))
        self.clear_btn = ttk.Button(btn_frame, text="[C] 清空日志", command=self._on_clear)
        self.clear_btn.pack(side=tk.LEFT)

        # ---- Right Column: Execution Status ----
        right = ttk.Frame(self.root, padding=8)
        right.grid(row=0, column=2, sticky="nsew")
        right.grid_rowconfigure(1, weight=1)
        right.grid_rowconfigure(3, weight=2)
        right.grid_columnconfigure(0, weight=1)

        # Current action
        ttk.Label(right, text="[动] 当前动作", style="Header.TLabel").grid(row=0, column=0, sticky="w", pady=(0, 4))
        self.action_label = ttk.Label(right, text="空闲", font=("Microsoft YaHei", 11, "bold"), foreground="#81c784")
        self.action_label.grid(row=0, column=0, sticky="w", pady=(24, 4))
        self.step_label = ttk.Label(right, text="步数: 0", font=("Microsoft YaHei", 9))
        self.step_label.grid(row=0, column=0, sticky="e", pady=(24, 4))

        # Execution history
        ttk.Label(right, text="[历] 执行历史", style="Header.TLabel").grid(row=1, column=0, sticky="nw", pady=(0, 4))
        self.exec_text = scrolledtext.ScrolledText(
            right, wrap=tk.WORD, height=8, bg="#252525", fg="#e0e0e0",
            font=("Microsoft YaHei", 9), insertbackground="white"
        )
        self.exec_text.grid(row=2, column=0, sticky="nsew", pady=(0, 8))
        self.exec_text.config(state=tk.DISABLED)

        # Goal tree
        ttk.Label(right, text="[目] 目标树", style="Header.TLabel").grid(row=3, column=0, sticky="nw", pady=(0, 4))
        self.goal_text = scrolledtext.ScrolledText(
            right, wrap=tk.WORD, bg="#252525", fg="#e0e0e0",
            font=("Microsoft YaHei", 9), insertbackground="white"
        )
        self.goal_text.grid(row=4, column=0, sticky="nsew")
        self.goal_text.config(state=tk.DISABLED)

    # ------------------------------------------------------------------
    # Event handling (called from BDIAgent observer, may be any thread)
    # ------------------------------------------------------------------
    def _on_agent_event(self, event_type: str, data: Dict[str, Any]):
        self._event_queue.put((event_type, data))

    def _process_events(self):
        """Drain event queue and update UI (called in main thread via after())."""
        try:
            while True:
                event_type, data = self._event_queue.get_nowait()
                self._handle_event(event_type, data)
        except queue.Empty:
            pass
        # Schedule next check
        if self._running:
            self.root.after(100, self._process_events)

    def _handle_event(self, event_type: str, data: Dict[str, Any]):
        if event_type == "perception":
            self._update_perception(data)
        elif event_type == "intent":
            self._update_intent(data)
        elif event_type == "step_result":
            self._update_step_result(data)
        elif event_type == "task_start":
            self._append_planning(f"[任务开始] {data.get('instruction', '')}\n")
            self._current_instruction = data.get("instruction", "")
            self.step_count = 0
        elif event_type == "task_end":
            success = data.get("success", False)
            status = "[成功]" if success else "[失败]"
            self._append_planning(f"[任务结束] {status} | 步数:{data.get('total_steps')} | 耗时:{data.get('elapsed_time_s', 0):.1f}s\n")
            self._current_action = "空闲"
            self._update_action_label()

    def _update_perception(self, data: Dict[str, Any]):
        for src, reading in data.items():
            if not isinstance(reading, dict):
                continue
            modality = reading.get("modality", "")
            payload = reading.get("payload")
            if modality == "rgb" and isinstance(payload, np.ndarray):
                self._latest_rgb = payload
                self._draw_image(self.rgb_canvas, payload, "RGB")
            elif modality == "depth" and isinstance(payload, np.ndarray):
                self._latest_depth = payload
                depth_vis = self._depth_to_color(payload)
                self._draw_image(self.depth_canvas, depth_vis, "Depth")
            elif modality == "proprioception":
                self._latest_joints[src] = payload
                self._update_arm_text()

    def _update_intent(self, data: Dict[str, Any]):
        intent = data.get("current_intent")
        if intent:
            steps = intent.get("plan_steps", [])
            idx = intent.get("current_step", 0)
            self._append_planning(f"[新意图] 计划 {len(steps)} 步 | 当前第 {idx + 1} 步\n")
            for i, step in enumerate(steps):
                marker = "▶" if i == idx else " "
                skill = step.get("skill", "?")
                self._append_planning(f"  {marker} {i + 1}. {skill}\n")
        goal_tree = data.get("goal_tree")
        if goal_tree:
            self._goal_tree_text = self._format_goal_tree(goal_tree)
            self._set_text(self.goal_text, self._goal_tree_text)

    def _update_step_result(self, data: Dict[str, Any]):
        self._step_count += 1
        self.step_label.config(text=f"步数: {self._step_count}")
        msg = data.get("message", "")
        success = data.get("success", False)
        status_icon = "[OK]" if success else "[FAIL]"
        self._append_exec(f"{status_icon} Step {self._step_count}: {msg}\n")
        if not success:
            self._append_planning(f"[执行失败] {msg}\n")
        else:
            if "Intent completed" in msg:
                self._current_action = "意图完成"
            else:
                self._current_action = msg[:40]
        self._update_action_label()

    # ------------------------------------------------------------------
    # UI helpers
    # ------------------------------------------------------------------
    def _draw_image(self, canvas: tk.Canvas, arr: np.ndarray, label: str):
        if not _PIL_AVAILABLE:
            return
        # Resize to fit canvas while keeping aspect ratio
        cw = max(canvas.winfo_width(), 200)
        ch = max(canvas.winfo_height(), 150)
        if cw <= 1 or ch <= 1:
            cw, ch = 400, 300

        h, w = arr.shape[:2]
        scale = min(cw / w, ch / h)
        nw, nh = int(w * scale), int(h * scale)

        try:
            img = Image.fromarray(arr.astype(np.uint8))
            img = img.resize((nw, nh), Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(img)
        except Exception as exc:
            logger.debug("Image draw failed: %s", exc)
            return

        # Keep reference to avoid GC
        canvas.photo = photo
        canvas.delete("all")
        canvas.create_image(cw // 2, ch // 2, image=photo, anchor=tk.CENTER)
        canvas.create_text(8, 8, text=label, anchor=tk.NW, fill="#4fc3f7", font=("Consolas", 9))

    @staticmethod
    def _depth_to_color(depth: np.ndarray) -> np.ndarray:
        """Convert depth (meters) to pseudo-color RGB for display."""
        d = np.nan_to_num(depth, nan=0.0)
        d = np.clip(d, 0.0, 2.0)  # clip to 2m
        d_uint8 = (d / 2.0 * 255).astype(np.uint8)
        # Apply matplotlib-style colormap (viridis-ish) manually
        cmap = np.zeros((256, 3), dtype=np.uint8)
        for i in range(256):
            t = i / 255.0
            # Simple viridis approximation
            r = np.clip(0.28 + 0.89 * t - 0.15 * t * t, 0, 1)
            g = np.clip(0.00 + 1.50 * t - 0.50 * t * t, 0, 1)
            b = np.clip(0.33 + 0.70 * t - 0.20 * t * t, 0, 1)
            cmap[i] = [int(r * 255), int(g * 255), int(b * 255)]
        return cmap[d_uint8]

    def _update_arm_text(self):
        lines = []
        if self.dual_arm is not None:
            try:
                from openrobot_demo.dual_arm.controller import ArmSide
                lp = self.dual_arm.get_pos(ArmSide.LEFT)
                rp = self.dual_arm.get_pos(ArmSide.RIGHT)
                lines.append(f"左臂: {[round(x, 3) for x in lp[:6]]}")
                lines.append(f"右臂: {[round(x, 3) for x in rp[:6]]}")
            except Exception as exc:
                lines.append(f"臂状态读取失败: {exc}")
        for src, payload in self._latest_joints.items():
            if isinstance(payload, dict) and "joint_positions" in payload:
                jp = payload["joint_positions"]
                lines.append(f"{src}: {[round(x, 3) for x in jp[:6]]}")
        self._set_text(self.arm_text, "\n".join(lines) if lines else "暂无数据")

    def _update_action_label(self):
        self.action_label.config(text=self._current_action or "空闲")

    def _append_planning(self, text: str):
        self.planning_text.config(state=tk.NORMAL)
        self.planning_text.insert(tk.END, text)
        self.planning_text.see(tk.END)
        self.planning_text.config(state=tk.DISABLED)

    def _append_exec(self, text: str):
        self.exec_text.config(state=tk.NORMAL)
        self.exec_text.insert(tk.END, text)
        self.exec_text.see(tk.END)
        self.exec_text.config(state=tk.DISABLED)

    def _set_text(self, widget: scrolledtext.ScrolledText, text: str):
        widget.config(state=tk.NORMAL)
        widget.delete("1.0", tk.END)
        widget.insert(tk.END, text)
        widget.config(state=tk.DISABLED)

    @staticmethod
    def _format_goal_tree(goal: Dict[str, Any], indent: int = 0) -> str:
        lines = []
        status_icon = {
            "pending": "[待]", "active": "[执]", "completed": "[成]",
            "failed": "[败]", "blocked": "[阻]",
        }.get(goal.get("status", ""), "[?]")
        lines.append("  " * indent + f"{status_icon} {goal.get('description', 'Goal')}")
        for sub in goal.get("sub_goals", []):
            lines.append(RobotDashboard._format_goal_tree(sub, indent + 1))
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Buttons
    # ------------------------------------------------------------------
    def _on_run(self):
        instruction = self.instruction_entry.get().strip()
        if not instruction:
            return
        if self.agent is None:
            self._append_planning("[错误] Agent 未初始化\n")
            return
        if self._agent_thread is not None and self._agent_thread.is_alive():
            self._append_planning("[警告] 已有任务正在执行\n")
            return

        self._append_planning(f"[用户指令] {instruction}\n")
        self.run_btn.config(state=tk.DISABLED, text="⏳ 执行中...")

        def _run():
            try:
                self.agent.execute(instruction, sensors=self.sensors)
            except Exception as exc:
                logger.exception("Agent execution failed")
                self._event_queue.put(("task_end", {"success": False, "message": str(exc), "total_steps": 0, "elapsed_time_s": 0}))
            finally:
                self.root.after(0, lambda: self.run_btn.config(state=tk.NORMAL, text="▶ 执行任务"))

        self._agent_thread = threading.Thread(target=_run, daemon=True)
        self._agent_thread.start()

    def _on_stop(self):
        self._append_planning("[用户操作] 请求停止（按 Ctrl+C 中断后台线程）\n")

    def _on_clear(self):
        self._set_text(self.planning_text, "")
        self._set_text(self.exec_text, "")
        self._set_text(self.goal_text, "")
        self._step_count = 0
        self.step_label.config(text="步数: 0")
        self._current_action = "空闲"
        self._update_action_label()

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------
    def run(self):
        self._running = True
        self.root.after(100, self._process_events)
        # Periodically refresh arm state even when idle
        self._schedule_arm_refresh()
        self.root.mainloop()
        self._running = False

    def _schedule_arm_refresh(self):
        if not self._running:
            return
        self._update_arm_text()
        self.root.after(500, self._schedule_arm_refresh)

    def close(self):
        self._running = False
        if self.agent is not None:
            try:
                self.agent.remove_observer(self._on_agent_event)
            except Exception:
                pass
        self.root.destroy()
