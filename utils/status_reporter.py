"""实时状态报告器 - 用于调试和监控VLN实验"""

import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional
import threading


class StatusReporter:
    """实时状态报告器，将运行状态写入JSON和HTML文件"""

    def __init__(self, output_dir: str = "."):
        self.status_file = Path(output_dir) / "realtime_status.json"
        self.html_file = Path(output_dir) / "realtime_dashboard.html"
        self.lock = threading.Lock()

        # 初始化状态
        self.status = {
            "timestamp": "",
            "episode": 0,
            "step": 0,
            "max_steps": 0,
            "current_phase": "初始化中...",
            "instruction": "",
            "current_subtask": None,
            "position": {"x": 0, "y": 0, "z": 0},
            "start_y": 0,
            "y_change": 0,
            "agents": {
                "perception": {"status": "waiting", "output": ""},
                "instruction": {"status": "waiting", "output": "", "subtasks": 0},
                "trajectory": {"status": "waiting", "output": ""},
                "decision": {"status": "waiting", "output": "", "sequence_progress": "0%"},
            },
            "actions_executed": [],
            "recent_log": [],
            "subtask_completed": False,
        }

        self._write()

    def start_episode(self, episode_id: int, instruction: str, max_steps: int, start_y: float):
        """开始新episode"""
        with self.lock:
            self.status["episode"] = episode_id
            self.status["instruction"] = instruction
            self.status["max_steps"] = max_steps
            self.status["step"] = 0
            self.status["start_y"] = start_y
            self.status["y_change"] = 0
            self.status["actions_executed"] = []
            self.status["recent_log"] = []
            self.status["subtask_completed"] = False
            self._update_timestamp()
            self._write()

    def update_phase(self, phase: str):
        """更新当前阶段"""
        with self.lock:
            self.status["current_phase"] = phase
            self._update_timestamp()
            self._write()

    def update_step(self, step: int):
        """更新步数"""
        with self.lock:
            self.status["step"] = step
            self._update_timestamp()
            self._write()

    def update_agent(self, agent_name: str, status: str, output: str = "", **kwargs):
        """更新Agent状态"""
        with self.lock:
            if agent_name in self.status["agents"]:
                self.status["agents"][agent_name]["status"] = status
                if output:
                    self.status["agents"][agent_name]["output"] = output[:500]  # 限制长度
                for k, v in kwargs.items():
                    self.status["agents"][agent_name][k] = v
            self._update_timestamp()
            self._write()

    def update_subtask(self, subtask_id: int, description: str, completion_condition: Dict = None, completed: bool = False):
        """更新子任务状态"""
        with self.lock:
            self.status["current_subtask"] = {
                "id": subtask_id,
                "description": description,
                "completion_condition": completion_condition,
                "completed": completed
            }
            self.status["subtask_completed"] = completed
            self._update_timestamp()
            self._write()

    def update_position(self, x: float, y: float, z: float):
        """更新位置"""
        with self.lock:
            self.status["position"] = {"x": round(x, 2), "y": round(y, 2), "z": round(z, 2)}
            self.status["y_change"] = round(y - self.status.get("start_y", y), 2)
            self._update_timestamp()
            self._write()

    def log_action(self, action: str):
        """记录执行的动作"""
        with self.lock:
            self.status["actions_executed"].append(action)
            # 保持最近20个
            if len(self.status["actions_executed"]) > 20:
                self.status["actions_executed"] = self.status["actions_executed"][-20:]
            self._update_timestamp()
            self._write()

    def log(self, message: str):
        """添加日志"""
        with self.lock:
            timestamp = datetime.now().strftime("%H:%M:%S")
            self.status["recent_log"].append(f"[{timestamp}] {message}")
            # 保持最近10条
            if len(self.status["recent_log"]) > 10:
                self.status["recent_log"] = self.status["recent_log"][-10:]
            self._write()

    def _update_timestamp(self):
        """更新时间戳"""
        self.status["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def _write(self):
        """写入JSON和HTML文件"""
        # 写JSON
        with open(self.status_file, 'w', encoding='utf-8') as f:
            json.dump(self.status, f, ensure_ascii=False, indent=2)

        # 写HTML
        self._write_html()

    def _write_html(self):
        """生成HTML看板"""
        html = f'''<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta http-equiv="refresh" content="2">
    <title>VLN 实时状态</title>
    <style>
        body {{
            font-family: 'Segoe UI', Arial, sans-serif;
            background: #1a1a2e;
            color: #eee;
            margin: 0;
            padding: 20px;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
        }}
        h1 {{
            color: #00d4ff;
            text-align: center;
        }}
        .phase {{
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            padding: 15px;
            border-radius: 10px;
            text-align: center;
            font-size: 1.3em;
            margin-bottom: 20px;
            animation: pulse 2s infinite;
        }}
        @keyframes pulse {{
            0%, 100% {{ opacity: 1; }}
            50% {{ opacity: 0.7; }}
        }}
        .grid {{
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 15px;
        }}
        .card {{
            background: #16213e;
            border-radius: 10px;
            padding: 15px;
            border-left: 4px solid #00d4ff;
        }}
        .card h3 {{
            margin-top: 0;
            color: #00d4ff;
        }}
        .agent-card {{
            border-left-color: #ffd700;
        }}
        .agent-thinking {{
            border-left-color: #ff6b6b;
            animation: pulse 1s infinite;
        }}
        .status-waiting {{ color: #888; }}
        .status-thinking {{ color: #ffd700; }}
        .status-done {{ color: #4ade80; }}
        .progress-bar {{
            background: #2d3748;
            border-radius: 5px;
            height: 20px;
            overflow: hidden;
            margin: 10px 0;
        }}
        .progress-fill {{
            background: linear-gradient(90deg, #00d4ff, #4ade80);
            height: 100%;
            transition: width 0.3s;
        }}
        .log {{
            font-family: monospace;
            font-size: 0.85em;
            background: #0f0f23;
            padding: 10px;
            border-radius: 5px;
            max-height: 200px;
            overflow-y: auto;
        }}
        .log-entry {{
            margin: 3px 0;
            border-bottom: 1px solid #333;
            padding-bottom: 3px;
        }}
        .y-positive {{ color: #ff6b6b; }}
        .y-negative {{ color: #4ade80; }}
        .actions {{
            display: flex;
            flex-wrap: wrap;
            gap: 5px;
        }}
        .action-tag {{
            background: #2d3748;
            padding: 3px 8px;
            border-radius: 3px;
            font-size: 0.85em;
        }}
        .subtask-done {{
            background: #166534;
            border-radius: 5px;
            padding: 10px;
            margin-top: 10px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🧭 VLN 导航实时状态</h1>

        <div class="phase">
            🔄 {self.status['current_phase']}
        </div>

        <div class="grid">
            <div class="card">
                <h3>📊 进度</h3>
                <p>Episode: <strong>{self.status['episode']}</strong> / 步数: <strong>{self.status['step']}</strong> / {self.status['max_steps']}</p>
                <div class="progress-bar">
                    <div class="progress-fill" style="width: {(self.status['step'] / max(self.status['max_steps'], 1)) * 100}%"></div>
                </div>
            </div>

            <div class="card">
                <h3>📍 位置</h3>
                <p>X: {self.status['position']['x']:.2f} | Y: {self.status['position']['y']:.2f} | Z: {self.status['position']['z']:.2f}</p>
                <p>Y变化: <span class="{'y-negative' if self.status['y_change'] < 0 else 'y-positive'}">{self.status['y_change']:.2f}m</span></p>
            </div>

            <div class="card">
                <h3>📝 指令</h3>
                <p style="font-size: 0.9em;">{self.status['instruction'][:100]}...</p>
            </div>

            <div class="card">
                <h3>🎯 当前子任务</h3>
                <p>{self.status['current_subtask']['description'] if self.status['current_subtask'] else '无'}</p>
                <p><small>完成条件: {self.status['current_subtask']['completion_condition'] if self.status['current_subtask'] else ''}</small></p>
                {f'<div class="subtask-done">✅ 子任务已完成</div>' if self.status.get('subtask_completed') else ''}
            </div>
        </div>

        <h3 style="color: #00d4ff; margin-top: 20px;">🤖 Agent 状态</h3>
        <div class="grid">
            {self._generate_agent_cards()}
        </div>

        <h3 style="color: #00d4ff; margin-top: 20px;">🎮 已执行动作</h3>
        <div class="actions">
            {' '.join(f'<span class="action-tag">{a}</span>' for a in self.status['actions_executed'][-15:])}
        </div>

        <h3 style="color: #00d4ff; margin-top: 20px;">📋 最近日志</h3>
        <div class="log">
            {'<br>'.join(f"<div class='log-entry'>{log}</div>" for log in self.status['recent_log'][-8:])}
        </div>

        <p style="text-align: center; color: #666; margin-top: 20px;">
            最后更新: {self.status['timestamp']} | 自动刷新: 2秒
        </p>
    </div>
</body>
</html>'''

        with open(self.html_file, 'w', encoding='utf-8') as f:
            f.write(html)

    def _generate_agent_cards(self) -> str:
        """生成Agent卡片HTML"""
        cards = []
        agent_names = {
            "observation": "👁️ ObservationAgent",
            "analysis": "🧠 AnalysisAgent",
            "planning": "🗺️ PlanningAgent",
            "review": "✅ ReviewAgent",
            "emergency": "🚨 EmergencyAgent",
            "decomposition": "📋 DecompositionAgent",
        }

        for key, name in agent_names.items():
            agent = self.status['agents'].get(key, {})
            status = agent.get('status', 'waiting')
            status_class = f"status-{status}"
            thinking_class = "agent-thinking" if status == "thinking" else ""

            cards.append(f'''
            <div class="card agent-card {thinking_class}">
                <h3>{name}</h3>
                <p>状态: <span class="{status_class}">{status}</span></p>
                {f'<p style="font-size:0.85em;color:#aaa;">{agent.get("output", "")[:80]}...</p>' if agent.get('output') else ''}
                {f'<p>序列进度: {agent.get("sequence_progress", "0%")}</p>' if key == 'decision' else ''}
                {f'<p>子任务数: {agent.get("subtasks", 0)}</p>' if key == 'instruction' else ''}
            </div>''')

        return '\n'.join(cards)


# 全局实例
_reporter: Optional[StatusReporter] = None


def get_reporter(output_dir: str = ".") -> StatusReporter:
    """获取全局报告器实例"""
    global _reporter
    if _reporter is None:
        _reporter = StatusReporter(output_dir)
    return _reporter


def init_reporter(output_dir: str = "."):
    """初始化全局报告器"""
    global _reporter
    _reporter = StatusReporter(output_dir)
    return _reporter