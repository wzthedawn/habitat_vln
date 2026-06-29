"""Habitat环境适配器 - 为Navigator提供统一接口，支持R2R离散模式"""

from typing import List, Dict, Any, Optional
import math
import numpy as np


class HabitatEnvAdapter:
    """适配 Habitat Simulator 为 Navigator 接口

    Navigator 期望的接口：
    - get_observations() → {"rgb": ..., "depth": ...}
    - step(action_type) → 执行动作
    - get_agent_position() → [x, y, z]
    - get_agent_rotation() → float（弧度）

    R2R离散模式：set_r2r_discrete(graph) 启用viewpoint瞬移
    """

    def __init__(
        self,
        sim,
        get_observations_func,
        config: Dict[str, Any] = None,
    ):
        self._sim = sim
        self._get_observations = get_observations_func
        self._config = config or {}
        self._r2r_graph = None  # R2R discrete viewpoint graph

    @property
    def sim(self):
        """Expose underlying simulator for R2R navigation."""
        return self._sim

    def set_r2r_discrete(self, viewpoint_graph: Dict[str, Any]) -> None:
        """Enable R2R discrete nav-graph mode.

        Args:
            viewpoint_graph: Dict with 'viewpoints' and 'adjacency'
        """
        self._r2r_graph = viewpoint_graph

    def get_observations(self) -> Dict[str, Any]:
        """获取当前观察"""
        rgb, depth = self._get_observations(self._sim)
        return {"rgb": rgb, "depth": depth}

    def step(self, action_type) -> None:
        """执行动作。R2R离散模式：viewpoint瞬移；连续模式：物理步进"""
        action_map = {
            0: "stop", 1: "move_forward", 2: "turn_left", 3: "turn_right",
        }
        if hasattr(action_type, 'value'):
            action_name = action_map.get(action_type.value, "move_forward")
        else:
            action_name = str(action_type)

        if self._r2r_graph and action_name in ("move_forward", "turn_left", "turn_right"):
            self._step_r2r(action_name)
        else:
            agent = self._sim.get_agent(0)
            agent.act(action_name)

    def _step_r2r(self, action_name: str) -> None:
        """R2R discrete: viewpoint-to-viewpoint teleportation."""
        agent = self._sim.get_agent(0)
        state = agent.get_state()
        pos = np.array(state.position)
        rot = state.rotation

        # Extract yaw from quaternion
        import quaternion
        q_arr = quaternion.as_float_array(rot)
        w, x, y, z = q_arr[0], q_arr[1], q_arr[2], q_arr[3]
        siny = 2 * (w * y + x * z)
        cosy = 1 - 2 * (y * y + z * z)
        yaw = math.atan2(siny, cosy)

        if action_name == "move_forward":
            viewpoints = self._r2r_graph['viewpoints']
            if not viewpoints:
                agent.act(action_name)
                return

            # Find nearest viewpoint
            nearest_idx = min(range(len(viewpoints)),
                             key=lambda i: np.linalg.norm(np.array(viewpoints[i]) - pos))

            # Find adjacent viewpoint closest to heading
            heading_vec = np.array([math.sin(yaw), 0, math.cos(yaw)])
            best_adj = None
            best_score = -2

            for adj_info in self._r2r_graph['adjacency'].get(nearest_idx, []):
                adj_pos = np.array(adj_info['position'])
                direction = adj_pos - pos
                direction[1] = 0
                direction_norm = np.linalg.norm(direction)
                if direction_norm < 0.1:
                    continue
                direction = direction / direction_norm
                score = np.dot(heading_vec, direction)
                if score > best_score and score > 0.1:
                    best_score = score
                    best_adj = adj_info

            if best_adj:
                state.position = np.array(best_adj['position'])
                agent.set_state(state)
            else:
                agent.act(action_name)

        elif action_name in ("turn_left", "turn_right"):
            angle = math.radians(30) * (-1 if action_name == "turn_left" else 1)
            new_q = quaternion.from_rotation_vector(np.array([0, angle, 0]))
            state.rotation = new_q
            agent.set_state(state)

    def get_agent_position(self) -> List[float]:
        state = self._sim.get_agent(0).get_state()
        return [float(state.position[0]), float(state.position[1]), float(state.position[2])]

    def get_agent_rotation(self) -> float:
        state = self._sim.get_agent(0).get_state()
        q = state.rotation
        siny_cosp = 2 * (q.w * q.y + q.x * q.z)
        cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
        return math.atan2(siny_cosp, cosy_cosp)