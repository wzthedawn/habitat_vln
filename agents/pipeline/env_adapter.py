"""Habitat环境适配器 - 为Navigator提供统一接口"""

from typing import List, Dict, Any
import math


class HabitatEnvAdapter:
    """适配 Habitat Simulator 为 Navigator 接口

    Navigator 期望的接口：
    - get_observations() → {"rgb": ..., "depth": ...}
    - step(action_type) → 执行动作
    - get_agent_position() → [x, y, z]
    - get_agent_rotation() → float（弧度）
    """

    def __init__(
        self,
        sim,
        get_observations_func,
        config: Dict[str, Any] = None,
    ):
        """初始化适配器

        Args:
            sim: Habitat simulator 实例
            get_observations_func: 获取观察的函数 (sim) -> (rgb, depth)
            config: 配置字典
        """
        self._sim = sim
        self._get_observations = get_observations_func
        self._config = config or {}

    def get_observations(self) -> Dict[str, Any]:
        """获取当前观察

        Returns:
            {"rgb": rgb_image, "depth": depth_image}
        """
        rgb, depth = self._get_observations(self._sim)
        return {"rgb": rgb, "depth": depth}

    def step(self, action_type) -> None:
        """执行动作

        Args:
            action_type: ActionType 枚举值
        """
        # Convert ActionType enum to Habitat action string
        action_map = {
            0: "stop",      # ActionType.STOP
            1: "move_forward",  # ActionType.MOVE_FORWARD
            2: "turn_left",     # ActionType.TURN_LEFT
            3: "turn_right",    # ActionType.TURN_RIGHT
        }

        # Get action name string
        if hasattr(action_type, 'value'):
            action_name = action_map.get(action_type.value, "move_forward")
        else:
            action_name = str(action_type)

        agent = self._sim.get_agent(0)
        agent.act(action_name)

    def get_agent_position(self) -> List[float]:
        """获取当前位置

        Returns:
            [x, y, z] 位置列表
        """
        state = self._sim.get_agent(0).get_state()
        return [
            float(state.position[0]),
            float(state.position[1]),
            float(state.position[2]),
        ]

    def get_agent_rotation(self) -> float:
        """获取当前朝向（yaw，弧度）

        Returns:
            yaw 角度（弧度），正值表示左转
        """
        state = self._sim.get_agent(0).get_state()
        q = state.rotation

        # 从四元数计算 yaw (Euler angle)
        # yaw = atan2(2*(w*y + x*z), 1 - 2*(y*y + z*z))
        siny_cosp = 2 * (q.w * q.y + q.x * q.z)
        cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)

        return math.atan2(siny_cosp, cosy_cosp)