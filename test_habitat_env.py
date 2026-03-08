#!/usr/bin/env python3
"""
Habitat 环境测试脚本

测试内容:
1. 加载测试场景
2. 创建导航任务
3. 测试 Agent 移动
4. 与 VLN 系统集成
"""

import sys
import os
import numpy as np

# 设置数据路径
DATA_PATH = "/root/habitat_vln/data/scene_datasets/habitat-test-scenes"

def test_habitat_sim():
    """测试 Habitat-sim 基本功能"""
    print("\n" + "=" * 60)
    print("测试 1: Habitat-sim 基本功能")
    print("=" * 60)

    import habitat_sim

    # 检查可用场景
    scenes = [f for f in os.listdir(DATA_PATH) if f.endswith('.glb')]
    print(f"✓ 找到 {len(scenes)} 个测试场景:")
    for scene in scenes:
        print(f"  - {scene}")

    # 创建仿真配置
    scene_file = os.path.join(DATA_PATH, "apartment_1.glb")

    if not os.path.exists(scene_file):
        print(f"✗ 场景文件不存在: {scene_file}")
        return False

    # 配置传感器
    sensor_specs = []

    # RGB 传感器
    rgb_sensor_spec = habitat_sim.sensor.SensorSpec()
    rgb_sensor_spec.uuid = "rgb_sensor"
    rgb_sensor_spec.sensor_type = habitat_sim.SensorType.COLOR
    rgb_sensor_spec.resolution = [256, 256]
    rgb_sensor_spec.position = [0.0, 1.5, 0.0]
    sensor_specs.append(rgb_sensor_spec)

    # 深度传感器
    depth_sensor_spec = habitat_sim.sensor.SensorSpec()
    depth_sensor_spec.uuid = "depth_sensor"
    depth_sensor_spec.sensor_type = habitat_sim.SensorType.DEPTH
    depth_sensor_spec.resolution = [256, 256]
    depth_sensor_spec.position = [0.0, 1.5, 0.0]
    sensor_specs.append(depth_sensor_spec)

    # Agent 配置
    agent_config = habitat_sim.agent.AgentConfiguration(
        height=1.5,
        radius=0.1,
        sensor_specifications=sensor_specs,
        action_space={
            "move_forward": habitat_sim.agent.ActionSpec(
                "move_forward", habitat_sim.agent.ActuationSpec(amount=0.25)
            ),
            "turn_left": habitat_sim.agent.ActionSpec(
                "turn_left", habitat_sim.agent.ActuationSpec(amount=15.0)
            ),
            "turn_right": habitat_sim.agent.ActionSpec(
                "turn_right", habitat_sim.agent.ActuationSpec(amount=15.0)
            ),
            "stop": habitat_sim.agent.ActionSpec(
                "stop", habitat_sim.agent.ActuationSpec(amount=0.0)
            ),
        },
    )

    # 创建仿真配置
    cfg = habitat_sim.Configuration(
        [agent_config],
        habitat_sim.SimulatorConfiguration(
            scene_id=scene_file,
            enable_physics=False,
        ),
    )

    # 创建仿真器
    sim = habitat_sim.Simulator(cfg)
    print("✓ Habitat 仿真器创建成功")

    # 获取导航网格信息
    navmesh = sim.pathfinder
    if navmesh.is_loaded:
        print(f"✓ 导航网格加载成功")
        nav_points = navmesh.navigable_points
        print(f"  - 可导航点数量: {len(nav_points)}")

        # 获取随机导航点
        random_point = navmesh.get_random_navigable_point()
        print(f"  - 随机导航点: {random_point}")

    # 测试移动
    agent = sim.get_agent(0)
    initial_state = agent.get_state()
    print(f"✓ Agent 初始位置: {initial_state.position}")

    # 执行移动动作
    observations = sim.step("move_forward")
    new_state = agent.get_state()
    print(f"✓ 移动后位置: {new_state.position}")

    # 检查观测
    if "rgb_sensor" in observations:
        rgb = observations["rgb_sensor"]
        print(f"✓ RGB 观测: shape={rgb.shape}, dtype={rgb.dtype}")

    if "depth_sensor" in observations:
        depth = observations["depth_sensor"]
        print(f"✓ Depth 观测: shape={depth.shape}, dtype={depth.dtype}")

    # 关闭仿真器
    sim.close()
    print("✓ Habitat-sim 测试通过")

    return True


def test_habitat_lab():
    """测试 Habitat-lab 环境"""
    print("\n" + "=" * 60)
    print("测试 2: Habitat-lab 导航任务")
    print("=" * 60)

    import habitat
    from habitat.config.default import get_config
    from habitat.core.env import Env

    # 创建 PointNav 任务配置
    config = get_config()

    # 修改配置使用测试场景
    config.habitat.simulator.scene = os.path.join(DATA_PATH, "apartment_1.glb")
    config.habitat.simulator.agent_0.height = 1.5
    config.habitat.simulator.agent_0.radius = 0.1

    # 设置任务
    config.habitat.task.type = "PointNav-v1"
    config.habitat.task.measurements = ["distance_to_goal", "success", "spl"]

    # 设置数据集
    config.habitat.dataset.type = "PointNav-v1"
    config.habitat.dataset.data_path = ""
    config.habitat.dataset.scenes_dir = DATA_PATH

    print("✓ 配置创建成功")

    # 创建环境
    try:
        env = Env(config=config)
        print("✓ Habitat 环境创建成功")

        # 重置环境
        obs = env.reset()
        print(f"✓ 环境重置成功")
        print(f"  - 观测键: {list(obs.keys())}")

        # 执行几步
        for i in range(3):
            action = np.random.randint(0, 3)  # 随机动作
            obs, reward, done, info = env.step(action)
            print(f"  步骤 {i+1}: reward={reward:.4f}, done={done}")

        env.close()
        print("✓ Habitat-lab 测试通过")
        return True

    except Exception as e:
        print(f"✗ Habitat-lab 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_vln_integration():
    """测试与 VLN 系统集成"""
    print("\n" + "=" * 60)
    print("测试 3: VLN 系统集成")
    print("=" * 60)

    import habitat_sim
    from utils.token_tracker import get_token_tracker

    # 创建 Habitat 环境
    scene_file = os.path.join(DATA_PATH, "apartment_1.glb")

    # 简单配置
    agent_config = habitat_sim.agent.AgentConfiguration(
        height=1.5,
        radius=0.1,
        sensor_specifications=[],
        action_space={
            "move_forward": habitat_sim.agent.ActionSpec("move_forward", habitat_sim.agent.ActuationSpec(amount=0.25)),
            "turn_left": habitat_sim.agent.ActionSpec("turn_left", habitat_sim.agent.ActuationSpec(amount=15.0)),
            "turn_right": habitat_sim.agent.ActionSpec("turn_right", habitat_sim.agent.ActuationSpec(amount=15.0)),
            "stop": habitat_sim.agent.ActionSpec("stop", habitat_sim.agent.ActuationSpec(amount=0.0)),
        },
    )

    cfg = habitat_sim.Configuration(
        [agent_config],
        habitat_sim.SimulatorConfiguration(scene_id=scene_file),
    )

    sim = habitat_sim.Simulator(cfg)
    agent = sim.get_agent(0)

    print("✓ Habitat 仿真器创建成功")

    # 初始化 Token 追踪
    tracker = get_token_tracker()
    tracker.reset()

    # 开始导航任务
    instruction = "走到房间的另一边"
    task_id = tracker.start_task(instruction, "Type-1")

    print(f"✓ 任务开始: {task_id}")
    print(f"  指令: {instruction}")

    # 模拟导航过程
    print("\n模拟导航过程:")

    # 记录初始位置
    initial_pos = agent.get_state().position
    print(f"  初始位置: {initial_pos}")

    # 执行一系列动作
    actions = ["move_forward", "move_forward", "turn_left", "move_forward", "move_forward"]
    for i, action in enumerate(actions):
        sim.step(action)
        tracker.record_usage("DecisionAgent", "Qwen/Qwen3-30B-A3B-Thinking-2507", 100, 50, action)

    final_pos = agent.get_state().position
    distance = np.linalg.norm(np.array(final_pos) - np.array(initial_pos))
    print(f"  最终位置: {final_pos}")
    print(f"  移动距离: {distance:.2f}m")

    # 结束任务
    completed = tracker.end_task()
    print(f"\n✓ 任务完成")
    print(f"  总 Tokens: {completed.total_tokens}")
    print(f"  API 调用: {completed.num_api_calls}")

    # 打印报告
    print("\n" + tracker.print_current_task_report())

    sim.close()
    print("✓ VLN 集成测试通过")

    return True


def main():
    """运行所有测试"""
    print("\n" + "=" * 60)
    print("Habitat 环境测试")
    print("=" * 60)

    # 检查场景文件
    if not os.path.exists(DATA_PATH):
        print(f"✗ 场景目录不存在: {DATA_PATH}")
        return 1

    results = {}

    # 测试 1: Habitat-sim
    try:
        results["habitat_sim"] = test_habitat_sim()
    except Exception as e:
        print(f"✗ Habitat-sim 测试失败: {e}")
        import traceback
        traceback.print_exc()
        results["habitat_sim"] = False

    # 测试 2: Habitat-lab
    try:
        results["habitat_lab"] = test_habitat_lab()
    except Exception as e:
        print(f"✗ Habitat-lab 测试失败: {e}")
        results["habitat_lab"] = False

    # 测试 3: VLN 集成
    try:
        results["vln_integration"] = test_vln_integration()
    except Exception as e:
        print(f"✗ VLN 集成测试失败: {e}")
        import traceback
        traceback.print_exc()
        results["vln_integration"] = False

    # 打印总结
    print("\n" + "=" * 60)
    print("测试总结")
    print("=" * 60)

    passed = sum(1 for v in results.values() if v)
    total = len(results)

    for name, success in results.items():
        status = "✅ 通过" if success else "❌ 失败"
        print(f"  {name}: {status}")

    print(f"\n总计: {passed}/{total} 测试通过")

    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())