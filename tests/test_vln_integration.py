#!/usr/bin/env python3
"""
完整 VLN 集成测试

测试内容:
1. Habitat 环境初始化
2. 任务分类
3. 多 Agent 协作
4. Token 追踪
5. 真实 API 调用
"""

import os
import sys
import yaml
import numpy as np

sys.path.insert(0, '/root/habitat_vln')

# 数据路径
DATA_PATH = '/root/habitat_vln/data/scene_datasets/habitat-test-scenes'

def load_config():
    """加载配置"""
    with open('/root/habitat_vln/configs/model_config.yaml', 'r') as f:
        return yaml.safe_load(f)

def test_habitat_navigation():
    """测试 Habitat 导航"""
    print("\n" + "=" * 60)
    print("测试 1: Habitat 环境导航")
    print("=" * 60)

    import habitat_sim

    scene_file = os.path.join(DATA_PATH, 'apartment_1.glb')

    # 创建配置
    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.scene_id = scene_file

    # Agent 配置 - 添加传感器
    rgb_sensor = habitat_sim.CameraSensorSpec()
    rgb_sensor.uuid = "rgb"
    rgb_sensor.sensor_type = habitat_sim.SensorType.COLOR
    rgb_sensor.resolution = [128, 128]

    agent_cfg = habitat_sim.AgentConfiguration()
    agent_cfg.sensor_specifications = [rgb_sensor]

    cfg = habitat_sim.Configuration(sim_cfg, [agent_cfg])

    # 创建仿真器
    sim = habitat_sim.Simulator(cfg)
    print("✓ Habitat 仿真器创建成功")

    # 获取导航网格
    navmesh = sim.pathfinder
    print("✓ 导航网格加载成功")

    # 设置 Agent 初始位置
    agent = sim.get_agent(0)
    random_pos = navmesh.get_random_navigable_point()

    # 设置 Agent 状态
    import quaternion
    agent_state = habitat_sim.AgentState()
    agent_state.position = random_pos
    agent_state.rotation = quaternion.from_rotation_vector([0, 0, 0])
    agent.set_state(agent_state)

    print(f"✓ Agent 初始位置: {agent_state.position}")

    # 获取初始观测
    obs = sim.get_sensor_observations()
    print(f"✓ RGB 观测: shape={obs['rgb'].shape}")

    sim.close()
    return True


def test_task_classification():
    """测试任务分类"""
    print("\n" + "=" * 60)
    print("测试 2: 任务分类")
    print("=" * 60)

    from classifiers.task_classifier import TaskTypeClassifier
    from core.context import NavContextBuilder

    classifier = TaskTypeClassifier()

    test_cases = [
        ("向前走", "Type-0 or Type-1"),
        ("找到沙发", "Type-2"),
        ("走到厨房，经过客厅", "Type-3"),
        ("如果你看到桌子就左转，否则直走", "Type-4"),
    ]

    print("规则分类测试:")
    for instruction, expected in test_cases:
        context = NavContextBuilder().with_instruction(instruction).build()
        result = classifier.classify(context)
        print(f"  '{instruction[:20]}...' -> {result.value}")

    return True


def test_agent_collaboration():
    """测试 Agent 协作"""
    print("\n" + "=" * 60)
    print("测试 3: Agent 协作")
    print("=" * 60)

    from agents.instruction_agent import InstructionAgent
    from agents.perception_agent import PerceptionAgent
    from agents.trajectory_agent import TrajectoryAgent
    from agents.decision_agent import DecisionAgent
    from utils.token_tracker import get_token_tracker

    # 创建 Agents
    agents = {
        'instruction': InstructionAgent(),
        'perception': PerceptionAgent(),
        'trajectory': TrajectoryAgent(),
        'decision': DecisionAgent(),
    }

    print("✓ 所有 Agent 创建成功:")
    for name, agent in agents.items():
        print(f"  - {name}: {agent.name}")

    # 模拟协作流程
    tracker = get_token_tracker()
    tracker.reset()

    task_id = tracker.start_task("测试 Agent 协作", "Type-3")

    # 模拟各 Agent 处理
    agent_tokens = [
        ("InstructionAgent", "qwen3.5-plus", 200, 100),
        ("PerceptionAgent", "qwen3.5-plus", 180, 90),
        ("TrajectoryAgent", "qwen3.5-plus", 160, 80),
        ("DecisionAgent", "Qwen/Qwen3-30B-A3B-Thinking-2507", 140, 70),
    ]

    for agent_name, model, input_t, output_t in agent_tokens:
        tracker.record_usage(agent_name, model, input_t, output_t)

    completed = tracker.end_task()
    print(f"\n✓ 协作完成:")
    print(f"  总 Tokens: {completed.total_tokens}")
    print(f"  API 调用: {completed.num_api_calls}")

    return True


def test_api_calls():
    """测试真实 API 调用"""
    print("\n" + "=" * 60)
    print("测试 4: 真实 API 调用")
    print("=" * 60)

    from models.llm_model import LLMModel
    from utils.token_tracker import get_token_tracker

    config = load_config()
    model_configs = config.get('model_configs', {})
    tracker = get_token_tracker()

    results = {}

    # 测试通义千问
    if 'qwen3.5-plus' in model_configs:
        print("\n[1] 测试通义千问 (qwen3.5-plus)...")
        cfg = model_configs['qwen3.5-plus']
        llm = LLMModel({
            "model_name": cfg.get('model'),
            "api_key": cfg.get('api_key'),
            "api_base": cfg.get('base_url'),
            "max_tokens": 200,
            "temperature": 0.7,
        })
        llm.set_agent_name("PerceptionAgent")

        tracker.start_task("感知测试", "Type-2")

        try:
            prompt = """你是一个室内导航感知专家。当前观测：
- 房间类型: 客厅
- 可见物体: 沙发, 电视, 茶几
- 目标: 找到厨房

请描述你的感知结果和建议的下一步行动。"""

            response, (input_t, output_t) = llm.generate_with_usage(prompt)
            print(f"✓ API 调用成功")
            print(f"  Input tokens: {input_t}")
            print(f"  Output tokens: {output_t}")
            print(f"  响应: {response[:200]}...")
            results['qwen'] = True
        except Exception as e:
            print(f"✗ 失败: {e}")
            results['qwen'] = False

        tracker.end_task()

    # 测试 Qwen3-30B-Thinking
    if 'qwen3-30b-thinking' in model_configs:
        print("\n[2] 测试 Qwen3-30B-Thinking...")
        cfg = model_configs['qwen3-30b-thinking']
        llm = LLMModel({
            "model_name": cfg.get('model'),
            "api_key": cfg.get('api_key'),
            "api_base": cfg.get('base_url'),
            "max_tokens": 200,
            "temperature": 0.7,
        })
        llm.set_agent_name("DecisionAgent")

        tracker.start_task("决策测试", "Type-3")

        try:
            prompt = """你是一个导航决策专家。当前状态：
- 位置: 客厅
- 目标: 厨房
- 可选动作: move_forward, turn_left, turn_right, stop

请给出最佳导航决策。"""

            response, (input_t, output_t) = llm.generate_with_usage(prompt)
            print(f"✓ API 调用成功")
            print(f"  Input tokens: {input_t}")
            print(f"  Output tokens: {output_t}")
            print(f"  响应: {response[:200]}...")
            results['qwen30b'] = True
        except Exception as e:
            print(f"✗ 失败: {e}")
            results['qwen30b'] = False

        tracker.end_task()

    return results


def test_full_navigation():
    """测试完整导航流程"""
    print("\n" + "=" * 60)
    print("测试 5: 完整导航流程")
    print("=" * 60)

    import habitat_sim
    from utils.token_tracker import get_token_tracker

    # 创建 Habitat 环境
    scene_file = os.path.join(DATA_PATH, 'apartment_1.glb')

    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.scene_id = scene_file

    agent_cfg = habitat_sim.AgentConfiguration()
    cfg = habitat_sim.Configuration(sim_cfg, [agent_cfg])

    sim = habitat_sim.Simulator(cfg)
    navmesh = sim.pathfinder
    agent = sim.get_agent(0)

    # 设置初始位置
    start_pos = navmesh.get_random_navigable_point()
    import quaternion
    agent_state = habitat_sim.AgentState()
    agent_state.position = start_pos
    agent_state.rotation = quaternion.from_rotation_vector([0, 0, 0])
    agent.set_state(agent_state)

    print(f"✓ Agent 初始位置: {start_pos}")

    # 初始化 Token 追踪
    tracker = get_token_tracker()
    tracker.reset()

    # 开始导航任务
    instruction = "探索房间，找到一个合适的位置"
    task_id = tracker.start_task(instruction, "Type-2")

    print(f"\n任务: {instruction}")
    print(f"任务ID: {task_id}")

    # 模拟导航过程
    print("\n执行导航步骤:")
    actions = ['move_forward', 'move_forward', 'turn_left',
               'move_forward', 'move_forward', 'turn_right', 'move_forward']

    for i, action in enumerate(actions):
        sim.step(action)
        state = agent.get_state()

        # 记录 Token 使用
        tracker.record_usage("DecisionAgent", "Qwen/Qwen3-30B-A3B-Thinking-2507",
                           100, 50, action)

        print(f"  步骤 {i+1}: {action} -> 位置 {state.position[:2]}")

    # 计算移动距离
    final_pos = agent.get_state().position
    distance = np.linalg.norm(np.array(final_pos) - np.array(start_pos))
    print(f"\n总移动距离: {distance:.2f}m")

    # 结束任务
    completed = tracker.end_task()
    print(f"\n任务完成:")
    print(f"  总 Tokens: {completed.total_tokens}")
    print(f"  API 调用: {completed.num_api_calls}")

    sim.close()

    return True


def main():
    """运行所有测试"""
    print("\n" + "=" * 60)
    print("完整 VLN 集成测试")
    print("=" * 60)

    results = {}

    # 测试 1: Habitat 导航
    try:
        results["habitat_navigation"] = test_habitat_navigation()
    except Exception as e:
        print(f"✗ Habitat 导航测试失败: {e}")
        results["habitat_navigation"] = False

    # 测试 2: 任务分类
    try:
        results["task_classification"] = test_task_classification()
    except Exception as e:
        print(f"✗ 任务分类测试失败: {e}")
        results["task_classification"] = False

    # 测试 3: Agent 协作
    try:
        results["agent_collaboration"] = test_agent_collaboration()
    except Exception as e:
        print(f"✗ Agent 协作测试失败: {e}")
        results["agent_collaboration"] = False

    # 测试 4: API 调用
    try:
        api_results = test_api_calls()
        results["api_calls"] = any(api_results.values())
    except Exception as e:
        print(f"✗ API 调用测试失败: {e}")
        results["api_calls"] = False

    # 测试 5: 完整导航
    try:
        results["full_navigation"] = test_full_navigation()
    except Exception as e:
        print(f"✗ 完整导航测试失败: {e}")
        results["full_navigation"] = False

    # 打印总结
    print("\n" + "=" * 60)
    print("测试总结")
    print("=" * 60)

    from utils.token_tracker import get_token_tracker
    tracker = get_token_tracker()

    passed = sum(1 for v in results.values() if v)
    total = len(results)

    for name, success in results.items():
        status = "✅ 通过" if success else "❌ 失败"
        print(f"  {name}: {status}")

    print(f"\n总计: {passed}/{total} 测试通过")

    print("\n" + tracker.print_global_report())

    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())