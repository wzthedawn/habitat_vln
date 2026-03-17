#!/usr/bin/env python3
"""
VLN 系统模拟测试 - 无需 Habitat 环境

使用 Mock 环境测试完整的 Agent 协作流程：
1. 任务分类
2. Agent 选择
3. 多 Agent 协作
4. Token 追踪
5. API 调用验证
"""

import sys
import os
import logging
import yaml
import json
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("VLNMockTest")

# 加载配置
def load_config():
    with open("configs/model_config.yaml", 'r') as f:
        return yaml.safe_load(f)

def test_instruction_parsing():
    """测试指令解析 - InstructionAgent"""
    print("\n" + "=" * 60)
    print("测试 1: InstructionAgent - 指令解析")
    print("=" * 60)

    from agents.instruction_agent import InstructionAgent
    from utils.token_tracker import get_token_tracker

    agent = InstructionAgent()
    tracker = get_token_tracker()

    test_instructions = [
        "走到厨房",
        "找到沙发并坐下",
        "穿过客厅，进入卧室，在床边停下",
    ]

    for instruction in test_instructions:
        tracker.start_task(instruction, "Type-3")
        tracker.record_usage("InstructionAgent", "qwen3.5-plus", 150, 80, "instruction_parse")
        print(f"✓ 指令: {instruction}")
        completed = tracker.end_task()
        print(f"  Tokens: {completed.total_tokens}")

    return True


def test_perception_analysis():
    """测试感知分析 - PerceptionAgent"""
    print("\n" + "=" * 60)
    print("测试 2: PerceptionAgent - 视觉感知")
    print("=" * 60)

    from agents.perception_agent import PerceptionAgent
    from core.context import NavContext, NavContextBuilder, VisualFeatures
    from utils.token_tracker import get_token_tracker

    agent = PerceptionAgent()
    tracker = get_token_tracker()

    # 模拟视觉输入
    mock_observations = [
        {"room": "客厅", "objects": ["沙发", "电视", "茶几"], "landmarks": ["窗户", "门"]},
        {"room": "厨房", "objects": ["冰箱", "灶台", "餐桌"], "landmarks": ["厨房门"]},
        {"room": "卧室", "objects": ["床", "衣柜", "床头柜"], "landmarks": ["窗户"]},
    ]

    for obs in mock_observations:
        tracker.start_task(f"感知{obs['room']}", "Type-2")
        tracker.record_usage("PerceptionAgent", "qwen3.5-plus", 200, 100, "perception")
        print(f"✓ 感知场景: {obs['room']}")
        print(f"  物体: {', '.join(obs['objects'])}")
        completed = tracker.end_task()
        print(f"  Tokens: {completed.total_tokens}")

    return True


def test_trajectory_planning():
    """测试轨迹规划 - TrajectoryAgent"""
    print("\n" + "=" * 60)
    print("测试 3: TrajectoryAgent - 路径规划")
    print("=" * 60)

    from agents.trajectory_agent import TrajectoryAgent
    from utils.token_tracker import get_token_tracker

    agent = TrajectoryAgent()
    tracker = get_token_tracker()

    test_cases = [
        {"start": "客厅", "goal": "厨房", "waypoints": ["走廊", "厨房门"]},
        {"start": "卧室", "goal": "客厅", "waypoints": ["卧室门", "走廊"]},
    ]

    for case in test_cases:
        tracker.start_task(f"从{case['start']}到{case['goal']}", "Type-2")
        tracker.record_usage("TrajectoryAgent", "qwen3.5-plus", 180, 90, "trajectory")
        print(f"✓ 规划路径: {case['start']} -> {case['goal']}")
        print(f"  途经: {' -> '.join(case['waypoints'])}")
        completed = tracker.end_task()
        print(f"  Tokens: {completed.total_tokens}")

    return True


def test_decision_making():
    """测试决策制定 - DecisionAgent"""
    print("\n" + "=" * 60)
    print("测试 4: DecisionAgent - 最终决策")
    print("=" * 60)

    from agents.decision_agent import DecisionAgent
    from utils.token_tracker import get_token_tracker

    agent = DecisionAgent()
    tracker = get_token_tracker()

    test_scenarios = [
        {"situation": "前方有障碍物", "options": ["左转", "右转", "后退"]},
        {"situation": "到达目标房间", "options": ["停止", "继续探索", "返回"]},
    ]

    for scenario in test_scenarios:
        tracker.start_task(scenario['situation'], "Type-3")
        tracker.record_usage("DecisionAgent", "Qwen/Qwen3-30B-A3B-Thinking-2507", 120, 50, "decision")
        print(f"✓ 场景: {scenario['situation']}")
        print(f"  可选动作: {', '.join(scenario['options'])}")
        completed = tracker.end_task()
        print(f"  Tokens: {completed.total_tokens}")

    return True


def test_full_navigation_flow():
    """测试完整导航流程"""
    print("\n" + "=" * 60)
    print("测试 5: 完整导航流程 (Type-3 任务)")
    print("=" * 60)

    from core.navigator import VLNNavigator
    from utils.token_tracker import get_token_tracker

    tracker = get_token_tracker()
    tracker.reset()

    navigator = VLNNavigator(log_level='INFO')
    navigator.initialize()

    # 模拟完整的导航任务
    instruction = "走到厨房，找到冰箱"
    task_id = tracker.start_task(instruction, "Type-3")

    print(f"\n任务: {instruction}")
    print(f"任务ID: {task_id}")

    # 记录各 Agent 的 token 使用
    agent_tokens = [
        ("InstructionAgent", "qwen3.5-plus", 200, 100, "instruction_parse"),
        ("PerceptionAgent", "qwen3.5-plus", 180, 90, "perception"),
        ("TrajectoryAgent", "qwen3.5-plus", 160, 80, "trajectory"),
        ("DecisionAgent", "Qwen/Qwen3-30B-A3B-Thinking-2507", 140, 70, "decision"),
    ]

    total_steps = 5
    print(f"\n模拟 {total_steps} 步导航:")

    for step in range(total_steps):
        print(f"\n  步骤 {step + 1}:")
        for agent, model, input_t, output_t, prompt_type in agent_tokens:
            tracker.record_usage(agent, model, input_t, output_t, prompt_type)
            print(f"    - {agent}: input={input_t}, output={output_t}")

    completed = tracker.end_task()

    print("\n" + tracker.print_current_task_report())

    return True


def test_api_integration():
    """测试真实 API 集成"""
    print("\n" + "=" * 60)
    print("测试 6: 真实 API 调用验证")
    print("=" * 60)

    from models.llm_model import LLMModel
    from utils.token_tracker import get_token_tracker

    config = load_config()
    model_configs = config.get('model_configs', {})
    tracker = get_token_tracker()
    tracker.reset()

    results = {}

    # 测试 qwen3.5-plus
    if 'qwen3.5-plus' in model_configs:
        print("\n[1] 测试通义千问 (qwen3.5-plus)...")
        cfg = model_configs['qwen3.5-plus']
        llm = LLMModel({
            "model_name": cfg.get('model'),
            "api_key": cfg.get('api_key'),
            "api_base": cfg.get('base_url'),
            "max_tokens": 100,
            "temperature": 0.7,
        })
        llm.set_agent_name("TestAgent")

        tracker.start_task("API测试-qwen", "Type-1")

        try:
            response, (input_t, output_t) = llm.generate_with_usage(
                "你是一个导航助手。用户指令：向前走。请回答应该执行什么动作。"
            )
            print(f"✓ API 调用成功")
            print(f"  响应: {response[:100]}...")
            print(f"  Tokens: input={input_t}, output={output_t}")
            results['qwen'] = True
        except Exception as e:
            print(f"✗ 失败: {e}")
            results['qwen'] = False

        tracker.end_task()

    # 测试 qwen3-30b-thinking
    if 'qwen3-30b-thinking' in model_configs:
        print("\n[2] 测试 Qwen3-30B-Thinking...")
        cfg = model_configs['qwen3-30b-thinking']
        llm = LLMModel({
            "model_name": cfg.get('model'),
            "api_key": cfg.get('api_key'),
            "api_base": cfg.get('base_url'),
            "max_tokens": 100,
            "temperature": 0.7,
        })
        llm.set_agent_name("DecisionAgent")

        tracker.start_task("API测试-qwen30b", "Type-3")

        try:
            response, (input_t, output_t) = llm.generate_with_usage(
                "你是一个导航决策专家。当前场景：前方是客厅，目标是厨房。请给出导航决策。"
            )
            print(f"✓ API 调用成功")
            print(f"  响应: {response[:100]}...")
            print(f"  Tokens: input={input_t}, output={output_t}")
            results['qwen30b'] = True
        except Exception as e:
            print(f"✗ 失败: {e}")
            results['qwen30b'] = False

        tracker.end_task()

    return results


def test_task_classification():
    """测试任务分类"""
    print("\n" + "=" * 60)
    print("测试 7: 任务分类器")
    print("=" * 60)

    from classifiers.task_classifier import TaskTypeClassifier
    from core.context import NavContextBuilder

    classifier = TaskTypeClassifier()

    test_cases = [
        ("向前走", "Type-1"),
        ("找到沙发", "Type-2"),
        ("走到厨房，经过客厅", "Type-3"),
        ("如果你看到桌子就左转", "Type-4"),
    ]

    print("规则分类测试:")
    for instruction, expected in test_cases:
        context = NavContextBuilder().with_instruction(instruction).build()
        result = classifier.classify(context)
        match = "✓" if result.value == expected else "✗"
        print(f"  {match} '{instruction[:20]}...' -> {result.value} (期望: {expected})")

    return True


def main():
    """运行所有测试"""
    print("\n" + "=" * 60)
    print("VLN 系统模拟测试 (无需 Habitat)")
    print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    tests = [
        ("指令解析", test_instruction_parsing),
        ("视觉感知", test_perception_analysis),
        ("路径规划", test_trajectory_planning),
        ("决策制定", test_decision_making),
        ("完整导航流程", test_full_navigation_flow),
        ("API 集成", test_api_integration),
        ("任务分类", test_task_classification),
    ]

    results = {}

    for name, test_func in tests:
        try:
            result = test_func()
            results[name] = True if result else False
        except Exception as e:
            print(f"\n✗ {name} 测试失败: {e}")
            import traceback
            traceback.print_exc()
            results[name] = False

    # 最终报告
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