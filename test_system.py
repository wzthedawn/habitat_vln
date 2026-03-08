#!/usr/bin/env python3
"""
快速测试脚本 - 测试多智能体VLN导航系统

用法:
    python test_system.py --mode basic      # 基础功能测试
    python test_system.py --mode inference  # 推理测试
    python test_system.py --mode full       # 完整流程测试
"""

import argparse
import logging
import sys

# 设置路径
sys.path.insert(0, '/root/habitat_vln')

from core.context import NavContext, NavContextBuilder, VisualFeatures
from core.action import Action, ActionType
from core.navigator import VLNNavigator
from classifiers import TaskTypeClassifier
from supernet import Supernet
from agents import InstructionAgent, PerceptionAgent, TrajectoryAgent, DecisionAgent
from strategies import ReActStrategy, CoTStrategy, DebateStrategy, ReflectionStrategy
from utils.logger import setup_logger


def test_basic_functionality():
    """测试基础功能"""
    print("=" * 60)
    print("1. 基础功能测试")
    print("=" * 60)

    # 测试上下文构建
    print("\n[1.1] 测试上下文构建...")
    builder = NavContextBuilder()
    builder.with_instruction("turn left and go forward")
    builder.with_position((0.0, 0.0, 0.0))
    builder.with_rotation(0.0)
    context = builder.build()
    print(f"  ✓ 上下文创建成功: instruction='{context.instruction[:30]}...'")
    print(f"  ✓ 位置: {context.position}")

    # 测试动作创建
    print("\n[1.2] 测试动作创建...")
    action = Action.forward(confidence=0.8)
    print(f"  ✓ 前进动作: {action.action_type.name}, confidence={action.confidence}")

    action = Action.turn_left(confidence=0.9, reasoning="instruction says turn left")
    print(f"  ✓ 左转动作: {action.action_type.name}, reasoning='{action.reasoning}'")

    action = Action.stop(confidence=1.0)
    print(f"  ✓ 停止动作: {action.action_type.name}")

    # 测试任务分类器
    print("\n[1.3] 测试任务分类器...")
    classifier = TaskTypeClassifier({'use_llm_fallback': False})

    test_instructions = [
        ("turn left", "Type-0"),
        ("walk down the hallway", "Type-1"),
        ("find the red chair", "Type-2"),
        ("go to the kitchen through the living room", "Type-3"),
        ("if you see a door turn left, otherwise go straight", "Type-4"),
    ]

    for instruction, expected in test_instructions:
        ctx = NavContextBuilder().with_instruction(instruction).build()
        task_type = classifier.classify(ctx)
        status = "✓" if task_type.value == expected else "✗"
        print(f"  {status} '{instruction[:30]}...' -> {task_type.value} (期望: {expected})")

    print("\n✅ 基础功能测试通过!")
    return True


def test_agents():
    """测试智能体"""
    print("\n" + "=" * 60)
    print("2. 智能体测试")
    print("=" * 60)

    # 测试指令智能体
    print("\n[2.1] 测试指令智能体...")
    instruction_agent = InstructionAgent()
    context = NavContextBuilder().with_instruction(
        "turn left and walk to the kitchen"
    ).build()
    output = instruction_agent.process(context)
    print(f"  ✓ 成功: {output.success}")
    print(f"  ✓ 子任务数: {len(output.data.get('subtasks', []))}")
    print(f"  ✓ 地标: {output.data.get('landmarks', [])}")

    # 测试感知智能体
    print("\n[2.2] 测试感知智能体...")
    perception_agent = PerceptionAgent()
    visual_features = VisualFeatures(
        scene_description="A living room with a sofa and TV",
        object_detections=[
            {"name": "sofa", "confidence": 0.9},
            {"name": "TV", "confidence": 0.85},
        ],
        room_classification="living_room",
    )
    context = NavContextBuilder().with_instruction("test").build()
    context.visual_features = visual_features
    output = perception_agent.process(context)
    print(f"  ✓ 成功: {output.success}")
    print(f"  ✓ 房间类型: {output.data.get('room_type', 'unknown')}")

    # 测试轨迹智能体
    print("\n[2.3] 测试轨迹智能体...")
    trajectory_agent = TrajectoryAgent()
    context = NavContextBuilder().with_instruction("test").build()
    context.trajectory = [(0, 0, 0), (1, 0, 0), (2, 0, 0), (3, 0, 0)]
    context.step_count = 4
    output = trajectory_agent.process(context)
    print(f"  ✓ 成功: {output.success}")
    print(f"  ✓ 进度: {output.data.get('progress_percentage', 0):.1f}%")

    # 测试决策智能体
    print("\n[2.4] 测试决策智能体...")
    decision_agent = DecisionAgent()
    context = NavContextBuilder().with_instruction("turn right").build()
    output = decision_agent.process(context)
    print(f"  ✓ 成功: {output.success}")
    print(f"  ✓ 动作: {output.data.get('action', 'unknown')}")
    print(f"  ✓ 置信度: {output.confidence:.2f}")

    print("\n✅ 智能体测试通过!")
    return True


def test_strategies():
    """测试策略"""
    print("\n" + "=" * 60)
    print("3. 策略测试")
    print("=" * 60)

    context = NavContextBuilder().with_instruction(
        "find the kitchen"
    ).build()
    context.visual_features = VisualFeatures(scene_description="A hallway")
    agents = [DecisionAgent()]

    # 测试ReAct策略
    print("\n[3.1] 测试ReAct策略...")
    react = ReActStrategy()
    result = react.execute(context, agents)
    print(f"  ✓ 成功: {result.success}")
    print(f"  ✓ 动作: {result.action.action_type.name if result.action else 'None'}")
    print(f"  ✓ 步骤数: {len(result.steps)}")

    # 测试CoT策略
    print("\n[3.2] 测试CoT策略...")
    cot = CoTStrategy()
    result = cot.execute(context, agents)
    print(f"  ✓ 成功: {result.success}")
    print(f"  ✓ 动作: {result.action.action_type.name if result.action else 'None'}")
    print(f"  ✓ 推理步骤: {len(result.steps)}")

    # 测试Debate策略
    print("\n[3.3] 测试Debate策略...")
    debate = DebateStrategy()
    result = debate.execute(context, agents)
    print(f"  ✓ 成功: {result.success}")
    print(f"  ✓ 动作: {result.action.action_type.name if result.action else 'None'}")

    # 测试Reflection策略
    print("\n[3.4] 测试Reflection策略...")
    reflection = ReflectionStrategy()
    context.action_history = [Action.forward(), Action.turn_left()]
    result = reflection.execute(context, agents)
    print(f"  ✓ 成功: {result.success}")
    print(f"  ✓ 动作: {result.action.action_type.name if result.action else 'None'}")
    print(f"  ✓ 存储的教训数: {len(reflection.get_lessons())}")

    print("\n✅ 策略测试通过!")
    return True


def test_supernet():
    """测试超网"""
    print("\n" + "=" * 60)
    print("4. 超网测试")
    print("=" * 60)

    supernet = Supernet()

    # 测试不同任务类型
    test_cases = [
        ("turn left", "Type-0"),
        ("walk down the hallway", "Type-1"),
        ("find the chair", "Type-2"),
        ("go to kitchen via living room", "Type-3"),
        ("if you see X turn left", "Type-4"),
    ]

    for instruction, expected_type in test_cases:
        context = NavContextBuilder().with_instruction(instruction).build()

        # 设置任务类型
        from core.context import TaskType
        type_map = {
            "Type-0": TaskType.TYPE_0,
            "Type-1": TaskType.TYPE_1,
            "Type-2": TaskType.TYPE_2,
            "Type-3": TaskType.TYPE_3,
            "Type-4": TaskType.TYPE_4,
        }
        context.task_type = type_map[expected_type]

        action = supernet.forward(context)
        print(f"  ✓ '{instruction[:25]}...' -> {action.action_type.name}")

    # 获取统计信息
    stats = supernet.get_statistics()
    print(f"\n  超网统计: 总调用={stats['total_forwards']}")

    print("\n✅ 超网测试通过!")
    return True


def test_full_navigation():
    """测试完整导航流程"""
    print("\n" + "=" * 60)
    print("5. 完整导航流程测试")
    print("=" * 60)

    # 初始化导航器
    print("\n[5.1] 初始化导航器...")
    navigator = VLNNavigator(
        config={'classifier': {'use_llm_fallback': False}},
        enable_fallback=True,
        log_level="WARNING",
    )
    navigator.initialize()
    print("  ✓ 导航器初始化完成")

    # 测试多个指令
    test_instructions = [
        "turn left",
        "go forward and find the door",
        "walk to the kitchen through the hallway",
    ]

    print("\n[5.2] 测试导航推理...")
    for instruction in test_instructions:
        navigator.reset()
        navigator.set_instruction(instruction)
        navigator.set_position((0.0, 0.0, 0.0), 0.0)

        action = navigator.navigate()

        print(f"\n  指令: '{instruction}'")
        print(f"  动作: {action.action_type.name}")
        print(f"  置信度: {action.confidence:.2f}")
        if action.reasoning:
            print(f"  原因: {action.reasoning[:50]}...")

    print("\n✅ 完整导航流程测试通过!")
    return True


def test_inference_mode():
    """交互式推理测试"""
    print("\n" + "=" * 60)
    print("6. 交互式推理测试")
    print("=" * 60)

    # 初始化导航器
    navigator = VLNNavigator(
        config={'classifier': {'use_llm_fallback': False}},
        enable_fallback=True,
        log_level="WARNING",
    )
    navigator.initialize()

    # 预设的测试指令
    test_instructions = [
        "turn left and go forward",
        "find the red chair in the living room",
        "go to the kitchen and stop at the refrigerator",
        "if you see a door turn right, otherwise keep going",
    ]

    print("\n运行预设指令测试:")
    print("-" * 40)

    for instruction in test_instructions:
        navigator.reset()
        navigator.set_instruction(instruction)

        # 模拟多步导航
        print(f"\n指令: '{instruction}'")
        print("动作序列:")

        for step in range(5):  # 最多5步
            action = navigator.navigate()
            print(f"  步骤 {step+1}: {action.action_type.name} (conf: {action.confidence:.2f})")

            if action.action_type == ActionType.STOP:
                break

    print("\n✅ 交互式推理测试完成!")
    return True


def main():
    parser = argparse.ArgumentParser(description="测试多智能体VLN导航系统")
    parser.add_argument(
        "--mode",
        type=str,
        default="basic",
        choices=["basic", "agents", "strategies", "supernet", "full", "inference", "all"],
        help="测试模式",
    )

    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("  多智能体VLN导航系统 - 测试套件")
    print("=" * 60)

    results = {}

    try:
        if args.mode in ["basic", "all"]:
            results["basic"] = test_basic_functionality()

        if args.mode in ["agents", "all"]:
            results["agents"] = test_agents()

        if args.mode in ["strategies", "all"]:
            results["strategies"] = test_strategies()

        if args.mode in ["supernet", "all"]:
            results["supernet"] = test_supernet()

        if args.mode in ["full", "all"]:
            results["full"] = test_full_navigation()

        if args.mode in ["inference"]:
            results["inference"] = test_inference_mode()

        # 总结
        print("\n" + "=" * 60)
        print("测试结果总结")
        print("=" * 60)

        for test_name, passed in results.items():
            status = "✅ 通过" if passed else "❌ 失败"
            print(f"  {test_name}: {status}")

        total = len(results)
        passed = sum(1 for v in results.values() if v)
        print(f"\n总计: {passed}/{total} 测试通过")

        if passed == total:
            print("\n🎉 所有测试通过! 系统可以正常使用。")
            print("\n下一步:")
            print("  - 运行推理: python scripts/inference.py --instruction 'your instruction'")
            print("  - 运行评估: python scripts/evaluate.py --episodes 10")
            print("  - 查看配置: cat configs/default.yaml")

    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())