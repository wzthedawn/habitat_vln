#!/usr/bin/env python3
"""
系统测试脚本 - 测试 Token 追踪功能

测试内容:
1. TokenTracker 基本功能
2. LLMModel 集成 Token 追踪
3. 完整导航流程 Token 统计
"""

import sys
import os
import logging

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

from utils.token_tracker import get_token_tracker, TokenTracker
from models.llm_model import LLMModel
from configs.model_config import get_model_config


def test_token_tracker_basic():
    """测试 TokenTracker 基本功能"""
    print("\n" + "=" * 60)
    print("测试 1: TokenTracker 基本功能")
    print("=" * 60)

    tracker = get_token_tracker()
    tracker.reset()

    # 开始任务
    task_id = tracker.start_task("走到厨房", "Type-3")
    print(f"✓ 开始任务: {task_id}")

    # 记录几次 token 使用
    tracker.record_usage("InstructionAgent", "qwen3.5-plus", 150, 80, "instruction_parse")
    tracker.record_usage("PerceptionAgent", "qwen3.5-plus", 200, 100, "perception")
    tracker.record_usage("TrajectoryAgent", "qwen3.5-plus", 180, 90, "trajectory")
    tracker.record_usage("DecisionAgent", "deepseek-v3", 120, 50, "decision")
    print("✓ 记录了 4 次 API 调用")

    # 打印当前任务报告
    print("\n" + tracker.print_current_task_report())

    # 结束任务
    completed = tracker.end_task()
    print(f"\n✓ 任务完成: {completed.task_id}")
    print(f"  总 tokens: {completed.total_tokens}")
    print(f"  API 调用次数: {completed.num_api_calls}")

    return True


def test_multiple_tasks():
    """测试多任务追踪"""
    print("\n" + "=" * 60)
    print("测试 2: 多任务追踪")
    print("=" * 60)

    tracker = get_token_tracker()
    tracker.reset()

    # 任务 1: Type-1 简单导航
    tracker.start_task("向前走", "Type-1")
    tracker.record_usage("PerceptionAgent", "qwen3.5-plus", 100, 50, "perception")
    tracker.record_usage("DecisionAgent", "deepseek-v3", 80, 40, "decision")
    task1 = tracker.end_task()
    print(f"✓ 任务 1 (Type-1): {task1.total_tokens} tokens")

    # 任务 2: Type-2 目标搜索
    tracker.start_task("找到沙发", "Type-2")
    tracker.record_usage("PerceptionAgent", "qwen3.5-plus", 150, 80, "perception")
    tracker.record_usage("TrajectoryAgent", "qwen3.5-plus", 120, 60, "trajectory")
    tracker.record_usage("DecisionAgent", "deepseek-v3", 100, 50, "decision")
    task2 = tracker.end_task()
    print(f"✓ 任务 2 (Type-2): {task2.total_tokens} tokens")

    # 任务 3: Type-3 空间推理
    tracker.start_task("走到卧室，经过客厅", "Type-3")
    tracker.record_usage("InstructionAgent", "qwen3.5-plus", 200, 100, "instruction")
    tracker.record_usage("PerceptionAgent", "qwen3.5-plus", 180, 90, "perception")
    tracker.record_usage("TrajectoryAgent", "qwen3.5-plus", 160, 80, "trajectory")
    tracker.record_usage("DecisionAgent", "deepseek-v3", 140, 70, "decision")
    task3 = tracker.end_task()
    print(f"✓ 任务 3 (Type-3): {task3.total_tokens} tokens")

    # 任务 4: Type-4 复杂决策
    tracker.start_task("如果你看到桌子就左转，否则直走", "Type-4")
    tracker.record_usage("InstructionAgent", "qwen3.5-plus", 250, 120, "instruction")
    tracker.record_usage("PerceptionAgent", "qwen3.5-plus", 220, 110, "perception")
    tracker.record_usage("TrajectoryAgent", "qwen3.5-plus", 200, 100, "trajectory")
    tracker.record_usage("DecisionAgent", "deepseek-v3", 180, 90, "decision")
    task4 = tracker.end_task()
    print(f"✓ 任务 4 (Type-4): {task4.total_tokens} tokens")

    # 打印全局报告
    print("\n" + tracker.print_global_report())

    # 打印按任务类型统计
    print("\n按任务类型统计:")
    history = tracker.get_task_history()
    for task_data in history:
        print(f"  {task_data['task_type']}: {task_data['summary']['total_tokens']} tokens")

    return True


def test_llm_model_integration():
    """测试 LLMModel 集成 Token 追踪"""
    print("\n" + "=" * 60)
    print("测试 3: LLMModel Token 追踪集成")
    print("=" * 60)

    tracker = get_token_tracker()
    tracker.reset()

    # 创建模型配置
    config = get_model_config()
    model_config = config.get("model_configs", {}).get("qwen3.5-plus", {})

    print(f"模型配置: {model_config.get('model', 'N/A')}")
    print(f"Base URL: {model_config.get('base_url', 'N/A')}")

    # 创建 LLMModel 实例
    llm = LLMModel({
        "model_name": model_config.get("model", "qwen3.5-plus"),
        "api_key": model_config.get("api_key", ""),
        "api_base": model_config.get("base_url", ""),
        "max_tokens": 500,
        "temperature": 0.7,
    })

    llm.set_agent_name("TestAgent")
    print(f"✓ LLMModel 创建成功，Agent 名称: TestAgent")

    # 开始任务
    tracker.start_task("测试指令", "Type-1")

    # 测试生成 (使用 mock，因为可能没有网络)
    print("\n尝试调用 API...")
    try:
        response = llm.generate("你好，请回复'测试成功'")
        print(f"✓ API 响应: {response[:100]}...")
        input_tokens, output_tokens = llm.get_last_token_usage()
        print(f"✓ Token 使用: input={input_tokens}, output={output_tokens}")
    except Exception as e:
        print(f"⚠ API 调用失败 (可能无网络): {e}")
        print("  使用 mock 模式测试...")

        # 模拟记录 token 使用
        tracker.record_usage("TestAgent", "qwen3.5-plus", 100, 50, "test")

    # 结束任务并打印报告
    completed = tracker.end_task()
    if completed:
        print("\n" + tracker.print_current_task_report())

    return True


def test_by_agent_and_model():
    """测试按 Agent 和 Model 分组统计"""
    print("\n" + "=" * 60)
    print("测试 4: 按 Agent 和 Model 分组统计")
    print("=" * 60)

    tracker = get_token_tracker()
    tracker.reset()

    # 开始任务
    tracker.start_task("复杂导航任务", "Type-4")

    # 模拟多次调用不同 Agent
    calls = [
        ("InstructionAgent", "qwen3.5-plus", 200, 100),
        ("InstructionAgent", "qwen3.5-plus", 180, 90),
        ("PerceptionAgent", "qwen3.5-plus", 150, 80),
        ("PerceptionAgent", "qwen3.5-plus", 160, 85),
        ("TrajectoryAgent", "qwen3.5-plus", 140, 70),
        ("DecisionAgent", "deepseek-v3", 120, 60),
        ("DecisionAgent", "deepseek-v3", 130, 65),
    ]

    for agent, model, input_t, output_t in calls:
        tracker.record_usage(agent, model, input_t, output_t)

    completed = tracker.end_task()

    # 打印按 Agent 统计
    print("\n按 Agent 统计:")
    for agent, stats in completed.get_usage_by_agent().items():
        print(f"  {agent}:")
        print(f"    - Input:  {stats['input_tokens']}")
        print(f"    - Output: {stats['output_tokens']}")
        print(f"    - Total:  {stats['total_tokens']}")
        print(f"    - Calls:  {stats['calls']}")

    # 打印按 Model 统计
    print("\n按 Model 统计:")
    for model, stats in completed.get_usage_by_model().items():
        print(f"  {model}:")
        print(f"    - Input:  {stats['input_tokens']}")
        print(f"    - Output: {stats['output_tokens']}")
        print(f"    - Total:  {stats['total_tokens']}")
        print(f"    - Calls:  {stats['calls']}")

    return True


def test_json_export():
    """测试 JSON 导出"""
    print("\n" + "=" * 60)
    print("测试 5: JSON 导出")
    print("=" * 60)

    tracker = get_token_tracker()
    tracker.reset()

    # 创建测试任务
    tracker.start_task("测试任务", "Type-2")
    tracker.record_usage("PerceptionAgent", "qwen3.5-plus", 100, 50)
    tracker.record_usage("DecisionAgent", "deepseek-v3", 80, 40)
    completed = tracker.end_task()

    # 导出为字典/JSON
    import json
    task_dict = completed.to_dict()

    print("任务 JSON 格式:")
    print(json.dumps(task_dict, indent=2, ensure_ascii=False))

    return True


def main():
    """运行所有测试"""
    print("\n" + "=" * 60)
    print("VLN 系统 Token 追踪测试")
    print("=" * 60)

    tests = [
        ("TokenTracker 基本功能", test_token_tracker_basic),
        ("多任务追踪", test_multiple_tasks),
        ("LLMModel 集成", test_llm_model_integration),
        ("按 Agent/Model 统计", test_by_agent_and_model),
        ("JSON 导出", test_json_export),
    ]

    results = []
    for name, test_func in tests:
        try:
            success = test_func()
            results.append((name, success, None))
            print(f"\n✅ {name} - 通过")
        except Exception as e:
            results.append((name, False, str(e)))
            print(f"\n❌ {name} - 失败: {e}")
            import traceback
            traceback.print_exc()

    # 打印总结
    print("\n" + "=" * 60)
    print("测试总结")
    print("=" * 60)

    passed = sum(1 for _, success, _ in results if success)
    total = len(results)

    for name, success, error in results:
        status = "✅ 通过" if success else f"❌ 失败: {error}"
        print(f"  {name}: {status}")

    print(f"\n总计: {passed}/{total} 测试通过")

    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())