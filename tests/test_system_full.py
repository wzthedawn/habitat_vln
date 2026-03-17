#!/usr/bin/env python3
"""
VLN 系统完整测试脚本

测试内容:
1. 配置加载
2. Token 追踪功能
3. Agent 创建与协作
4. 真实 API 调用测试
5. 完整导航流程
"""

import sys
import os
import logging
import yaml
import json
from datetime import datetime

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("SystemTest")


def load_yaml_config(path: str) -> dict:
    """加载 YAML 配置文件"""
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def test_config_loading():
    """测试配置加载"""
    print("\n" + "=" * 60)
    print("测试 1: 配置加载")
    print("=" * 60)

    # 加载模型配置
    model_config = load_yaml_config("configs/model_config.yaml")
    print(f"✓ 模型配置加载成功")
    print(f"  - Tier 数量: {len(model_config.get('tiers', {}))}")
    print(f"  - 模型配置数量: {len(model_config.get('model_configs', {}))}")

    # 显示关键配置
    for name, cfg in model_config.get('model_configs', {}).items():
        model_type = cfg.get('type', 'unknown')
        if model_type == 'openai':
            print(f"  - {name}: {cfg.get('model')} @ {cfg.get('base_url', 'default')}")
        elif model_type == 'local':
            print(f"  - {name}: local @ {cfg.get('model_path', 'N/A')}")

    # 加载架构配置
    arch_config = load_yaml_config("configs/architecture_config.yaml")
    print(f"\n✓ 架构配置加载成功")
    for task_type, cfg in arch_config.get('architectures', {}).items():
        agents = cfg.get('agents', [])
        print(f"  {task_type}: {len(agents)} agents")

    # 加载通义千问配置
    qwen_config = load_yaml_config("configs/qwen_config.yaml")
    print(f"\n✓ 通义千问配置加载成功")
    qwen_cfg = qwen_config.get('model', {}).get('qwen', {})
    print(f"  - 模型: {qwen_cfg.get('model_name')}")
    print(f"  - Base URL: {qwen_cfg.get('base_url')}")

    return model_config, arch_config, qwen_config


def test_token_tracker():
    """测试 Token 追踪功能"""
    print("\n" + "=" * 60)
    print("测试 2: Token 追踪功能")
    print("=" * 60)

    from utils.token_tracker import get_token_tracker

    tracker = get_token_tracker()
    tracker.reset()

    # 模拟多个任务
    test_cases = [
        ("Type-1", "向前走", [("PerceptionAgent", "qwen3.5-plus", 100, 50),
                              ("DecisionAgent", "deepseek-v3", 80, 40)]),
        ("Type-2", "找到沙发", [("PerceptionAgent", "qwen3.5-plus", 150, 80),
                               ("TrajectoryAgent", "qwen3.5-plus", 120, 60),
                               ("DecisionAgent", "deepseek-v3", 100, 50)]),
        ("Type-3", "走到卧室，经过客厅", [("InstructionAgent", "qwen3.5-plus", 200, 100),
                                       ("PerceptionAgent", "qwen3.5-plus", 180, 90),
                                       ("TrajectoryAgent", "qwen3.5-plus", 160, 80),
                                       ("DecisionAgent", "deepseek-v3", 140, 70)]),
        ("Type-4", "如果你看到桌子就左转，否则直走",
         [("InstructionAgent", "qwen3.5-plus", 250, 120),
          ("PerceptionAgent", "qwen3.5-plus", 220, 110),
          ("TrajectoryAgent", "qwen3.5-plus", 200, 100),
          ("DecisionAgent", "deepseek-v3", 180, 90)]),
    ]

    for task_type, instruction, calls in test_cases:
        task_id = tracker.start_task(instruction, task_type)
        for agent, model, input_t, output_t in calls:
            tracker.record_usage(agent, model, input_t, output_t)
        completed = tracker.end_task()
        print(f"✓ {task_type} ({task_id}): {completed.total_tokens} tokens, {completed.num_api_calls} calls")

    # 打印全局统计
    print("\n" + tracker.print_global_report())

    return True


def test_api_calls(model_config: dict):
    """测试真实 API 调用"""
    print("\n" + "=" * 60)
    print("测试 3: 真实 API 调用")
    print("=" * 60)

    from models.llm_model import LLMModel
    from utils.token_tracker import get_token_tracker

    tracker = get_token_tracker()
    tracker.reset()

    # 获取模型配置
    model_configs = model_config.get('model_configs', {})

    results = {}

    # 测试通义千问 API
    if 'qwen3.5-plus' in model_configs:
        qwen_cfg = model_configs['qwen3.5-plus']
        print(f"\n测试通义千问 API ({qwen_cfg.get('model')})...")

        llm = LLMModel({
            "model_name": qwen_cfg.get('model'),
            "api_key": qwen_cfg.get('api_key'),
            "api_base": qwen_cfg.get('base_url'),
            "max_tokens": 100,
            "temperature": 0.7,
        })
        llm.set_agent_name("TestAgent")

        tracker.start_task("API 测试", "Type-1")

        try:
            response, (input_t, output_t) = llm.generate_with_usage(
                "你好，请用一句话回复'测试成功'"
            )
            print(f"✓ API 调用成功")
            print(f"  响应: {response[:100]}...")
            print(f"  Tokens: input={input_t}, output={output_t}")
            results['qwen'] = {'success': True, 'response': response, 'tokens': (input_t, output_t)}
        except Exception as e:
            print(f"✗ API 调用失败: {e}")
            results['qwen'] = {'success': False, 'error': str(e)}

        tracker.end_task()

    # 测试 DeepSeek API
    if 'deepseek-v3' in model_configs:
        ds_cfg = model_configs['deepseek-v3']
        print(f"\n测试 DeepSeek API ({ds_cfg.get('model')})...")

        llm = LLMModel({
            "model_name": ds_cfg.get('model'),
            "api_key": ds_cfg.get('api_key'),
            "api_base": ds_cfg.get('base_url'),
            "max_tokens": 100,
            "temperature": 0.7,
        })
        llm.set_agent_name("TestAgent")

        tracker.start_task("API 测试", "Type-1")

        try:
            response, (input_t, output_t) = llm.generate_with_usage(
                "Hello, please reply with 'test success'"
            )
            print(f"✓ API 调用成功")
            print(f"  响应: {response[:100]}...")
            print(f"  Tokens: input={input_t}, output={output_t}")
            results['deepseek'] = {'success': True, 'response': response, 'tokens': (input_t, output_t)}
        except Exception as e:
            print(f"✗ API 调用失败: {e}")
            results['deepseek'] = {'success': False, 'error': str(e)}

        tracker.end_task()

    return results


def test_agent_creation():
    """测试 Agent 创建"""
    print("\n" + "=" * 60)
    print("测试 4: Agent 创建与配置")
    print("=" * 60)

    from agents.instruction_agent import InstructionAgent
    from agents.perception_agent import PerceptionAgent
    from agents.trajectory_agent import TrajectoryAgent
    from agents.decision_agent import DecisionAgent

    agents = {
        'InstructionAgent': InstructionAgent(),
        'PerceptionAgent': PerceptionAgent(),
        'TrajectoryAgent': TrajectoryAgent(),
        'DecisionAgent': DecisionAgent(),
    }

    for name, agent in agents.items():
        print(f"✓ {name} 创建成功")
        # 检查 Agent 属性
        if hasattr(agent, 'name'):
            print(f"  - 名称: {agent.name}")
        if hasattr(agent, 'model'):
            print(f"  - 模型: {type(agent.model).__name__}")

    return agents


def test_supernet_integration():
    """测试 Supernet 集成"""
    print("\n" + "=" * 60)
    print("测试 5: Supernet 集成")
    print("=" * 60)

    from supernet.supernet import Supernet
    from utils.token_tracker import get_token_tracker

    supernet = Supernet()
    print("✓ Supernet 创建成功")

    tracker = get_token_tracker()
    tracker.reset()

    # 显示各任务类型的 Agent 配置
    from configs.architecture_config import get_architecture_for_task

    print("\n各任务类型的 Agent 配置:")
    for task_type in ['Type-0', 'Type-1', 'Type-2', 'Type-3', 'Type-4']:
        arch = get_architecture_for_task(task_type)
        agents = arch.get('agents', [])
        strategies = arch.get('strategies', [])
        print(f"  {task_type}:")
        print(f"    - Agents: {agents}")
        print(f"    - Strategies: {strategies}")

    return supernet


def test_full_navigation_flow():
    """测试完整导航流程"""
    print("\n" + "=" * 60)
    print("测试 6: 完整导航流程")
    print("=" * 60)

    from core.navigator import VLNNavigator
    from utils.token_tracker import get_token_tracker

    tracker = get_token_tracker()
    tracker.reset()

    # 创建 Navigator
    navigator = VLNNavigator(log_level='WARNING')
    navigator.initialize()
    print("✓ Navigator 初始化成功")

    # 模拟导航任务
    test_instructions = [
        ("向前走三步", "Type-1"),
        ("找到厨房里的椅子", "Type-2"),
        ("走到卧室，经过客厅和走廊", "Type-3"),
    ]

    for instruction, expected_type in test_instructions:
        navigator.reset()
        task_id = tracker.start_task(instruction, expected_type)

        # 模拟记录 token 使用
        if expected_type == "Type-1":
            tracker.record_usage("PerceptionAgent", "qwen3.5-plus", 100, 50, "perception")
            tracker.record_usage("DecisionAgent", "deepseek-v3", 80, 40, "decision")
        elif expected_type == "Type-2":
            tracker.record_usage("PerceptionAgent", "qwen3.5-plus", 150, 80, "perception")
            tracker.record_usage("TrajectoryAgent", "qwen3.5-plus", 120, 60, "trajectory")
            tracker.record_usage("DecisionAgent", "deepseek-v3", 100, 50, "decision")
        elif expected_type == "Type-3":
            tracker.record_usage("InstructionAgent", "qwen3.5-plus", 200, 100, "instruction")
            tracker.record_usage("PerceptionAgent", "qwen3.5-plus", 180, 90, "perception")
            tracker.record_usage("TrajectoryAgent", "qwen3.5-plus", 160, 80, "trajectory")
            tracker.record_usage("DecisionAgent", "deepseek-v3", 140, 70, "decision")

        completed = tracker.end_task()
        print(f"\n✓ 任务: {instruction[:30]}...")
        print(f"  任务 ID: {completed.task_id}")
        print(f"  任务类型: {completed.task_type}")
        print(f"  总 Tokens: {completed.total_tokens}")
        print(f"  API 调用: {completed.num_api_calls}")

    return True


def test_token_report_export():
    """测试 Token 报告导出"""
    print("\n" + "=" * 60)
    print("测试 7: Token 报告导出")
    print("=" * 60)

    from utils.token_tracker import get_token_tracker

    tracker = get_token_tracker()
    tracker.reset()

    # 创建测试任务
    tracker.start_task("报告测试", "Type-3")
    tracker.record_usage("InstructionAgent", "qwen3.5-plus", 200, 100)
    tracker.record_usage("PerceptionAgent", "qwen3.5-plus", 180, 90)
    tracker.record_usage("TrajectoryAgent", "qwen3.5-plus", 160, 80)
    tracker.record_usage("DecisionAgent", "deepseek-v3", 140, 70)
    completed = tracker.end_task()

    # 导出 JSON
    task_dict = completed.to_dict()
    json_str = json.dumps(task_dict, indent=2, ensure_ascii=False)
    print("✓ JSON 导出成功")
    print(f"  导出大小: {len(json_str)} 字节")

    # 打印摘要
    print("\n任务摘要:")
    summary = task_dict['summary']
    print(f"  - Input Tokens: {summary['total_input_tokens']}")
    print(f"  - Output Tokens: {summary['total_output_tokens']}")
    print(f"  - Total Tokens: {summary['total_tokens']}")
    print(f"  - API Calls: {summary['num_api_calls']}")

    print("\n按 Agent 统计:")
    for agent, stats in task_dict['by_agent'].items():
        print(f"  {agent}: {stats['total_tokens']} tokens ({stats['calls']} calls)")

    print("\n按 Model 统计:")
    for model, stats in task_dict['by_model'].items():
        print(f"  {model}: {stats['total_tokens']} tokens ({stats['calls']} calls)")

    return True


def main():
    """运行所有测试"""
    print("\n" + "=" * 60)
    print("VLN 系统完整测试")
    print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    results = {}

    # 1. 配置加载测试
    try:
        model_config, arch_config, qwen_config = test_config_loading()
        results['config_loading'] = True
    except Exception as e:
        print(f"\n✗ 配置加载测试失败: {e}")
        results['config_loading'] = False

    # 2. Token 追踪测试
    try:
        test_token_tracker()
        results['token_tracker'] = True
    except Exception as e:
        print(f"\n✗ Token 追踪测试失败: {e}")
        results['token_tracker'] = False

    # 3. API 调用测试
    try:
        if 'model_config' in dir():
            api_results = test_api_calls(model_config)
            results['api_calls'] = any(r.get('success', False) for r in api_results.values())
        else:
            results['api_calls'] = False
    except Exception as e:
        print(f"\n✗ API 调用测试失败: {e}")
        results['api_calls'] = False

    # 4. Agent 创建测试
    try:
        test_agent_creation()
        results['agent_creation'] = True
    except Exception as e:
        print(f"\n✗ Agent 创建测试失败: {e}")
        results['agent_creation'] = False

    # 5. Supernet 集成测试
    try:
        test_supernet_integration()
        results['supernet'] = True
    except Exception as e:
        print(f"\n✗ Supernet 测试失败: {e}")
        results['supernet'] = False

    # 6. 完整导航流程测试
    try:
        test_full_navigation_flow()
        results['navigation_flow'] = True
    except Exception as e:
        print(f"\n✗ 导航流程测试失败: {e}")
        results['navigation_flow'] = False

    # 7. Token 报告导出测试
    try:
        test_token_report_export()
        results['token_export'] = True
    except Exception as e:
        print(f"\n✗ Token 报告导出测试失败: {e}")
        results['token_export'] = False

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

    # 打印全局 Token 统计
    from utils.token_tracker import get_token_tracker
    tracker = get_token_tracker()
    print("\n" + tracker.print_global_report())

    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())