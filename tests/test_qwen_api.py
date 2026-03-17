#!/usr/bin/env python3
"""
测试通义千问 API 连接
"""

import sys
sys.path.insert(0, '/root/habitat_vln')

def test_qwen_api():
    """测试通义千问 API"""
    print("=" * 60)
    print("测试通义千问 API 连接")
    print("=" * 60)

    # 配置
    config = {
        "model_name": "qwen3.5-plus",
        "api_key": "sk-0a47c3d5c923462c9696de8d36dc08ba",
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "max_tokens": 500,
        "temperature": 0.7,
    }

    print(f"\n配置信息:")
    print(f"  模型: {config['model_name']}")
    print(f"  Base URL: {config['base_url']}")

    try:
        from models.llm_model import LLMModel

        # 创建模型实例
        llm = LLMModel(config)

        # 测试简单生成
        print("\n[测试1] 简单导航指令理解...")
        prompt = "你是一个导航助手。指令：向左转。请回答应该执行什么动作（forward/turn_left/turn_right/stop）。"
        response = llm.generate(prompt)
        print(f"  指令: 向左转")
        print(f"  响应: {response}")

        # 测试任务分类
        print("\n[测试2] 任务分类...")
        prompt = """分析以下导航指令的复杂度，并分类为 Type-0 到 Type-4：
- Type-0: 简单单步指令（如：左转、前进）
- Type-1: 路径跟随（如：沿着走廊走）
- Type-2: 目标搜索（如：找到椅子）
- Type-3: 空间推理（如：去厨房经过客厅）
- Type-4: 复杂决策（如：如果看到门就左转）

指令："找到客厅里的红色椅子"

请只回答类型（Type-0/Type-1/Type-2/Type-3/Type-4）"""

        response = llm.generate(prompt)
        print(f"  指令: 找到客厅里的红色椅子")
        print(f"  响应: {response}")

        # 测试推理
        print("\n[测试3] 导航推理...")
        prompt = """当前状态：
- 指令：走到厨房找到冰箱
- 当前位置：客厅
- 已走步数：5
- 可见物体：沙发、电视、门

请分析下一步应该采取什么动作（forward/turn_left/turn_right/stop），并说明理由。"""

        response = llm.generate(prompt)
        print(f"  响应: {response[:200]}...")

        print("\n" + "=" * 60)
        print("✅ 通义千问 API 测试成功！")
        print("=" * 60)

        return True

    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_navigation_with_qwen():
    """使用通义千问测试完整导航流程"""
    print("\n" + "=" * 60)
    print("使用通义千问测试导航系统")
    print("=" * 60)

    try:
        from core.navigator import VLNNavigator
        from core.context import NavContextBuilder

        # 配置导航器使用通义千问
        config = {
            "classifier": {
                "use_llm_fallback": True,
                "llm": {
                    "model_name": "qwen3.5-plus",
                    "api_key": "sk-0a47c3d5c923462c9696de8d36dc08ba",
                    "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
                    "max_tokens": 500,
                    "temperature": 0.7,
                }
            },
            "supernet": {
                "llm_model": {
                    "model_name": "qwen3.5-plus",
                    "api_key": "sk-0a47c3d5c923462c9696de8d36dc08ba",
                    "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
                }
            }
        }

        # 初始化导航器
        navigator = VLNNavigator(config=config, log_level="INFO")
        navigator.initialize()

        # 测试指令
        test_instructions = [
            "turn left",
            "find the kitchen",
            "go to the bedroom through the hallway",
        ]

        print("\n开始导航测试...")

        for instruction in test_instructions:
            print(f"\n指令: '{instruction}'")

            navigator.reset()
            navigator.set_instruction(instruction)
            navigator.set_position((0.0, 0.0, 0.0), 0.0)

            action = navigator.navigate()

            print(f"  动作: {action.action_type.name}")
            print(f"  置信度: {action.confidence:.2f}")
            if action.reasoning:
                print(f"  原因: {action.reasoning[:50]}...")

        print("\n" + "=" * 60)
        print("✅ 导航系统测试成功！")
        print("=" * 60)

        return True

    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    # 先测试 API 连接
    api_success = test_qwen_api()

    if api_success:
        # 再测试导航系统
        nav_success = test_navigation_with_qwen()

        if nav_success:
            print("\n🎉 所有测试通过！系统已配置完成，可以使用通义千问进行推理。")
            print("\n使用方法:")
            print("  python scripts/inference.py --instruction 'your instruction'")
            print("  python test_system.py --mode inference")
        else:
            print("\n⚠️ API 连接正常，但导航系统集成有问题")
    else:
        print("\n❌ API 连接失败，请检查配置")