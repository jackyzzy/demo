#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os
import io
import logging
import warnings
from contextlib import redirect_stdout, redirect_stderr

# 设置标准输入输出编码为UTF-8
try:
    sys.stdin.reconfigure(encoding='utf-8')
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
except AttributeError:
    # Python版本较老，使用环境变量
    os.environ['PYTHONIOENCODING'] = 'utf-8'

# 配置日志，抑制详细输出
logging.basicConfig(level=logging.ERROR)
warnings.filterwarnings("ignore")

# 添加src目录到Python路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from agents import ParlantAgent

# 添加根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from model_config import ModelManager, display_available_models

def create_agent_quietly(model_key):
    """静默创建Agent，抑制创建时的输出"""
    import io
    import sys
    from contextlib import redirect_stdout, redirect_stderr

    # 创建一个空的输出缓冲区
    null_output = io.StringIO()

    try:
        # 重定向标准输出和错误输出
        with redirect_stdout(null_output), redirect_stderr(null_output):
            agent = ParlantAgent(model_key=model_key)
        return agent
    except Exception as e:
        # 如果创建失败，显示错误信息
        print(f"❌ 创建Agent失败: {e}")
        raise

def demo_parlant_agent():
    """Parlant Agent 演示"""
    print("\n" + "="*60)
    print("🚀 Parlant智能助手演示")
    print("="*60)

    # 显示可用模型
    available_models = ModelManager.get_models_by_availability()

    if not available_models:
        print("❌ 没有可用的模型，请检查API密钥配置")
        display_available_models()
        return

    print("🤖 可用的模型：")
    model_list = list(available_models.items())

    for i, (key, config) in enumerate(model_list, 1):
        print(f"{i}. ({config.provider.value}){key}: {config.name} ")
        print(f"   {config.description}")

    # 选择模型
    while True:
        try:
            choice = input(f"\n请选择模型 (1-{len(model_list)}, 默认1): ").strip()
            if not choice:
                model_key = model_list[0][0]
                break

            choice_idx = int(choice) - 1
            if 0 <= choice_idx < len(model_list):
                model_key = model_list[choice_idx][0]
                break
            else:
                print(f"❌ 请输入1-{len(model_list)}之间的数字")
        except ValueError:
            print("❌ 请输入有效的数字")

    # 创建智能助手（静默模式）
    print(f"\n🔧 正在初始化智能助手...")
    agent = create_agent_quietly(model_key)
    print(f"✅ 智能助手初始化完成！")

    # 测试场景
    test_cases = [
        {
            "name": "简单对话",
            "message": "你好！请介绍一下你的能力",
        },
        {
            "name": "数学计算",
            "message": "计算复合增长率：初值100，年增长率15%，5年后的值是多少？"
        },
        {
            "name": "信息研究",
            "message": "2024年AI大模型的最新发展趋势是什么？"
        },
        {
            "name": "股票分析",
            "message": "分析微软(MSFT)的股票投资价值，包括基本面和技术面"
        },
        {
            "name": "复杂推理",
            "message": "如果要在2025年开始投资AI相关股票，应该考虑哪些因素？请给出详细的分析框架"
        }
    ]

    for i, test in enumerate(test_cases, 1):
        print(f"\n{'='*40}")
        print(f"📋 测试 {i}: {test['name']}")
        print(f"{'='*40}")
        print(f"❓ 问题: {test['message']}")
        print("\n💭 处理中...")

        # 执行测试
        response = agent.chat(test['message'], session_id=f"demo_{i}")

        print(f"\n🤖 回答:")
        print("-" * 40)
        # 提取响应内容
        if hasattr(response, 'content'):
            print(response.content)
        elif hasattr(response, 'text'):
            print(response.text)
        else:
            print(str(response))
        print("-" * 40)

        # 询问是否继续
        if i < len(test_cases):
            continue_test = input(f"\n继续下一个测试？(y/n, 默认y): ").strip().lower()
            if continue_test == 'n':
                break

    print(f"\n✅ Parlant Agent演示完成！")

    # 显示会话历史示例
    print(f"\n📚 会话历史示例 (最后一个会话):")
    history = agent.get_chat_history("demo_1")
    for entry in history[-4:]:  # 显示最后4条记录
        print(f"  {entry[:100]}...")

def interactive_mode():
    """交互模式"""
    print("\n" + "="*60)
    print("🎯 进入交互模式")
    print("="*60)

    # 显示可用模型
    available_models = ModelManager.get_models_by_availability()

    if not available_models:
        print("❌ 没有可用的模型，请检查API密钥配置")
        display_available_models()
        return

    print("🤖 可用的模型：")
    model_list = list(available_models.items())

    for i, (key, config) in enumerate(model_list, 1):
        print(f"{i}. {key}: {config.name} ({config.provider.value})")

    # 选择模型
    while True:
        try:
            choice = input(f"\n请选择模型 (1-{len(model_list)}, 默认1): ").strip()
            if not choice:
                model_key = model_list[0][0]
                break

            choice_idx = int(choice) - 1
            if 0 <= choice_idx < len(model_list):
                model_key = model_list[choice_idx][0]
                break
            else:
                print(f"❌ 请输入1-{len(model_list)}之间的数字")
        except ValueError:
            print("❌ 请输入有效的数字")

    print(f"\n🔧 正在初始化智能助手...")
    agent = create_agent_quietly(model_key)
    print(f"✅ 智能助手初始化完成！")

    print("\n✨ 智能助手已就绪！输入 'quit' 退出，'clear' 清除历史")
    print(f"💡 提示：我可以帮你搜索信息、分析数据、计算数学题、分析股票等")

    session_id = "interactive"

    while True:
        try:
            user_input = input(f"\n💬 你: ").strip()

            if user_input.lower() == 'quit':
                print("👋 再见！")
                break
            elif user_input.lower() == 'clear':
                agent.clear_history(session_id)
                continue
            elif user_input.lower() == 'history':
                history = agent.get_chat_history(session_id)
                print("\n📚 对话历史:")
                for entry in history:
                    print(f"  {entry}")
                continue
            elif not user_input:
                continue

            print(f"\n🤖 助手: 思考中...")
            response = agent.chat(user_input, session_id)
            print(f"\n🤖 助手: {response}")

        except KeyboardInterrupt:
            print(f"\n\n👋 再见！")
            break
        except Exception as e:
            print(f"❌ 出错了: {str(e)}")

if __name__ == "__main__":
    print("🎊 欢迎使用Parlant智能助手！")
    print("\n选择运行模式:")
    print("1. 演示模式 (自动运行测试案例)")
    print("2. 交互模式 (手动对话)")

    mode = input("\n请选择模式 (1 或 2，默认1): ").strip()

    if mode == "2":
        interactive_mode()
    else:
        demo_parlant_agent()
