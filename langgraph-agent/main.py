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

# 配置日志，抑制HTTP模型创建的详细输出
logging.basicConfig(level=logging.ERROR)
warnings.filterwarnings("ignore")

# 添加src目录到Python路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from agents import LangGraphAgent

# 添加根目录到Python路径
sys.path.insert(0, os.path.dirname(__file__))

from model_config import ModelManager, display_available_models

def create_agent_quietly(model_key):
    """静默创建Agent，抑制HTTP模型创建时的输出"""
    import io
    import sys
    from contextlib import redirect_stdout, redirect_stderr
    
    # 创建一个空的输出缓冲区
    null_output = io.StringIO()
    
    try:
        # 重定向标准输出和错误输出
        with redirect_stdout(null_output), redirect_stderr(null_output):
            agent = LangGraphAgent(model_key=model_key)
        return agent
    except Exception as e:
        # 如果创建失败，显示错误信息
        print(f"❌ 创建Agent失败: {e}")
        raise

def demo_langgraph_agent():
    """LangGraph Agent 演示"""
    print("\n" + "="*60)
    print("🚀 LangGraph智能助手演示")
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
            "message": "你好，请介绍一下自己的能力"
        },
        {
            "name": "数学计算", 
            "message": "请帮我计算 (125 + 75) * 2 - 50"
        },
        {
            "name": "信息研究",
            "message": "请调研一下2024年人工智能的最新发展趋势"
        },
        {
            "name": "股票分析",
            "message": "分析一下特斯拉(TSLA)股票的投资价值"
        }
    ]
    
    for i, test in enumerate(test_cases, 1):
        print(f"\n📋 测试{i}：{test['name']}")
        print(f"问题：{test['message']}")
        print("\n💭 处理中...")
        
        response = agent.chat(test['message'], session_id=f"demo_{i}")
        
        print(f"\n🤖 回答：")
        print(response)
        
        # 显示推理步骤
        steps = agent.get_reasoning_steps(f"demo_{i}")
        if steps:
            print(f"\n🧠 推理步骤：")
            for step in steps:
                print(f"  • {step}")
        
        print("\n" + "-"*40)
    
    print("\n✅ LangGraph Agent演示完成！")

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
    
    print("\n✨ 智能助手已就绪！输入 'quit' 退出")
    print(f"💡 提示：我可以帮你搜索信息、分析数据、计算数学题、分析股票等")
    
    session_id = "interactive"
    
    while True:
        try:
            user_input = input(f"\n💬 你: ").strip()
            
            if user_input.lower() == 'quit':
                print("👋 再见！")
                break
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
    print("🎊 欢迎使用LangGraph智能助手！")
    print("\n选择运行模式:")
    print("1. 演示模式 (自动运行测试案例)")
    print("2. 交互模式 (手动对话)")
    
    mode = input("\n请选择模式 (1 或 2，默认1): ").strip()
    
    if mode == "2":
        interactive_mode()
    else:
        demo_langgraph_agent()