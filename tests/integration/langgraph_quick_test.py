#!/usr/bin/env python3
"""快速测试LangGraph主程序"""

import sys
import os

# 添加src目录到Python路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from agents import LangGraphAgent
from model_config import ModelManager, display_available_models

def quick_demo():
    """快速演示"""
    print("🚀 LangGraph HTTP模型快速测试")
    print("="*50)
    
    model_key = "http-deepseek-r1-huawei"
    
    # 检查模型是否可用
    if not ModelManager.is_model_available(model_key):
        print(f"❌ 模型 {model_key} 不可用")
        return
    
    print(f"🤖 使用模型: {model_key}")
    
    # 创建智能助手
    print("🔧 正在初始化智能助手...")
    try:
        agent = LangGraphAgent(model_key=model_key)
        print("✅ 智能助手初始化完成！")
    except Exception as e:
        print(f"❌ 初始化失败: {e}")
        return
    
    # 简单测试
    test_cases = [
        "你好！请介绍一下自己",
        "请帮我计算 125 + 75",
    ]
    
    for i, message in enumerate(test_cases, 1):
        print(f"\n📋 测试{i}")
        print(f"问题：{message}")
        print("💭 处理中...")
        
        try:
            response = agent.chat(message, f"test_{i}")
            print(f"🤖 回答：{response}")
            
            # 显示推理步骤
            steps = agent.get_reasoning_steps(f"test_{i}")
            if steps:
                print("🧠 推理步骤：")
                for step in steps:
                    print(f"  • {step}")
            
        except Exception as e:
            print(f"❌ 测试失败: {e}")
        
        print("-" * 40)
    
    print("✅ 快速测试完成！")

if __name__ == "__main__":
    quick_demo()