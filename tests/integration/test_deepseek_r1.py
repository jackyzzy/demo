#!/usr/bin/env python3
"""
测试自定义DeepSeek-R1模型在LangGraph中的使用
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from agents import LangGraphAgent

def test_custom_deepseek_r1():
    """测试自定义部署的DeepSeek-R1模型"""
    print("🧪 测试LangGraph中的自定义DeepSeek-R1模型")
    print("=" * 55)
    
    try:
        # 使用自定义DeepSeek-R1模型
        agent = LangGraphAgent("deepseek-r1-huawei")
        print("✅ 自定义DeepSeek-R1 LangGraph Agent初始化成功")
        
        # 测试简单对话
        response = agent.chat("你好，请简单介绍一下你的推理能力")
        print(f"🤖 DeepSeek-R1回答:")
        print(f"   {response}")
        
        # 测试推理能力
        print(f"\n🧠 测试复杂推理...")
        reasoning_query = "一个正方形的面积是36平方米，求它的周长。请详细说明计算过程。"
        response = agent.chat(reasoning_query)
        print(f"🤖 推理计算结果:")
        print(f"   {response}")
        
        return True
        
    except Exception as e:
        print(f"❌ 测试异常: {e}")
        return False

if __name__ == "__main__":
    print("🚀 开始LangGraph中的自定义DeepSeek-R1模型测试")
    print("=" * 65)
    
    success = test_custom_deepseek_r1()
    
    print("\n" + "=" * 65)
    if success:
        print("🎉 LangGraph中的自定义DeepSeek-R1模型测试成功!")
    else:
        print("❌ LangGraph中的自定义DeepSeek-R1模型测试失败")