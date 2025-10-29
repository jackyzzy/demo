#!/usr/bin/env python3
"""
简单测试LangGraph Agent与HTTP模型的集成
"""

import sys
sys.path.append("src")

from agents.langgraph_agent import LangGraphAgent

def test_http_model():
    """测试HTTP模型"""
    print("🧪 测试LangGraph Agent与HTTP模型集成")
    
    try:
        # 创建Agent
        print("\n1. 创建LangGraph Agent...")
        agent = LangGraphAgent(model_key="http-deepseek-r1-huawei")
        print("✅ Agent创建成功")
        
        # 测试简单对话
        print("\n2. 测试简单对话...")
        response1 = agent.chat("你好！请介绍一下自己的能力", "test1")
        print(f"🤖 回答: {response1}")
        
        # 测试数学计算
        print("\n3. 测试数学计算...")
        response2 = agent.chat("请帮我计算 (125 + 75) * 2 - 50", "test2")
        print(f"🤖 回答: {response2}")
        
        # 测试推理步骤
        print("\n4. 测试推理步骤...")
        steps = agent.get_reasoning_steps("test1")
        if steps:
            print("🧠 推理步骤:")
            for step in steps:
                print(f"  • {step}")
        else:
            print("📝 无推理步骤记录")
        
        print("\n✅ 所有测试通过！")
        
    except Exception as e:
        import traceback
        print(f"\n❌ 测试失败: {str(e)}")
        print("🐛 详细错误:")
        traceback.print_exc()

if __name__ == "__main__":
    test_http_model()