#!/usr/bin/env python3
"""
框架对比测试脚本
用于快速测试LangGraph和Agno两个框架的基本功能
"""

import sys
import os
from pathlib import Path

def test_langgraph():
    """测试LangGraph框架"""
    print("\n" + "="*50)
    print("🧪 测试 LangGraph Agent")
    print("="*50)
    
    try:
        # 添加路径
        langgraph_path = Path(__file__).parent / "langgraph-agent" / "src"
        sys.path.insert(0, str(langgraph_path))
        
        from agents import LangGraphAgent
        
        agent = LangGraphAgent()
        
        # 简单测试
        response = agent.chat("你好，请介绍一下你的能力")
        print(f"✅ LangGraph基本测试通过")
        print(f"回答: {response[:100]}...")
        
        return True
        
    except Exception as e:
        print(f"❌ LangGraph测试失败: {str(e)}")
        return False

def test_agno():
    """测试Agno框架"""
    print("\n" + "="*50)
    print("🧪 测试 Agno Agent")
    print("="*50)
    
    try:
        # 添加路径
        agno_path = Path(__file__).parent / "agno-agent" / "src"
        sys.path.insert(0, str(agno_path))
        
        from agents import AgnoAgent
        
        agent = AgnoAgent()
        
        # 简单测试
        response = agent.chat("你好，请介绍一下你的能力")
        print(f"✅ Agno基本测试通过")
        print(f"回答: {response[:100]}...")
        
        return True
        
    except Exception as e:
        print(f"❌ Agno测试失败: {str(e)}")
        return False

def main():
    """主测试函数"""
    print("🚀 AI Agent框架对比测试")
    
    # 检查环境变量文件
    env_file = Path(__file__).parent / ".env"
    if not env_file.exists():
        print("⚠️  .env文件不存在，请先配置API密钥")
        print("参考.env.template文件配置")
        return
    
    # 测试结果
    results = {}
    
    # 测试LangGraph
    results['langgraph'] = test_langgraph()
    
    # 清理路径
    sys.path = [p for p in sys.path if 'langgraph-agent' not in p]
    
    # 测试Agno
    results['agno'] = test_agno()
    
    # 输出结果
    print("\n" + "="*50)
    print("📊 测试结果总结")
    print("="*50)
    
    for framework, success in results.items():
        status = "✅ 通过" if success else "❌ 失败"
        print(f"{framework.capitalize()}: {status}")
    
    if all(results.values()):
        print("\n🎉 所有框架测试通过！")
    else:
        print("\n⚠️  部分框架测试失败，请检查配置")

if __name__ == "__main__":
    main()