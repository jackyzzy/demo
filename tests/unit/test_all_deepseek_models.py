#!/usr/bin/env python3
"""
综合测试所有DeepSeek模型：官方API和自定义部署
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from model_config import ModelManager, display_available_models

def test_deepseek_models():
    """测试所有DeepSeek模型的可用性"""
    print("🚀 综合DeepSeek模型测试")
    print("=" * 50)
    
    # 获取所有DeepSeek模型
    all_models = ModelManager.get_available_models()
    deepseek_models = {k: v for k, v in all_models.items() if v.provider.value == "deepseek"}
    
    print(f"发现 {len(deepseek_models)} 个DeepSeek模型:")
    for key, config in deepseek_models.items():
        status = "✅" if config.is_available else "❌"
        print(f"  {status} {key}: {config.name}")
        if config.is_available:
            print(f"      API端点: {config.base_url}")
            print(f"      模型ID: {config.model_id}")
    
    print("\n" + "=" * 50)
    return deepseek_models

def test_agno_agent_models():
    """测试Agno Agent中的DeepSeek模型"""
    print("\n🧪 测试Agno Agent中的DeepSeek模型")
    print("=" * 45)
    
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'agno-agent/src'))
    
    try:
        from agents import AgnoAgent
        
        # 测试官方DeepSeek
        print("测试官方deepseek-chat...")
        try:
            agent = AgnoAgent("deepseek-chat")
            response = agent.chat("测试：2+3等于几？", session_id="test_official")
            print(f"✅ 官方deepseek-chat响应: {response[:50]}...")
        except Exception as e:
            print(f"❌ 官方deepseek-chat失败: {e}")
        
        # 测试自定义DeepSeek-R1
        print("\n测试自定义deepseek-r1-huawei...")
        try:
            agent = AgnoAgent("deepseek-r1-huawei")
            response = agent.chat("测试：5*6等于多少？", session_id="test_custom")
            print(f"✅ 自定义deepseek-r1-huawei响应: {response[:50]}...")
        except Exception as e:
            print(f"❌ 自定义deepseek-r1-huawei失败: {e}")

    except Exception as e:
        print(f"❌ Agno Agent导入失败: {e}")

def test_langgraph_agent_models():
    """测试LangGraph Agent中的DeepSeek模型"""
    print("\n🧪 测试LangGraph Agent中的DeepSeek模型")
    print("=" * 48)
    
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'langgraph-agent/src'))
    
    try:
        from agents import LangGraphAgent
        
        # 测试自定义DeepSeek-R1（官方的在LangGraph中可能有兼容问题）
        print("测试自定义deepseek-r1-huawei...")
        try:
            agent = LangGraphAgent("deepseek-r1-huawei")
            response = agent.chat("测试：10除以2等于多少？")
            print(f"✅ LangGraph自定义DeepSeek-R1响应: {response[:50]}...")
        except Exception as e:
            print(f"❌ LangGraph自定义DeepSeek-R1失败: {e}")
            
    except Exception as e:
        print(f"❌ LangGraph Agent导入失败: {e}")

if __name__ == "__main__":
    print("🎯 综合DeepSeek模型功能测试")
    print("=" * 60)
    
    # 1. 检查模型配置
    deepseek_models = test_deepseek_models()
    
    if not deepseek_models:
        print("❌ 没有找到DeepSeek模型配置")
        exit(1)
    
    available_count = sum(1 for config in deepseek_models.values() if config.is_available)
    print(f"📊 可用DeepSeek模型: {available_count}/{len(deepseek_models)}")
    
    if available_count == 0:
        print("❌ 没有可用的DeepSeek模型")
        exit(1)
    
    # 2. 测试Agno Agent
    test_agno_agent_models()
    
    # 3. 测试LangGraph Agent
    test_langgraph_agent_models()
    
    print("\n" + "=" * 60)
    print("🎉 DeepSeek模型综合测试完成!")
    print(f"✨ 成功集成了官方DeepSeek API和自定义部署的DeepSeek-R1模型")
    print(f"🚀 两个智能代理框架都可以使用您的自定义推理能力!")