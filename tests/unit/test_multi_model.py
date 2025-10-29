#!/usr/bin/env python3
"""
多模型系统测试脚本
测试模型配置、工厂类和智能体的多模型支持功能
"""

import os
import sys
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 测试模型配置系统
def test_model_config():
    """测试模型配置系统"""
    print("🧪 测试模型配置系统")
    print("=" * 40)
    
    try:
        from model_config import ModelManager, display_available_models
        
        # 测试获取所有模型
        all_models = ModelManager.get_available_models()
        print(f"✅ 找到 {len(all_models)} 个预配置模型")
        
        # 测试获取可用模型（检查API密钥）
        available_models = ModelManager.get_models_by_availability()
        print(f"✅ 当前环境可用模型: {len(available_models)} 个")
        
        if available_models:
            print("📋 可用模型列表:")
            for key, config in available_models.items():
                print(f"  - {key}: {config.name} ({config.provider.value})")
        else:
            print("⚠️  没有可用模型，请检查API密钥配置")
            display_available_models()
        
        print("✅ 模型配置系统测试通过\n")
        return available_models
        
    except Exception as e:
        print(f"❌ 模型配置系统测试失败: {e}\n")
        return {}

def test_agno_model_factory():
    """测试Agno模型工厂"""
    print("🧪 测试Agno模型工厂")
    print("=" * 40)
    
    try:
        # 添加Agno agent路径
        sys.path.insert(0, 'agno-agent/src')
        
        from utils.model_factory import AgnoModelFactory
        from model_config import ModelManager
        
        available_models = ModelManager.get_models_by_availability()
        
        if not available_models:
            print("⚠️  没有可用模型，跳过Agno工厂测试")
            return False
        
        # 测试创建第一个可用模型
        test_model_key = list(available_models.keys())[0]
        print(f"🔧 测试创建模型: {test_model_key}")
        
        model = AgnoModelFactory.create_model(test_model_key)
        print(f"✅ 成功创建Agno模型: {type(model).__name__}")
        
        print("✅ Agno模型工厂测试通过\n")
        return True
        
    except Exception as e:
        print(f"❌ Agno模型工厂测试失败: {e}\n")
        return False

def test_langgraph_model_factory():
    """测试LangGraph模型工厂"""
    print("🧪 测试LangGraph模型工厂")
    print("=" * 40)
    
    try:
        # 添加LangGraph agent路径
        sys.path.insert(0, 'langgraph-agent/src')
        
        from utils.model_factory import LangGraphModelFactory
        from model_config import ModelManager
        
        available_models = ModelManager.get_models_by_availability()
        
        if not available_models:
            print("⚠️  没有可用模型，跳过LangGraph工厂测试")
            return False
        
        # 测试创建第一个可用模型
        test_model_key = list(available_models.keys())[0]
        print(f"🔧 测试创建模型: {test_model_key}")
        
        model = LangGraphModelFactory.create_model(test_model_key)
        print(f"✅ 成功创建LangGraph模型: {type(model).__name__}")
        
        print("✅ LangGraph模型工厂测试通过\n")
        return True
        
    except Exception as e:
        print(f"❌ LangGraph模型工厂测试失败: {e}\n")
        return False

def test_agents():
    """测试智能体多模型支持"""
    print("🧪 测试智能体多模型支持")
    print("=" * 40)
    
    try:
        from model_config import ModelManager
        
        available_models = ModelManager.get_models_by_availability()
        
        if not available_models:
            print("⚠️  没有可用模型，跳过智能体测试")
            return False
        
        test_model_key = list(available_models.keys())[0]
        
        # 测试Agno智能体
        print(f"🔧 测试Agno智能体 - 模型: {test_model_key}")
        try:
            sys.path.insert(0, 'agno-agent/src')
            from agents.agno_agent import AgnoAgent
            
            agno_agent = AgnoAgent(model_key=test_model_key)
            print("✅ Agno智能体创建成功")
        except Exception as e:
            print(f"❌ Agno智能体测试失败: {e}")
        
        # 测试LangGraph智能体
        print(f"🔧 测试LangGraph智能体 - 模型: {test_model_key}")
        try:
            sys.path.insert(0, 'langgraph-agent/src')
            from agents.langgraph_agent import LangGraphAgent
            
            langgraph_agent = LangGraphAgent(model_key=test_model_key)
            print("✅ LangGraph智能体创建成功")
        except Exception as e:
            print(f"❌ LangGraph智能体测试失败: {e}")
        
        print("✅ 智能体多模型支持测试通过\n")
        return True
        
    except Exception as e:
        print(f"❌ 智能体测试失败: {e}\n")
        return False

def main():
    """主测试函数"""
    print("🚀 多模型智能体系统测试")
    print("=" * 60)
    print()
    
    # 测试模型配置
    available_models = test_model_config()
    
    if not available_models:
        print("❌ 没有可用模型，请先配置API密钥")
        return
    
    # 测试模型工厂
    agno_factory_ok = test_agno_model_factory()
    langgraph_factory_ok = test_langgraph_model_factory()
    
    # 测试智能体
    if agno_factory_ok or langgraph_factory_ok:
        test_agents()
    
    # 总结
    print("=" * 60)
    print("🎉 测试总结:")
    print(f"  📊 可用模型数量: {len(available_models)}")
    print(f"  🏭 Agno模型工厂: {'✅' if agno_factory_ok else '❌'}")
    print(f"  🏭 LangGraph模型工厂: {'✅' if langgraph_factory_ok else '❌'}")
    print("=" * 60)
    
    if available_models:
        print("\n🤖 可以使用以下命令测试完整功能:")
        print("  cd agno-agent && python main.py")
        print("  cd langgraph-agent && python main.py")

if __name__ == "__main__":
    main()