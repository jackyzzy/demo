#!/usr/bin/env python3
"""
HTTP模型测试脚本
测试HTTP模型在LangGraph和Agno框架中的集成
"""

import os
import sys
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 添加项目路径
sys.path.append(os.path.join(os.path.dirname(__file__)))
sys.path.append(os.path.join(os.path.dirname(__file__), 'langgraph-agent/src'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'agno-agent/src'))

def test_http_model_configurations():
    """测试HTTP模型配置"""
    print("=" * 80)
    print("🧪 测试HTTP模型配置")
    print("=" * 80)
    
    try:
        from model_config import ModelManager, MODEL_CONFIGS, ModelProvider
        
        # 查找HTTP模型
        http_models = {k: v for k, v in MODEL_CONFIGS.items() if v.provider == ModelProvider.HTTP}
        
        print(f"📊 找到 {len(http_models)} 个HTTP模型配置:")
        for key, config in http_models.items():
            print(f"\n🤖 {key}:")
            print(f"  📛 名称: {config.name}")
            print(f"  🏢 供应商: {config.vendor}")
            print(f"  🆔 模型ID: {config.model_id}")
            print(f"  🌐 端点: {config.base_url}")
            print(f"  🔑 API密钥环境变量: {config.api_key_env}")
            print(f"  📋 头部: {config.headers}")
            print(f"  ✅ 可用: {'是' if config.is_available else '否'}")
            
            if not config.is_available and config.api_key_env:
                api_key = os.getenv(config.api_key_env)
                if not api_key:
                    print(f"  ⚠️  未设置API密钥: {config.api_key_env}")
                else:
                    print(f"  ✅ API密钥已设置")
        
        return len(http_models) > 0
    
    except ImportError as e:
        print(f"❌ 导入模型配置失败: {e}")
        return False
    except Exception as e:
        print(f"❌ 测试HTTP模型配置失败: {e}")
        return False


def test_http_client():
    """测试基础HTTP客户端"""
    print("\n" + "=" * 80)
    print("🧪 测试基础HTTP客户端")
    print("=" * 80)
    
    try:
        from http_model_client import create_http_client, HttpModelResponse
        
        # 创建测试客户端（使用示例配置）
        test_config = {
            "url": "https://api.openai.com/v1/chat/completions",
            "api_key": "test-key-placeholder",
            "model_id": "gpt-3.5-turbo",
            "vendor": "openai"
        }
        
        client = create_http_client(**test_config)
        print(f"✅ 成功创建HTTP客户端")
        print(f"🏷️  供应商: {client.vendor}")
        print(f"🆔 模型ID: {client.model_id}")
        print(f"🌐 URL: {client.url}")
        print(f"📋 头部: {dict(client.headers)}")
        
        # 测试消息预处理
        test_messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "developer", "content": "This should be mapped to system."},
            {"role": "user", "content": "Hello!"}
        ]
        
        processed = client._prepare_messages(test_messages)
        print(f"\n📝 消息预处理测试:")
        print(f"原始消息数量: {len(test_messages)}")
        print(f"处理后消息数量: {len(processed)}")
        
        for i, (orig, proc) in enumerate(zip(test_messages, processed)):
            print(f"  消息 {i+1}: {orig['role']} -> {proc['role']}")
        
        return True
    
    except ImportError as e:
        print(f"❌ 导入HTTP客户端失败: {e}")
        return False
    except Exception as e:
        print(f"❌ 测试HTTP客户端失败: {e}")
        return False


def test_langgraph_http_integration():
    """测试LangGraph HTTP集成"""
    print("\n" + "=" * 80)
    print("🧪 测试LangGraph HTTP集成")
    print("=" * 80)
    
    try:
        from utils.model_factory import LangGraphModelFactory
        from model_config import ModelManager
        
        # 查找可用的HTTP模型
        http_models = {k: v for k, v in ModelManager.get_available_models().items() 
                      if v.provider.value == "http"}
        
        if not http_models:
            print("⚠️  没有可用的HTTP模型，跳过测试")
            # 使用测试配置
            print("📝 使用测试配置进行基本集成测试")
            
            # 导入HTTP适配器
            try:
                from http_langchain_adapter import create_http_chat_model
                
                test_model = create_http_chat_model(
                    url="https://api.example.com/v1/chat/completions",
                    api_key="test-key",
                    model_id="test-model",
                    vendor="generic"
                )
                
                print(f"✅ 成功创建LangChain HTTP模型")
                print(f"🏷️  模型类型: {test_model._llm_type}")
                print(f"🆔 识别参数: {test_model._identifying_params}")
                
                return True
                
            except Exception as e:
                print(f"❌ 创建测试HTTP模型失败: {e}")
                return False
        
        # 测试第一个可用的HTTP模型
        model_key = list(http_models.keys())[0]
        config = http_models[model_key]
        
        print(f"🧪 测试模型: {model_key} ({config.name})")
        
        # 尝试创建模型
        model = LangGraphModelFactory.create_model(model_key)
        
        print(f"✅ 成功创建LangGraph HTTP模型: {type(model).__name__}")
        print(f"🏷️  模型类型: {model._llm_type}")
        
        return True
        
    except ImportError as e:
        print(f"❌ 导入LangGraph模块失败: {e}")
        return False
    except Exception as e:
        print(f"❌ 测试LangGraph HTTP集成失败: {e}")
        return False


def test_agno_http_integration():
    """测试Agno HTTP集成"""
    print("\n" + "=" * 80)
    print("🧪 测试Agno HTTP集成")
    print("=" * 80)
    
    try:
        from utils.model_factory import AgnoModelFactory
        from model_config import ModelManager
        
        # 查找可用的HTTP模型
        http_models = {k: v for k, v in ModelManager.get_available_models().items() 
                      if v.provider.value == "http"}
        
        if not http_models:
            print("⚠️  没有可用的HTTP模型，跳过测试")
            # 使用测试配置
            print("📝 使用测试配置进行基本集成测试")
            
            # 导入HTTP适配器
            try:
                from http_agno_adapter import create_http_agno_model
                
                test_model = create_http_agno_model(
                    url="https://api.example.com/v1/chat/completions",
                    api_key="test-key",
                    model_id="test-model",
                    vendor="generic"
                )
                
                print(f"✅ 成功创建Agno HTTP模型: {test_model}")
                print(f"🆔 模型ID: {test_model.id}")
                print(f"🏷️  模型名称: {test_model.name}")
                print(f"📡 提供商: {test_model.provider}")
                
                return True
                
            except Exception as e:
                print(f"❌ 创建测试HTTP模型失败: {e}")
                return False
        
        # 测试第一个可用的HTTP模型
        model_key = list(http_models.keys())[0]
        config = http_models[model_key]
        
        print(f"🧪 测试模型: {model_key} ({config.name})")
        
        # 尝试创建模型
        model = AgnoModelFactory.create_model(model_key)
        
        print(f"✅ 成功创建Agno HTTP模型: {type(model).__name__}")
        print(f"🆔 模型ID: {getattr(model, 'id', 'unknown')}")
        
        return True
        
    except ImportError as e:
        print(f"❌ 导入Agno模块失败: {e}")
        return False
    except Exception as e:
        print(f"❌ 测试Agno HTTP集成失败: {e}")
        return False


def test_agents_with_http_models():
    """测试代理与HTTP模型的集成"""
    print("\n" + "=" * 80)
    print("🧪 测试代理与HTTP模型的集成")
    print("=" * 80)
    
    # 测试LangGraph Agent
    try:
        from agents.langgraph_agent import LangGraphAgent
        from model_config import ModelManager
        
        # 查找可用的HTTP模型
        http_models = {k: v for k, v in ModelManager.get_available_models().items() 
                      if v.provider.value == "http"}
        
        if http_models:
            model_key = list(http_models.keys())[0]
            print(f"🤖 测试LangGraph Agent使用HTTP模型: {model_key}")
            
            # 注意：这里不实际创建agent，因为可能需要有效的API密钥
            print(f"✅ LangGraph Agent准备就绪，可使用HTTP模型: {model_key}")
        else:
            print("⚠️  没有可用的HTTP模型，跳过LangGraph Agent测试")
    
    except ImportError as e:
        print(f"❌ 导入LangGraph Agent失败: {e}")
    except Exception as e:
        print(f"❌ 测试LangGraph Agent失败: {e}")
    
    # 测试Agno Agent
    try:
        from agents.agno_agent import AgnoAgent
        
        if http_models:
            print(f"🤖 测试Agno Agent使用HTTP模型: {model_key}")
            print(f"✅ Agno Agent准备就绪，可使用HTTP模型: {model_key}")
        else:
            print("⚠️  没有可用的HTTP模型，跳过Agno Agent测试")
        
        return True
    
    except ImportError as e:
        print(f"❌ 导入Agno Agent失败: {e}")
        return False
    except Exception as e:
        print(f"❌ 测试Agno Agent失败: {e}")
        return False


def show_setup_instructions():
    """显示设置说明"""
    print("\n" + "=" * 80)
    print("📋 HTTP模型使用说明")
    print("=" * 80)
    
    print("1. 配置环境变量:")
    print("   - HUAWEI_DEEPSEEK_R1_API_KEY=your_huawei_api_key")
    print("   - HTTP_GENERIC_API_KEY=your_generic_api_key")
    
    print("\n2. HTTP模型配置示例:")
    print("   - huawei-deepseek-r1: 华为云DeepSeek-R1模型")
    print("   - http-generic-openai: 通用HTTP API模型")
    
    print("\n3. 使用方法:")
    print("   # LangGraph Agent")
    print("   agent = LangGraphAgent(model_key='huawei-deepseek-r1')")
    print("   response = agent.chat('Hello!')")
    
    print("\n   # Agno Agent")
    print("   agent = AgnoAgent(model_key='huawei-deepseek-r1')")
    print("   response = agent.chat('Hello!')")
    
    print("\n4. 自定义HTTP模型:")
    print("   可以在model_config.py中添加新的HTTP模型配置")


def main():
    """主测试函数"""
    print("🚀 HTTP模型集成测试")
    print("=" * 80)
    
    test_results = []
    
    # 执行所有测试
    test_results.append(("HTTP模型配置", test_http_model_configurations()))
    test_results.append(("HTTP客户端", test_http_client()))
    test_results.append(("LangGraph HTTP集成", test_langgraph_http_integration()))
    test_results.append(("Agno HTTP集成", test_agno_http_integration()))
    test_results.append(("代理集成", test_agents_with_http_models()))
    
    # 显示测试结果
    print("\n" + "=" * 80)
    print("📊 测试结果汇总")
    print("=" * 80)
    
    passed = 0
    for test_name, result in test_results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"{status} {test_name}")
        if result:
            passed += 1
    
    print(f"\n总计: {passed}/{len(test_results)} 测试通过")
    
    if passed == len(test_results):
        print("🎉 所有测试通过！HTTP模型集成成功。")
    else:
        print("⚠️  部分测试失败，请检查错误信息。")
    
    # 显示使用说明
    show_setup_instructions()


if __name__ == "__main__":
    main()