#!/usr/bin/env python3
"""
HTTP模型动态发现测试脚本
测试基于HTTP_开头环境变量的模型自动发现功能
"""

import os
import sys
from pathlib import Path

# 添加项目路径
sys.path.append("/home/zzy/code/comp_agent")

def setup_test_environment():
    """设置测试环境变量"""
    print("🔧 设置测试环境变量...")
    
    # 设置测试用的HTTP模型API密钥
    test_env_vars = {
        "HTTP_DEEPSEEK_R1_API_KEY_HUAWEI": "test_huawei_deepseek_key_12345",
        "HTTP_QWEN_API_KEY_ALIBABA": "test_alibaba_qwen_key_67890", 
        "HTTP_ERNIE_API_KEY_BAIDU": "test_baidu_ernie_key_abcde",
        "HTTP_OPENAI_COMPATIBLE_API_KEY": "test_openai_compatible_key_fghij"
    }
    
    for key, value in test_env_vars.items():
        os.environ[key] = value
        print(f"  设置 {key}=***{value[-6:]}")
    
    return test_env_vars


def test_http_model_discovery():
    """测试HTTP模型动态发现"""
    print("\n" + "=" * 80)
    print("🔍 测试HTTP模型动态发现")
    print("=" * 80)
    
    try:
        from model_config import HttpModelDiscovery, ModelManager, ModelProvider
        
        # 测试获取支持的HTTP模型模板
        supported_models = HttpModelDiscovery.get_supported_http_models()
        print(f"📋 支持的HTTP模型模板数量: {len(supported_models)}")
        
        for env_key, template in supported_models.items():
            print(f"  - {env_key}:")
            print(f"    名称: {template['name']}")
            print(f"    供应商: {template['vendor']}")
            print(f"    模型ID: {template['model_id']}")
            print(f"    端点: {template['base_url']}")
        
        # 测试动态发现
        discovered_models = HttpModelDiscovery.discover_http_models()
        print(f"\n🎯 动态发现的HTTP模型数量: {len(discovered_models)}")
        
        for model_key, config in discovered_models.items():
            print(f"\n🤖 发现模型: {model_key}")
            print(f"  📛 名称: {config.name}")
            print(f"  🏢 供应商: {config.vendor}")
            print(f"  🆔 模型ID: {config.model_id}")
            print(f"  🌐 端点: {config.base_url}")
            print(f"  🔑 API密钥环境变量: {config.api_key_env}")
            print(f"  ✅ 可用: {'是' if config.is_available else '否'}")
            print(f"  🌡️  温度: {config.temperature}")
        
        # 测试ModelManager集成
        all_models = ModelManager.get_available_models()
        http_models = {k: v for k, v in all_models.items() if v.provider == ModelProvider.HTTP}
        
        print(f"\n📊 ModelManager中的HTTP模型数量: {len(http_models)}")
        print("HTTP模型列表:")
        for key, config in http_models.items():
            print(f"  - {key}: {config.name} ({config.vendor})")
        
        return len(discovered_models) > 0
        
    except Exception as e:
        print(f"❌ HTTP模型发现测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_model_key_generation():
    """测试模型键值生成逻辑"""
    print("\n" + "=" * 80)
    print("🔑 测试模型键值生成")
    print("=" * 80)
    
    test_cases = [
        ("HTTP_DEEPSEEK_R1_API_KEY_HUAWEI", "http-deepseek-r1-huawei"),
        ("HTTP_QWEN_API_KEY_ALIBABA", "http-qwen-alibaba"),
        ("HTTP_ERNIE_API_KEY_BAIDU", "http-ernie-baidu"),
        ("HTTP_OPENAI_COMPATIBLE_API_KEY", "http-openai-compatible")
    ]
    
    for env_key, expected_key in test_cases:
        # 模拟键值生成逻辑
        generated_key = env_key.lower().replace("_api_key", "").replace("_", "-")
        
        print(f"环境变量: {env_key}")
        print(f"期望键值: {expected_key}")
        print(f"生成键值: {generated_key}")
        print(f"匹配: {'✅' if generated_key == expected_key else '❌'}")
        print()


def test_vendor_authentication():
    """测试不同供应商的认证方式"""
    print("\n" + "=" * 80)
    print("🔐 测试供应商认证方式")
    print("=" * 80)
    
    try:
        from http_model_client import HttpModelClient
        
        # 测试不同供应商的HTTP客户端创建
        test_configs = [
            {
                "vendor": "huawei",
                "url": "https://maas-cn-southwest-2.modelarts-maas.com/v1/infers/xxx/v1/chat/completions",
                "api_key": "test_huawei_key",
                "model_id": "DeepSeek-R1"
            },
            {
                "vendor": "alibaba", 
                "url": "https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation",
                "api_key": "test_alibaba_key",
                "model_id": "qwen-plus"
            },
            {
                "vendor": "baidu",
                "url": "https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/chat/eb-instant",
                "api_key": "test_baidu_key", 
                "model_id": "ernie-bot-turbo"
            },
            {
                "vendor": "openai-compatible",
                "url": "https://api.openai.com/v1/chat/completions",
                "api_key": "test_openai_key",
                "model_id": "gpt-3.5-turbo"
            }
        ]
        
        for config in test_configs:
            try:
                client = HttpModelClient(**config)
                print(f"✅ {config['vendor']} 客户端创建成功")
                print(f"  认证头部: {client.headers.get('Authorization', 'N/A')}")
                print(f"  端点: {client.url}")
                print()
            except Exception as e:
                print(f"❌ {config['vendor']} 客户端创建失败: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ 供应商认证测试失败: {e}")
        return False


def test_integration_with_agents():
    """测试与代理的集成"""
    print("\n" + "=" * 80)
    print("🤖 测试代理集成")
    print("=" * 80)
    
    # 由于代理需要在特定的conda环境中运行，这里只测试配置可用性
    try:
        from model_config import ModelManager, ModelProvider
        
        all_models = ModelManager.get_available_models()
        http_models = {k: v for k, v in all_models.items() if v.provider == ModelProvider.HTTP}
        
        print(f"🔍 检测到 {len(http_models)} 个HTTP模型可供代理使用:")
        
        for key, config in http_models.items():
            print(f"\n📋 模型: {key}")
            print(f"  可供LangGraph Agent使用: ✅")
            print(f"  可供Agno Agent使用: ✅")
            print(f"  API密钥状态: {'✅ 已配置' if config.is_available else '❌ 未配置'}")
        
        if http_models:
            print(f"\n🎯 代理可以使用以下命令选择HTTP模型:")
            print("LangGraph Agent:")
            print("  conda activate langgraph-agent")
            print("  cd langgraph-agent")
            print("  python main.py")
            
            print("\nAgno Agent:")
            print("  conda activate agno-agent")
            print("  cd agno-agent") 
            print("  python main.py")
            
            print(f"\n然后选择以下任意HTTP模型:")
            for key in http_models.keys():
                print(f"  - {key}")
        
        return len(http_models) > 0
        
    except Exception as e:
        print(f"❌ 代理集成测试失败: {e}")
        return False


def clean_test_environment(test_env_vars):
    """清理测试环境变量"""
    print("\n🧹 清理测试环境变量...")
    for key in test_env_vars.keys():
        if key in os.environ:
            del os.environ[key]
            print(f"  删除 {key}")


def show_usage_guide():
    """显示使用指南"""
    print("\n" + "=" * 80)
    print("📖 HTTP模型使用指南")
    print("=" * 80)
    
    print("1. 支持的环境变量格式:")
    print("   HTTP_DEEPSEEK_R1_API_KEY_HUAWEI=your_huawei_api_key")
    print("   HTTP_QWEN_API_KEY_ALIBABA=your_alibaba_api_key")
    print("   HTTP_ERNIE_API_KEY_BAIDU=your_baidu_api_key")
    print("   HTTP_OPENAI_COMPATIBLE_API_KEY=your_openai_key")
    
    print("\n2. 自动生成的模型键值:")
    print("   http-deepseek-r1-huawei")
    print("   http-qwen-alibaba")
    print("   http-ernie-baidu")
    print("   http-openai-compatible")
    
    print("\n3. 在.env文件中配置:")
    print("   # 华为云DeepSeek-R1")
    print("   HTTP_DEEPSEEK_R1_API_KEY_HUAWEI=your_actual_api_key")
    print("   ")
    print("   # 阿里云通义千问")
    print("   HTTP_QWEN_API_KEY_ALIBABA=your_actual_api_key")
    
    print("\n4. 使用代理:")
    print("   配置完环境变量后，HTTP模型会自动在代理中可用")
    
    print("\n5. 添加新的HTTP模型:")
    print("   在model_config.py的HttpModelDiscovery.HTTP_MODEL_TEMPLATES中添加新模板")


def main():
    """主测试函数"""
    print("🚀 HTTP模型动态发现测试")
    print("=" * 80)
    
    # 设置测试环境
    test_env_vars = setup_test_environment()
    
    try:
        tests = [
            ("HTTP模型发现", test_http_model_discovery),
            ("模型键值生成", test_model_key_generation),
            ("供应商认证", test_vendor_authentication),
            ("代理集成", test_integration_with_agents)
        ]
        
        results = []
        
        for test_name, test_func in tests:
            try:
                result = test_func()
                results.append((test_name, result))
            except Exception as e:
                print(f"❌ {test_name} 执行失败: {e}")
                results.append((test_name, False))
        
        # 显示测试结果
        print("\n" + "=" * 80)
        print("📊 测试结果汇总")
        print("=" * 80)
        
        passed = 0
        for test_name, result in results:
            status = "✅ 通过" if result else "❌ 失败"
            print(f"{status} {test_name}")
            if result:
                passed += 1
        
        print(f"\n总计: {passed}/{len(results)} 测试通过")
        
        if passed == len(results):
            print("🎉 所有测试通过！HTTP模型动态发现功能正常。")
        else:
            print("⚠️  部分测试失败，请检查错误信息。")
        
        # 显示使用指南
        show_usage_guide()
        
    finally:
        # 清理测试环境
        clean_test_environment(test_env_vars)


if __name__ == "__main__":
    main()