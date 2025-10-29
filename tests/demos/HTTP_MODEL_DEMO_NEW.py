#!/usr/bin/env python3
"""
新HTTP模型配置演示脚本
展示基于HTTP_前缀环境变量的动态模型发现和使用
"""

import os
import sys
from pathlib import Path

# 添加项目路径
sys.path.append("/home/zzy/code/comp_agent")

def demo_http_discovery():
    """演示HTTP模型动态发现"""
    print("🔍 HTTP模型动态发现演示")
    print("=" * 60)
    
    try:
        from model_config import HttpModelDiscovery, ModelManager, ModelProvider
        
        print("📋 支持的HTTP模型配置模板:")
        templates = HttpModelDiscovery.get_supported_http_models()
        
        for env_key, template in templates.items():
            print(f"\n🔑 环境变量: {env_key}")
            print(f"   📛 模型名称: {template['name']}")
            print(f"   🏢 供应商: {template['vendor']}")
            print(f"   🆔 模型ID: {template['model_id']}")
            print(f"   🌐 端点: {template['base_url']}")
            
            # 检查是否已配置
            if os.getenv(env_key):
                print(f"   ✅ 状态: 已在环境中配置")
            else:
                print(f"   ❌ 状态: 未配置")
        
        # 显示当前发现的模型
        discovered = HttpModelDiscovery.discover_http_models()
        print(f"\n🎯 当前环境中发现的HTTP模型: {len(discovered)} 个")
        
        if discovered:
            for key, config in discovered.items():
                print(f"   - {key}: {config.name}")
        else:
            print("   (无已配置的HTTP模型)")
        
        return True
        
    except Exception as e:
        print(f"❌ HTTP发现演示失败: {e}")
        return False


def demo_configuration_guide():
    """演示配置指南"""
    print("\n📖 HTTP模型配置指南")
    print("=" * 60)
    
    print("1. 在 .env 文件中添加HTTP模型API密钥:")
    print()
    
    configs = [
        ("华为云DeepSeek-R1", "HTTP_DEEPSEEK_R1_API_KEY_HUAWEI", "your_huawei_deepseek_key"),
        ("阿里云通义千问", "HTTP_QWEN_API_KEY_ALIBABA", "your_alibaba_qwen_key"),
        ("百度云文心一言", "HTTP_ERNIE_API_KEY_BAIDU", "your_baidu_ernie_key"),
        ("OpenAI兼容模型", "HTTP_OPENAI_COMPATIBLE_API_KEY", "your_openai_compatible_key")
    ]
    
    for name, env_key, example_value in configs:
        print(f"   # {name}")
        print(f"   {env_key}={example_value}")
        print()
    
    print("2. 重启应用后，HTTP模型将自动可用")
    print()
    
    print("3. 生成的模型键值:")
    model_keys = [
        "http-deepseek-r1-huawei",
        "http-qwen-alibaba", 
        "http-ernie-baidu",
        "http-openai-compatible"
    ]
    
    for key in model_keys:
        print(f"   - {key}")


def demo_agent_usage():
    """演示代理使用方式"""
    print("\n🤖 代理使用演示")
    print("=" * 60)
    
    print("配置完HTTP模型后，可以在代理中使用:")
    print()
    
    print("LangGraph Agent:")
    print("   conda activate langgraph-agent")
    print("   cd langgraph-agent")
    print("   python main.py")
    print("   # 在模型选择时选择 http-deepseek-r1-huawei 等")
    print()
    
    print("Agno Agent:")
    print("   conda activate agno-agent")
    print("   cd agno-agent")
    print("   python main.py")
    print("   # 在模型选择时选择 http-deepseek-r1-huawei 等")
    print()
    
    print("Python代码中直接使用:")
    print("""
from model_config import ModelManager

# 获取所有可用模型（包括动态发现的HTTP模型）
models = ModelManager.get_available_models()
http_models = {k: v for k, v in models.items() if v.provider.value == 'http'}

print("可用的HTTP模型:")
for key, config in http_models.items():
    print(f"  {key}: {config.name}")
""")


def demo_live_test():
    """演示实时测试（如果配置了API密钥）"""
    print("\n🧪 实时测试演示")
    print("=" * 60)
    
    # 检查是否有配置的HTTP模型
    try:
        from model_config import HttpModelDiscovery
        
        discovered = HttpModelDiscovery.discover_http_models()
        
        if not discovered:
            print("❌ 未发现已配置的HTTP模型")
            print("请先在 .env 文件中配置至少一个HTTP模型API密钥")
            return False
        
        print(f"✅ 发现 {len(discovered)} 个已配置的HTTP模型:")
        
        # 尝试创建HTTP客户端进行测试
        from http_model_client import create_http_client
        
        for key, config in discovered.items():
            if not config.is_available:
                continue  # 跳过未正确配置的模型
                
            print(f"\n🧪 测试模型: {key}")
            try:
                # 创建HTTP客户端
                client = create_http_client(
                    url=config.base_url,
                    api_key=config.api_key,
                    model_id=config.model_id,
                    vendor=config.vendor
                )
                
                print(f"✅ 客户端创建成功")
                print(f"   端点: {client.url}")
                print(f"   模型: {client.model_id}")
                print(f"   供应商: {client.vendor}")
                
                # 注意: 不实际调用API，避免产生费用
                print("   (跳过实际API调用以避免费用)")
                
            except Exception as e:
                print(f"❌ 客户端创建失败: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ 实时测试失败: {e}")
        return False


def demo_add_custom_model():
    """演示如何添加自定义HTTP模型"""
    print("\n➕ 添加自定义HTTP模型演示")
    print("=" * 60)
    
    print("要添加新的HTTP模型提供商，需要:")
    print()
    
    print("1. 在 model_config.py 的 HttpModelDiscovery.HTTP_MODEL_TEMPLATES 中添加:")
    print("""
# 例如添加腾讯云混元模型
"HTTP_HUNYUAN_API_KEY_TENCENT": {
    "name": "腾讯混元 (腾讯云)",
    "model_id": "hunyuan-lite",
    "base_url": "https://hunyuan.tencentcloudapi.com/",
    "vendor": "tencent",
    "temperature": 0.7,
    "description": "腾讯云混元模型"
}
""")
    
    print("2. 在 .env 文件中配置API密钥:")
    print("   HTTP_HUNYUAN_API_KEY_TENCENT=your_tencent_api_key")
    print()
    
    print("3. 重启应用，新模型将自动可用，键值为:")
    print("   http-hunyuan-tencent")
    print()
    
    print("4. 如果需要特殊的认证方式，可以在 http_model_client.py 中")
    print("   的 HttpModelClient.__init__ 方法中添加对应的vendor处理逻辑")


def main():
    """主演示函数"""
    print("🚀 新HTTP模型配置系统演示")
    print("=" * 80)
    print("基于HTTP_前缀环境变量的动态模型发现系统")
    print("=" * 80)
    
    demos = [
        ("HTTP模型发现", demo_http_discovery),
        ("配置指南", demo_configuration_guide),
        ("代理使用", demo_agent_usage), 
        ("实时测试", demo_live_test),
        ("添加自定义模型", demo_add_custom_model)
    ]
    
    results = []
    
    for demo_name, demo_func in demos:
        try:
            print()
            result = demo_func()
            if result is not None:
                results.append((demo_name, result))
        except Exception as e:
            print(f"❌ {demo_name} 演示失败: {e}")
            results.append((demo_name, False))
    
    # 显示结果
    if results:
        print("\n" + "=" * 80)
        print("📊 演示结果")
        print("=" * 80)
        
        for demo_name, result in results:
            status = "✅ 成功" if result else "❌ 失败"
            print(f"{status} {demo_name}")
    
    print("\n🎯 核心优势:")
    print("✅ 自动发现: 根据环境变量自动发现可用的HTTP模型")
    print("✅ 多供应商: 支持华为云、阿里云、百度云等多个供应商")
    print("✅ 零配置: 设置环境变量后即可使用，无需修改代码")
    print("✅ 扩展性: 易于添加新的模型供应商和端点")
    print("✅ 统一接口: LangGraph和Agno代理统一使用方式")
    
    print("\n💡 下一步:")
    print("1. 在 .env 文件中配置你需要的HTTP模型API密钥")
    print("2. 运行代理程序，选择对应的HTTP模型")
    print("3. 享受多云模型服务的便利！")


if __name__ == "__main__":
    main()