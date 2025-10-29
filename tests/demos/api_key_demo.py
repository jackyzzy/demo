#!/usr/bin/env python3
"""
API密钥管理演示脚本
展示模型配置如何自动加载和应用.env中的API密钥
"""

import os
from model_config import (
    ModelManager, ModelConfig, ModelProvider,
    display_available_models, display_environment_setup_guide,
    quick_setup_check, display_detailed_model_info
)

def demo_api_key_loading():
    """演示API密钥加载功能"""
    print("🔑 API密钥管理演示")
    print("=" * 60)
    
    # 1. 显示环境检查
    print("1️⃣ 环境配置检查:")
    has_models = quick_setup_check()
    print()
    
    # 2. 显示具体的API密钥状态
    print("2️⃣ API密钥详细状态:")
    validation = ModelManager.validate_environment()
    missing_keys = validation['missing_keys']
    
    if missing_keys:
        print("❌ 发现缺失的API密钥:")
        for provider, models in missing_keys.items():
            print(f"  📡 {provider}: {len(models)} 个模型受影响")
    else:
        print("✅ 所有需要的API密钥都已正确配置")
    print()
    
    # 3. 演示模型配置对象的API密钥获取
    print("3️⃣ 模型配置API密钥获取演示:")
    
    # 选择几个代表性的模型进行演示
    demo_models = ['gpt-4o', 'claude-3.5-sonnet', 'llama3-70b', 'llama2']
    
    for model_key in demo_models:
        config = ModelManager.get_model_config(model_key)
        if config:
            api_key = config.api_key
            api_key_display = (api_key[:8] + "..." + api_key[-4:]) if api_key else "None"
            
            print(f"  🤖 {model_key}:")
            print(f"    环境变量: {config.api_key_env}")
            print(f"    API密钥: {api_key_display}")
            print(f"    可用性: {'✅' if config.is_available else '❌'}")
    print()
    
    # 4. 演示动态重载功能
    print("4️⃣ 动态重载演示:")
    print("  如果您在程序运行时更新了.env文件，可以使用以下方法重新加载:")
    print("  ModelManager.reload_all_api_keys()")
    
    # 获取一个模型配置进行演示
    config = ModelManager.get_model_config('gpt-4o')
    if config:
        print(f"  重载前API密钥状态: {'✅' if config.api_key else '❌'}")
        config.reload_api_key()
        print(f"  重载后API密钥状态: {'✅' if config.api_key else '❌'}")
    print()

def demo_model_factory_integration():
    """演示模型工厂如何使用API密钥"""
    print("5️⃣ 模型工厂集成演示:")
    
    available_models = ModelManager.get_models_by_availability()
    if not available_models:
        print("  ❌ 没有可用模型，无法进行工厂演示")
        return
    
    # 选择第一个可用模型进行演示
    model_key = list(available_models.keys())[0]
    config = available_models[model_key]
    
    print(f"  📦 使用模型: {model_key}")
    print(f"  🏷️  模型名称: {config.name}")
    print(f"  🏢 提供商: {config.provider.value}")
    print(f"  🔑 API密钥: {'✅ 已自动加载' if config.api_key else '❌ 未配置'}")
    
    print("  💡 模型工厂现在可以直接使用 config.api_key 获取密钥")
    print("     无需手动调用 os.getenv() 或处理环境变量")
    print()

def demo_custom_model_addition():
    """演示添加自定义模型"""
    print("6️⃣ 自定义模型添加演示:")
    
    # 创建一个自定义模型配置
    custom_config = ModelConfig(
        name="Custom Test Model",
        provider=ModelProvider.OPENAI,  # 使用OpenAI接口
        model_id="gpt-3.5-turbo",
        api_key_env="OPENAI_API_KEY",  # 复用现有的API密钥
        description="自定义测试模型配置"
    )
    
    # 添加到管理器
    ModelManager.add_custom_model("custom-test", custom_config)
    
    print("  ➕ 添加自定义模型: custom-test")
    print(f"  🔑 API密钥状态: {'✅' if custom_config.api_key else '❌'}")
    print(f"  📊 可用性: {'✅' if custom_config.is_available else '❌'}")
    print()

def main():
    """主演示函数"""
    print("🚀 多模型系统API密钥管理演示")
    print("=" * 80)
    print()
    
    # 演示各个功能
    demo_api_key_loading()
    demo_model_factory_integration()
    demo_custom_model_addition()
    
    # 最终状态展示
    print("7️⃣ 完整的模型状态:")
    display_available_models()
    
    print("\n" + "=" * 80)
    print("📋 总结:")
    print("✅ API密钥自动从.env文件加载到模型配置")
    print("✅ 模型工厂可以直接使用配置中的API密钥")
    print("✅ 支持运行时重新加载API密钥")
    print("✅ 提供详细的配置状态和错误指导")
    print("✅ 支持自定义模型配置")

if __name__ == "__main__":
    main()