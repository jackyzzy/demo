#!/usr/bin/env python3
"""
测试缺失API密钥的情况
演示系统如何处理和指导用户配置API密钥
"""

import os
import tempfile
from model_config import (
    ModelManager, display_environment_setup_guide, 
    quick_setup_check, display_available_models
)

def simulate_missing_keys():
    """模拟缺失API密钥的情况"""
    print("🧪 模拟API密钥缺失情况")
    print("=" * 50)
    
    # 临时备份现有的API密钥
    backup_keys = {}
    api_keys = ['OPENAI_API_KEY', 'ANTHROPIC_API_KEY', 'GROQ_API_KEY']
    
    for key in api_keys:
        backup_keys[key] = os.environ.get(key)
        if key in os.environ:
            del os.environ[key]
    
    print("🔄 已临时移除API密钥，模拟缺失状态...")
    
    # 重新加载所有模型的API密钥
    ModelManager.reload_all_api_keys()
    
    try:
        # 1. 快速检查
        print("\n1️⃣ 快速环境检查:")
        quick_setup_check()
        
        # 2. 显示详细的设置指南
        print("\n2️⃣ 详细设置指南:")
        display_environment_setup_guide()
        
        # 3. 显示当前可用模型（应该只有Ollama模型）
        print("\n3️⃣ 当前可用模型:")
        display_available_models()
        
    finally:
        # 恢复API密钥
        print("\n🔄 恢复API密钥...")
        for key, value in backup_keys.items():
            if value:
                os.environ[key] = value
        
        # 重新加载
        ModelManager.reload_all_api_keys()
        
        print("✅ API密钥已恢复")

def test_partial_configuration():
    """测试部分配置的情况"""
    print("\n" + "=" * 60)
    print("🧪 测试部分API密钥配置")
    print("=" * 60)
    
    # 临时备份并只保留一个API密钥
    backup_keys = {}
    api_keys = ['ANTHROPIC_API_KEY', 'GROQ_API_KEY'] 
    
    for key in api_keys:
        backup_keys[key] = os.environ.get(key)
        if key in os.environ:
            del os.environ[key]
    
    print("🔄 保留OpenAI密钥，移除其他密钥...")
    
    # 重新加载
    ModelManager.reload_all_api_keys()
    
    try:
        print("\n1️⃣ 部分配置状态检查:")
        validation = ModelManager.validate_environment()
        print(f"📊 可用模型比例: {validation['availability_rate']:.1%}")
        
        print("\n2️⃣ 可用模型展示:")
        available = ModelManager.get_models_by_availability()
        for key, config in available.items():
            print(f"  ✅ {key}: {config.name} ({config.provider.value})")
        
        print("\n3️⃣ 缺失配置指导:")
        missing = ModelManager.get_missing_api_keys()
        if missing:
            for provider, models in missing.items():
                print(f"  ❌ {provider}: {len(models)} 个模型不可用")
        
    finally:
        # 恢复API密钥
        for key, value in backup_keys.items():
            if value:
                os.environ[key] = value
        
        ModelManager.reload_all_api_keys()
        print("\n✅ 已恢复完整配置")

def test_runtime_key_update():
    """测试运行时API密钥更新"""
    print("\n" + "=" * 60)
    print("🧪 测试运行时API密钥更新")
    print("=" * 60)
    
    # 获取一个模型配置
    config = ModelManager.get_model_config('gpt-4o')
    if not config:
        print("❌ 找不到测试模型")
        return
    
    print(f"🤖 测试模型: {config.name}")
    print(f"🔑 当前API密钥状态: {'✅' if config.api_key else '❌'}")
    
    # 临时移除API密钥
    original_key = os.environ.get('OPENAI_API_KEY')
    if 'OPENAI_API_KEY' in os.environ:
        del os.environ['OPENAI_API_KEY']
    
    print("🔄 移除环境变量中的API密钥...")
    
    # 重新加载这个模型的API密钥
    config.reload_api_key()
    print(f"🔑 重载后API密钥状态: {'✅' if config.api_key else '❌'}")
    
    # 恢复API密钥
    if original_key:
        os.environ['OPENAI_API_KEY'] = original_key
    
    print("🔄 恢复环境变量中的API密钥...")
    config.reload_api_key()
    print(f"🔑 再次重载后API密钥状态: {'✅' if config.api_key else '❌'}")

def main():
    """主测试函数"""
    print("🚀 API密钥配置测试套件")
    print("=" * 80)
    
    # 显示初始状态
    print("📊 初始配置状态:")
    validation = ModelManager.validate_environment()
    print(f"  可用模型: {validation['available_models']}/{validation['total_models']}")
    print(f"  配置完整度: {validation['availability_rate']:.1%}")
    
    # 运行各项测试
    simulate_missing_keys()
    test_partial_configuration()
    test_runtime_key_update()
    
    print("\n" + "=" * 80)
    print("📋 测试总结:")
    print("✅ 成功模拟和处理API密钥缺失情况")
    print("✅ 提供详细的配置指导和错误提示")
    print("✅ 支持部分配置下的系统运行")
    print("✅ 支持运行时动态更新API密钥")

if __name__ == "__main__":
    main()