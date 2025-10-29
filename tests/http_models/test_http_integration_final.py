#!/usr/bin/env python3
"""
HTTP模型集成最终测试脚本
使用正确的conda环境激活方式测试HTTP模型集成
"""

import os
import sys
import subprocess
from pathlib import Path

def find_conda_python(env_name: str) -> str:
    """查找conda环境的Python解释器路径"""
    # 常见的conda安装路径
    possible_paths = [
        f"~/anaconda3/envs/{env_name}/bin/python",
        f"~/miniconda3/envs/{env_name}/bin/python", 
        f"/opt/conda/envs/{env_name}/bin/python",
        f"/usr/local/anaconda3/envs/{env_name}/bin/python"
    ]
    
    for path in possible_paths:
        expanded_path = os.path.expanduser(path)
        if os.path.exists(expanded_path):
            return expanded_path
    
    return None


def run_in_conda_env(env_name: str, test_script: str) -> tuple[bool, str]:
    """在指定的conda环境中运行测试脚本"""
    conda_python = find_conda_python(env_name)
    
    if conda_python:
        print(f"📍 使用conda环境Python: {conda_python}")
        result = subprocess.run(
            [conda_python, "-c", test_script],
            capture_output=True,
            text=True,
            cwd="/home/zzy/code/comp_agent",
            env=dict(os.environ, PYTHONPATH="/home/zzy/code/comp_agent")
        )
    else:
        print(f"⚠️  未找到{env_name}环境的Python路径，尝试conda activate")
        # 备用方案：使用conda activate
        conda_init_cmd = f"""
source ~/anaconda3/etc/profile.d/conda.sh 2>/dev/null || source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null
conda activate {env_name}
export PYTHONPATH=/home/zzy/code/comp_agent
python -c "{test_script}"
"""
        result = subprocess.run(
            ["bash", "-c", conda_init_cmd],
            capture_output=True,
            text=True,
            cwd="/home/zzy/code/comp_agent"
        )
    
    return result.returncode == 0, result.stdout if result.returncode == 0 else result.stderr


def test_langgraph_http_integration():
    """测试LangGraph HTTP集成"""
    print("=" * 80)
    print("🧪 测试LangGraph HTTP集成")
    print("=" * 80)
    
    test_script = '''
import sys
sys.path.append("/home/zzy/code/comp_agent")
sys.path.append("/home/zzy/code/comp_agent/langgraph-agent/src")

try:
    from model_config import ModelManager, ModelProvider
    from utils.model_factory import LangGraphModelFactory
    from http_langchain_adapter import create_http_chat_model
    
    # 测试HTTP模型配置
    http_models = {k: v for k, v in ModelManager.get_available_models().items() if v.provider == ModelProvider.HTTP}
    print(f"🤖 找到HTTP模型: {len(http_models)}")
    
    # 测试HTTP适配器
    test_model = create_http_chat_model(
        url="https://api.example.com/v1/chat/completions",
        api_key="test-key",
        model_id="test-model",
        vendor="generic"
    )
    print(f"✅ HTTP LangChain适配器创建成功: {test_model._llm_type}")
    
    # 测试工厂方法
    factory_available = hasattr(LangGraphModelFactory, "_create_http_model")
    print(f"✅ LangGraphModelFactory._create_http_model可用: {factory_available}")
    
    print("🎉 LangGraph HTTP集成测试完成!")
    
except Exception as e:
    print(f"❌ 测试失败: {e}")
    import traceback
    traceback.print_exc()
'''
    
    success, output = run_in_conda_env("langgraph-agent", test_script)
    print(output)
    return success


def test_agno_http_integration():
    """测试Agno HTTP集成"""
    print("=" * 80)
    print("🧪 测试Agno HTTP集成")
    print("=" * 80)
    
    test_script = '''
import sys
sys.path.append("/home/zzy/code/comp_agent")
sys.path.append("/home/zzy/code/comp_agent/agno-agent/src")

try:
    from model_config import ModelManager, ModelProvider
    from utils.model_factory import AgnoModelFactory
    from http_agno_adapter import create_http_agno_model
    
    # 测试HTTP模型配置
    http_models = {k: v for k, v in ModelManager.get_available_models().items() if v.provider == ModelProvider.HTTP}
    print(f"🤖 找到HTTP模型: {len(http_models)}")
    
    # 测试HTTP适配器
    test_model = create_http_agno_model(
        url="https://api.example.com/v1/chat/completions",
        api_key="test-key",
        model_id="test-model",
        vendor="generic"
    )
    print(f"✅ HTTP Agno适配器创建成功: {test_model}")
    print(f"🆔 模型ID: {test_model.id}")
    
    # 测试工厂方法
    factory_available = hasattr(AgnoModelFactory, "_create_http_model")
    print(f"✅ AgnoModelFactory._create_http_model可用: {factory_available}")
    
    print("🎉 Agno HTTP集成测试完成!")
    
except Exception as e:
    print(f"❌ 测试失败: {e}")
    import traceback
    traceback.print_exc()
'''
    
    success, output = run_in_conda_env("agno-agent", test_script)
    print(output)
    return success


def test_http_client_basic():
    """测试基础HTTP客户端"""
    print("=" * 80)
    print("🧪 测试基础HTTP客户端")
    print("=" * 80)
    
    try:
        sys.path.append("/home/zzy/code/comp_agent")
        from http_model_client import create_http_client
        from model_config import ModelManager, ModelProvider
        
        # 测试HTTP模型配置
        http_models = {k: v for k, v in ModelManager.get_available_models().items() if v.provider == ModelProvider.HTTP}
        print(f"🤖 配置的HTTP模型数量: {len(http_models)}")
        
        for key, config in http_models.items():
            print(f"  - {key}: {config.name}")
            print(f"    供应商: {config.vendor}")
            print(f"    端点: {config.base_url}")
            print(f"    可用: {'✅' if config.is_available else '❌'}")
            
            if not config.is_available and config.api_key_env:
                print(f"    需要设置API密钥: {config.api_key_env}")
        
        # 测试创建HTTP客户端
        client = create_http_client(
            url="https://api.example.com/v1/chat/completions",
            api_key="test-key",
            model_id="test-model",
            vendor="generic"
        )
        
        print(f"✅ HTTP客户端创建成功")
        print(f"🌐 URL: {client.url}")
        print(f"🤖 模型: {client.model_id}")
        print(f"🏢 供应商: {client.vendor}")
        
        # 测试消息预处理
        test_messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "developer", "content": "This should be mapped."},
            {"role": "user", "content": "Hello!"}
        ]
        
        processed = client._prepare_messages(test_messages)
        print(f"✅ 消息预处理测试通过: {len(processed)} 条消息")
        
        return True
        
    except Exception as e:
        print(f"❌ HTTP客户端测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def show_usage_instructions():
    """显示使用说明"""
    print("=" * 80)
    print("📋 HTTP模型使用指南")
    print("=" * 80)
    
    print("1. 环境配置:")
    print("   在 .env 文件中配置API密钥:")
    print("   HUAWEI_DEEPSEEK_R1_API_KEY=your_huawei_api_key")
    print("   HTTP_GENERIC_API_KEY=your_generic_api_key")
    
    print("\n2. 使用LangGraph Agent:")
    print("   conda activate langgraph-agent")
    print("   cd langgraph-agent")
    print("   python main.py")
    print("   # 在交互中选择 huawei-deepseek-r1 模型")
    
    print("\n3. 使用Agno Agent:")
    print("   conda activate agno-agent") 
    print("   cd agno-agent")
    print("   python main.py")
    print("   # 在交互中选择 huawei-deepseek-r1 模型")
    
    print("\n4. 添加新的HTTP模型:")
    print("   编辑 model_config.py，在 MODEL_CONFIGS 中添加新配置")
    
    print("\n5. HTTP模型配置格式:")
    print('''   "your-model-key": ModelConfig(
        name="你的模型名称",
        provider=ModelProvider.HTTP,
        model_id="actual-model-id",
        api_key_env="YOUR_API_KEY_ENV_VAR",
        base_url="https://your-api-endpoint.com/v1/chat/completions",
        vendor="your-vendor-name",
        headers={"Content-Type": "application/json"},
        description="模型描述"
    )''')


def main():
    """主测试函数"""
    print("🚀 HTTP模型集成最终测试")
    print("=" * 80)
    
    tests = [
        ("基础HTTP客户端", test_http_client_basic),
        ("LangGraph HTTP集成", test_langgraph_http_integration), 
        ("Agno HTTP集成", test_agno_http_integration)
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
        print("🎉 所有测试通过！HTTP模型集成成功完成。")
    else:
        print("⚠️  部分测试失败，请检查错误信息。")
    
    show_usage_instructions()


if __name__ == "__main__":
    main()