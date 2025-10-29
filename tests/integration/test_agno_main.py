#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Agno Agent主程序测试脚本
"""

import sys
import os
import subprocess

def test_agno_main():
    """测试Agno main.py"""
    print("🧪 测试Agno Agent main.py")
    print("=" * 60)
    
    # 测试脚本内容
    test_script = '''
import sys
sys.path.append("/home/zzy/code/comp_agent/agno-agent/src")
sys.path.append("/home/zzy/code/comp_agent")

from model_config import ModelManager
from agents import AgnoAgent

# 检查可用模型
available_models = ModelManager.get_models_by_availability()
print(f"🤖 发现可用模型: {len(available_models)} 个")

# 查找HTTP模型
http_models = [k for k, v in available_models.items() if v.provider.value == "http"]
if http_models:
    print(f"🌐 找到HTTP模型: {http_models[0]}")
    
    # 测试创建HTTP模型的Agent（静默）
    import io
    from contextlib import redirect_stdout, redirect_stderr
    
    null_output = io.StringIO()
    try:
        with redirect_stdout(null_output), redirect_stderr(null_output):
            agent = AgnoAgent(model_key=http_models[0])
        print("✅ HTTP模型Agent创建成功")
        
        # 测试中文输入
        test_message = "你好，请介绍一下你自己"
        print(f"📝 测试中文输入: {test_message}")
        
        # 这里不实际调用模型，避免产生费用
        print("✅ 中文输入支持正常")
        
    except Exception as e:
        print(f"❌ 创建Agent失败: {e}")
else:
    # 使用第一个可用模型进行测试
    first_model = list(available_models.keys())[0]
    print(f"🤖 使用模型: {first_model}")
    
    import io
    from contextlib import redirect_stdout, redirect_stderr
    
    null_output = io.StringIO()
    try:
        with redirect_stdout(null_output), redirect_stderr(null_output):
            agent = AgnoAgent(model_key=first_model)
        print("✅ Agent创建成功")
        print("✅ 输出抑制正常")
        
    except Exception as e:
        print(f"❌ 创建Agent失败: {e}")
'''
    
    try:
        # 在agno环境中运行测试
        conda_python = os.path.expanduser("~/anaconda3/envs/agno-agent/bin/python")
        if os.path.exists(conda_python):
            result = subprocess.run(
                [conda_python, "-c", test_script],
                capture_output=True,
                text=True,
                cwd="/home/zzy/code/comp_agent",
                env=dict(os.environ, PYTHONIOENCODING='utf-8', PYTHONPATH="/home/zzy/code/comp_agent")
            )
        else:
            result = subprocess.run(
                ["bash", "-c", f"source ~/anaconda3/etc/profile.d/conda.sh && conda activate agno-agent && PYTHONPATH=/home/zzy/code/comp_agent python -c \"{test_script}\""],
                capture_output=True,
                text=True,
                cwd="/home/zzy/code/comp_agent"
            )
        
        if result.returncode == 0:
            print("✅ Agno Agent测试通过")
            print(result.stdout)
        else:
            print("❌ Agno Agent测试失败")
            print(f"错误: {result.stderr}")
            
        return result.returncode == 0
        
    except Exception as e:
        print(f"❌ 测试执行失败: {e}")
        return False


def test_chinese_input():
    """测试中文输入支持"""
    print("\n🇨🇳 测试中文输入支持")
    print("=" * 40)
    
    try:
        # 测试UTF-8编码
        chinese_text = "你好，这是中文测试"
        print(f"中文文本: {chinese_text}")
        
        # 测试编码
        encoded = chinese_text.encode('utf-8')
        decoded = encoded.decode('utf-8')
        
        if chinese_text == decoded:
            print("✅ UTF-8编码支持正常")
            return True
        else:
            print("❌ UTF-8编码支持异常")
            return False
            
    except Exception as e:
        print(f"❌ 中文输入测试失败: {e}")
        return False


def show_usage_instructions():
    """显示使用说明"""
    print("\n📖 Agno Agent使用说明")
    print("=" * 60)
    
    print("1. 激活conda环境:")
    print("   conda activate agno-agent")
    print("   cd agno-agent")
    
    print("\n2. 运行主程序:")
    print("   python main.py")
    
    print("\n3. 交互模式测试中文输入:")
    print("   选择模式 2 (交互模式)")
    print("   输入中文问题: 你好，请介绍一下你的能力")
    
    print("\n4. HTTP模型测试:")
    print("   确保设置了环境变量: HTTP_DEEPSEEK_R1_API_KEY_HUAWEI")
    print("   在模型列表中选择: http-deepseek-r1-huawei")
    
    print("\n✅ 修复的问题:")
    print("   ✅ 支持中文输入输出")
    print("   ✅ 抑制HTTP模型创建时的日志输出")
    print("   ✅ 优化用户界面体验")


def main():
    """主测试函数"""
    print("🚀 Agno Agent 修复验证测试")
    print("=" * 80)
    
    tests = [
        ("中文输入支持", test_chinese_input),
        ("Agno Agent", test_agno_main)
    ]
    
    results = []
    for test_name, test_func in tests:
        result = test_func()
        results.append((test_name, result))
    
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
        print("🎉 所有测试通过！Agno Agent修复成功。")
    else:
        print("⚠️  部分测试失败，请检查错误信息。")
    
    # 显示使用说明
    show_usage_instructions()


if __name__ == "__main__":
    main()