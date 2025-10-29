#!/usr/bin/env python3
"""
HTTP模型演示脚本
展示如何使用华为云DeepSeek-R1模型进行对话
"""

import os
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

def demo_huawei_deepseek_r1():
    """演示华为云DeepSeek-R1模型"""
    print("=" * 80)
    print("🤖 华为云DeepSeek-R1模型演示")
    print("=" * 80)
    
    # 检查API密钥
    api_key = os.getenv("HUAWEI_DEEPSEEK_R1_API_KEY")
    if not api_key:
        print("❌ 未设置API密钥: HUAWEI_DEEPSEEK_R1_API_KEY")
        print("请在.env文件中设置：")
        print("HUAWEI_DEEPSEEK_R1_API_KEY=TvzfCiQput8sQH8VwdQZ9bjYP-GseLLta6MifZxGRDC82awhL_Sb8RYpU6CVlfN0a20V7bytnwb33jQgl0aBnA")
        return False
    
    try:
        # 导入HTTP客户端
        from http_model_client import create_http_client
        
        # 创建华为云客户端
        client = create_http_client(
            url="https://maas-cn-southwest-2.modelarts-maas.com/v1/infers/8a062fd4-7367-4ab4-a936-5eeb8fb821c4/v1/chat/completions",
            api_key=api_key,
            model_id="DeepSeek-R1",
            vendor="huawei"
        )
        
        print(f"✅ 华为云HTTP客户端创建成功")
        print(f"🌐 端点: {client.url}")
        print(f"🤖 模型: {client.model_id}")
        
        # 准备测试消息
        messages = [
            {"role": "system", "content": "你是一个有用的AI助手。"},
            {"role": "user", "content": "你好，请介绍一下你自己。"}
        ]
        
        print(f"\n📝 发送消息:")
        for msg in messages:
            print(f"  {msg['role']}: {msg['content']}")
        
        # 发送请求
        print(f"\n⏳ 正在调用华为云DeepSeek-R1模型...")
        response = client.chat_completion(
            messages=messages,
            temperature=1.0,
            max_tokens=200
        )
        
        print(f"\n✅ 收到响应:")
        print(f"📄 内容: {response.content}")
        if response.usage:
            print(f"📊 使用情况: {response.usage}")
        
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        return False


def demo_langgraph_agent():
    """演示LangGraph Agent使用HTTP模型"""
    print("\n" + "=" * 80)
    print("🤖 LangGraph Agent HTTP模型演示")
    print("=" * 80)
    
    # 检查API密钥
    api_key = os.getenv("HUAWEI_DEEPSEEK_R1_API_KEY")
    if not api_key:
        print("❌ 未设置API密钥，跳过Agent演示")
        return False
    
    try:
        import sys
        sys.path.append('langgraph-agent/src')
        
        # 激活langgraph环境
        import subprocess
        import os
        
        # 准备测试脚本
        test_script = '''
import sys
sys.path.append('..')
from model_config import ModelManager

# 检查huawei-deepseek-r1模型是否可用
if ModelManager.is_model_available("huawei-deepseek-r1"):
    print("✅ huawei-deepseek-r1 模型可用")
    # 这里可以创建LangGraph Agent
    # agent = LangGraphAgent(model_key="huawei-deepseek-r1")
    # response = agent.chat("你好!")
    print("🤖 LangGraph Agent可以使用HTTP模型")
else:
    print("❌ huawei-deepseek-r1 模型不可用，请检查API密钥")
        '''
        
        # 在langgraph环境中运行 (使用conda环境的Python解释器)
        conda_env_python = os.path.expanduser("~/anaconda3/envs/langgraph-agent/bin/python")
        if os.path.exists(conda_env_python):
            result = subprocess.run(
                [conda_env_python, "-c", test_script],
                capture_output=True,
                text=True,
                cwd=".",
                env=dict(os.environ, PYTHONPATH="/home/zzy/code/comp_agent")
            )
        else:
            # 备用方案：尝试使用conda activate
            result = subprocess.run(
                ["bash", "-c", f"source ~/anaconda3/etc/profile.d/conda.sh && conda activate langgraph-agent && python -c \"{test_script}\""],
                capture_output=True,
                text=True,
                cwd="."
            )
        
        if result.returncode == 0:
            print(result.stdout)
        else:
            print(f"❌ LangGraph Agent测试失败: {result.stderr}")
        
        return result.returncode == 0
        
    except Exception as e:
        print(f"❌ LangGraph Agent演示失败: {e}")
        return False


def demo_agno_agent():
    """演示Agno Agent使用HTTP模型"""
    print("\n" + "=" * 80)
    print("🤖 Agno Agent HTTP模型演示")
    print("=" * 80)
    
    # 检查API密钥
    api_key = os.getenv("HUAWEI_DEEPSEEK_R1_API_KEY")
    if not api_key:
        print("❌ 未设置API密钥，跳过Agent演示")
        return False
    
    try:
        # 准备测试脚本
        test_script = '''
import sys
sys.path.append('/home/zzy/code/comp_agent')
sys.path.append('/home/zzy/code/comp_agent/agno-agent/src')
from model_config import ModelManager

# 检查huawei-deepseek-r1模型是否可用
if ModelManager.is_model_available("huawei-deepseek-r1"):
    print("✅ huawei-deepseek-r1 模型可用")
    # 这里可以创建Agno Agent
    # agent = AgnoAgent(model_key="huawei-deepseek-r1")
    # response = agent.chat("你好!")
    print("🤖 Agno Agent可以使用HTTP模型")
else:
    print("❌ huawei-deepseek-r1 模型不可用，请检查API密钥")
        '''
        
        # 在agno环境中运行 (使用conda环境的Python解释器)
        import subprocess
        conda_env_python = os.path.expanduser("~/anaconda3/envs/agno-agent/bin/python")
        if os.path.exists(conda_env_python):
            result = subprocess.run(
                [conda_env_python, "-c", test_script],
                capture_output=True,
                text=True,
                cwd=".",
                env=dict(os.environ, PYTHONPATH="/home/zzy/code/comp_agent")
            )
        else:
            # 备用方案：尝试使用conda activate
            result = subprocess.run(
                ["bash", "-c", f"source ~/anaconda3/etc/profile.d/conda.sh && conda activate agno-agent && PYTHONPATH=/home/zzy/code/comp_agent python -c \"{test_script}\""],
                capture_output=True,
                text=True,
                cwd="."
            )
        
        if result.returncode == 0:
            print(result.stdout)
        else:
            print(f"❌ Agno Agent测试失败: {result.stderr}")
        
        return result.returncode == 0
        
    except Exception as e:
        print(f"❌ Agno Agent演示失败: {e}")
        return False


def main():
    """主演示函数"""
    print("🚀 HTTP模型集成演示")
    print("=" * 80)
    print("本演示展示如何在LangGraph和Agno框架中使用HTTP方式访问模型")
    print("示例使用华为云部署的DeepSeek-R1模型")
    
    results = []
    
    # 1. 基础HTTP客户端演示
    results.append(("华为云HTTP客户端", demo_huawei_deepseek_r1()))
    
    # 2. LangGraph Agent演示
    results.append(("LangGraph Agent", demo_langgraph_agent()))
    
    # 3. Agno Agent演示
    results.append(("Agno Agent", demo_agno_agent()))
    
    # 显示演示结果
    print("\n" + "=" * 80)
    print("📊 演示结果汇总")
    print("=" * 80)
    
    passed = 0
    for demo_name, result in results:
        status = "✅ 成功" if result else "❌ 失败"
        print(f"{status} {demo_name}")
        if result:
            passed += 1
    
    print(f"\n总计: {passed}/{len(results)} 个演示成功")
    
    # 显示使用说明
    print("\n" + "=" * 80)
    print("📋 HTTP模型使用说明")
    print("=" * 80)
    
    print("1. 配置API密钥:")
    print("   在.env文件中添加:")
    print("   HUAWEI_DEEPSEEK_R1_API_KEY=your_api_key_here")
    
    print("\n2. LangGraph Agent使用:")
    print("   cd langgraph-agent")
    print("   conda activate langgraph-agent")
    print("   python main.py --model huawei-deepseek-r1")
    
    print("\n3. Agno Agent使用:")
    print("   cd agno-agent")
    print("   conda activate agno-agent")
    print("   python main.py --model huawei-deepseek-r1")
    
    print("\n4. 添加新HTTP模型:")
    print("   在model_config.py的MODEL_CONFIGS中添加新配置")


if __name__ == "__main__":
    main()