#!/usr/bin/env python3
"""
快速测试超时修复是否生效
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from agents import AgnoAgent

def quick_test():
    print("🧪 快速测试DeepSeek超时修复")
    print("=" * 40)
    
    try:
        agent = AgnoAgent("deepseek-chat")
        print("✅ Agent初始化成功")
        
        # 测试简单计算
        response = agent.chat("请计算 10 + 20", session_id="quick")
        print(f"🤖 计算回答: {response}")
        
        # 测试是否有超时改进的迹象
        if "处理失败" not in response and "HTTPSConnectionPool" not in response:
            print("✅ 基本功能正常，超时修复可能生效")
            return True
        else:
            print("❌ 仍有问题")
            return False
            
    except Exception as e:
        print(f"❌ 测试异常: {e}")
        return False

if __name__ == "__main__":
    quick_test()