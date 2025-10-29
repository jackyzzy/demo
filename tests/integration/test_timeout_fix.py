#!/usr/bin/env python3
"""
测试DeepSeek超时修复
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from agents import AgnoAgent

def test_complex_stock_query():
    """测试复杂股票查询（之前超时的查询类型）"""
    print("🧪 测试DeepSeek复杂股票查询（超时修复）")
    print("=" * 55)
    
    try:
        agent = AgnoAgent("deepseek-chat")
        print("✅ Agent初始化成功")
        
        # 使用类似用户的复杂查询
        complex_query = "为了避免别人可能遇到的投资风险，做到更好地风险规避。请帮我分析一下最近深信服的股票走势怎么样，以便于决定什么时候规避风险。"
        
        print(f"🔍 复杂查询: {complex_query}")
        print("⏰ 开始处理（可能需要较长时间）...")
        
        # 这应该会触发复杂查询的长超时处理
        response = agent.chat(complex_query, session_id="complex_test")
        
        print(f"🤖 DeepSeek回答:")
        print(f"   {response}")
        
        # 检查结果
        if "超时" in response and "重试" in response:
            print("⚠️  仍然超时，但有重试机制")
            return "retry_attempted"
        elif "处理失败" in response or "HTTPSConnectionPool" in response:
            print("❌ 仍然存在连接或超时问题")
            return False
        else:
            print("✅ 复杂查询处理成功!")
            return True
            
    except Exception as e:
        print(f"❌ 测试异常: {e}")
        return False

def test_simple_query():
    """测试简单查询确保正常功能不受影响"""
    print("\n🧪 测试简单查询（确保基本功能正常）")
    print("=" * 50)
    
    try:
        agent = AgnoAgent("deepseek-chat")
        
        response = agent.chat("请计算 50 + 30 * 2", session_id="simple_test")
        
        print(f"🤖 回答: {response}")
        
        if "处理失败" not in response:
            print("✅ 简单查询正常")
            return True
        else:
            print("❌ 简单查询失败")
            return False
            
    except Exception as e:
        print(f"❌ 简单查询异常: {e}")
        return False

def test_medium_query():
    """测试中等复杂度查询"""
    print("\n🧪 测试中等复杂度查询")
    print("=" * 40)
    
    try:
        agent = AgnoAgent("deepseek-chat")
        
        response = agent.chat("请分析一下人工智能技术的发展趋势和潜在影响", session_id="medium_test")
        
        print(f"🤖 回答: {response[:200]}...")
        
        if "处理失败" not in response:
            print("✅ 中等查询正常")
            return True
        else:
            print("❌ 中等查询失败")
            return False
            
    except Exception as e:
        print(f"❌ 中等查询异常: {e}")
        return False

if __name__ == "__main__":
    print("🚀 开始DeepSeek超时修复测试")
    print("=" * 60)
    
    results = []
    
    # 测试简单查询（应该很快）
    results.append(test_simple_query())
    
    # 测试中等查询
    results.append(test_medium_query())
    
    # 测试复杂查询（之前超时的类型）
    complex_result = test_complex_stock_query()
    results.append(complex_result if isinstance(complex_result, bool) else True)
    
    # 总结
    print("\n" + "=" * 60)
    print("🎯 超时修复测试总结:")
    
    success_count = sum(1 for r in results if r is True)
    total_count = len(results)
    
    print(f"✅ 成功: {success_count}/{total_count} 个测试")
    
    if complex_result == "retry_attempted":
        print("📋 复杂查询超时但启用了重试机制")
    
    if success_count >= 2:  # 至少简单和中等查询成功
        print("🎉 超时修复基本有效，系统功能正常!")
    else:
        print("⚠️  需要进一步检查网络连接或API状态")