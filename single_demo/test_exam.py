#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
测试 exam.py 的基本功能
"""

import sys
import os

# 设置环境
sys.path.insert(0, os.path.dirname(__file__))

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

def test_basic_import():
    """测试基本导入"""
    print("=" * 70)
    print("测试 1: 检查所有导入是否正常")
    print("=" * 70)

    try:
        from typing import TypedDict, List, Dict, Any, Literal
        from langchain_openai import ChatOpenAI
        from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, BaseMessage
        from langchain_core.tools import tool
        from langgraph.graph import StateGraph, START, END
        from langgraph.checkpoint.memory import MemorySaver

        print("✅ 所有导入成功！")
        return True
    except Exception as e:
        print(f"❌ 导入失败: {e}")
        return False


def test_model_init():
    """测试模型初始化"""
    print("\n" + "=" * 70)
    print("测试 2: 测试模型初始化")
    print("=" * 70)

    try:
        # 使用 DeepSeek API
        llm = ChatOpenAI(
            model="deepseek-chat",
            api_key=os.getenv("DEEPSEEK_API_KEY", "sk-2ee0851f6c5545f099190c4fb27bf3db"),
            base_url="https://api.deepseek.com",
            temperature=0.1
        )

        print("✅ 模型初始化成功！")

        # 测试简单调用
        print("\n测试简单调用...")
        response = llm.invoke([HumanMessage(content="你好，请回答：1+1等于几？")])
        print(f"✅ 模型响应: {response.content[:100]}...")

        return True
    except Exception as e:
        print(f"❌ 模型初始化或调用失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_calculator_tool():
    """测试计算器工具"""
    print("\n" + "=" * 70)
    print("测试 3: 测试计算器工具")
    print("=" * 70)

    try:
        from langchain_core.tools import tool

        @tool
        def calculator(expression: str) -> str:
            """执行数学计算"""
            try:
                allowed_chars = set('0123456789+-*/() .')
                if not all(c in allowed_chars for c in expression):
                    return f"错误: 表达式包含非法字符"
                result = eval(expression)
                return f"计算结果: {expression} = {result}"
            except Exception as e:
                return f"计算错误: {str(e)}"

        # 测试计算器
        result = calculator.invoke({"expression": "2 + 3 * 4"})
        print(f"✅ 计算器测试: {result}")

        return True
    except Exception as e:
        print(f"❌ 计算器工具测试失败: {e}")
        return False


def test_state_graph():
    """测试 StateGraph 创建"""
    print("\n" + "=" * 70)
    print("测试 4: 测试 StateGraph 创建")
    print("=" * 70)

    try:
        from typing import TypedDict, List, Any
        from langgraph.graph import StateGraph, START, END
        from langchain_core.messages import BaseMessage

        class TestState(TypedDict):
            messages: List[BaseMessage]
            counter: int

        def test_node(state: TestState) -> TestState:
            state["counter"] = state.get("counter", 0) + 1
            return state

        workflow = StateGraph(TestState)
        workflow.add_node("test", test_node)
        workflow.add_edge(START, "test")
        workflow.add_edge("test", END)

        app = workflow.compile()

        print("✅ StateGraph 创建成功！")

        # 测试执行
        result = app.invoke({"messages": [], "counter": 0})
        print(f"✅ StateGraph 执行成功！counter = {result['counter']}")

        return True
    except Exception as e:
        print(f"❌ StateGraph 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("🧪 开始测试 exam.py 的依赖和功能")
    print("=" * 70)

    results = []

    # 运行所有测试
    results.append(("基本导入", test_basic_import()))
    results.append(("模型初始化", test_model_init()))
    results.append(("计算器工具", test_calculator_tool()))
    results.append(("StateGraph", test_state_graph()))

    # 输出总结
    print("\n" + "=" * 70)
    print("📊 测试总结")
    print("=" * 70)

    for name, result in results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"{status} - {name}")

    all_passed = all(r for _, r in results)

    if all_passed:
        print("\n🎉 所有测试通过！exam.py 可以正常运行！")
        print("\n运行方式：")
        print("  conda activate langgraph-agent")
        print("  python exam.py")
    else:
        print("\n⚠️  部分测试失败，请检查环境配置")

    print("=" * 70)
