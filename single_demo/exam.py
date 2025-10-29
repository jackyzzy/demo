#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
LangGraph Agent Demo - 独立演示程序
使用原生 LangGraph 包实现一个简单的 Agent，包含规划器和多个工作节点
"""

import sys
import os
from dotenv import load_dotenv
from typing import TypedDict, List, Dict, Any, Literal
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, BaseMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

# 设置标准输入输出编码为UTF-8
try:
    sys.stdin.reconfigure(encoding="utf-8")
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except AttributeError:
    os.environ["PYTHONIOENCODING"] = "utf-8"


# ============================================================
# 1. 定义 Agent 状态
# ============================================================


class AgentState(TypedDict):
    """Agent 状态定义"""

    messages: List[BaseMessage]  # 对话消息列表
    task_type: str  # 任务类型: math_calc, math_proof, logic, general
    plan: str  # planner 制定的计划
    current_step: str  # 当前执行步骤
    results: Dict[str, Any]  # 各节点的执行结果
    final_answer: str  # 最终答案


# ============================================================
# 2. 定义工具函数
# ============================================================


@tool
def calculator(expression: str) -> str:
    """执行数学计算

    Args:
        expression: 数学表达式，如 "2 + 3 * 4" 或 "25 * 4 + 10"

    Returns:
        计算结果字符串
    """
    try:
        # 安全检查：只允许数字和基本运算符
        allowed_chars = set("0123456789+-*/() .")
        if not all(c in allowed_chars for c in expression):
            return f"错误: 表达式包含非法字符"

        result = eval(expression)
        return f"计算结果: {expression} = {result}"
    except Exception as e:
        return f"计算错误: {str(e)}"


# ============================================================
# 3. 定义节点函数
# ============================================================

# 全局变量，用于在节点中访问 LLM
llm = None


def planner_node(state: AgentState) -> AgentState:
    """规划器节点：分析用户输入，制定执行计划"""
    print("\n" + "=" * 70)
    print("🎯 [PLANNER] 正在分析任务并制定计划...")
    print("=" * 70)

    messages = state["messages"]
    user_input = messages[-1].content if messages else ""

    # 使用 LLM 分析任务类型
    system_prompt = """你是一个任务规划器。分析用户的问题，判断任务类型并制定简要计划。

任务类型分类：
- math_calc: 需要数学计算的问题（如：计算、求和、求积等）
- math_proof: 需要数学证明的问题（如：证明定理、推导公式等）
- logic: 需要逻辑推理的问题（如：逻辑谜题、推理题等）
- general: 一般性问题（如：解释概念、回答常识等）

请分析用户问题并输出：
1. 任务类型（从上述4种中选择一种）
2. 简要执行计划（1-2句话）

格式如下：
任务类型: [类型]
执行计划: [计划内容]"""

    try:
        response = llm.invoke(
            [
                SystemMessage(content=system_prompt),
                HumanMessage(content=f"用户问题: {user_input}"),
            ]
        )

        plan_text = response.content

        # 解析任务类型
        task_type = "general"
        if "math_calc" in plan_text.lower() or "计算" in plan_text:
            task_type = "math_calc"
        elif "math_proof" in plan_text.lower() or "证明" in plan_text:
            task_type = "math_proof"
        elif "logic" in plan_text.lower() or "逻辑" in plan_text or "推理" in plan_text:
            task_type = "logic"

        state["plan"] = plan_text
        state["task_type"] = task_type
        state["current_step"] = "planner"

        print(f"📋 任务类型: {task_type}")
        print(f"📋 执行计划:\n{plan_text}")
        print("=" * 70)

    except Exception as e:
        print(f"❌ 规划器出错: {e}")
        state["task_type"] = "general"
        state["plan"] = "使用默认流程处理"

    return state


def math_calculator_node(state: AgentState) -> AgentState:
    """数学计算节点：执行数学运算"""
    print("\n" + "=" * 70)
    print("🔢 [CALCULATOR] 执行数学计算...")
    print("=" * 70)

    messages = state["messages"]
    user_input = messages[-1].content if messages else ""

    system_prompt = """你是一个数学计算助手。分析用户的数学问题，提取需要计算的表达式并求解。

如果需要计算，请使用 calculator 工具。
给出详细的计算步骤和最终结果。"""

    try:
        # 绑定工具的 LLM
        llm_with_tools = llm.bind_tools([calculator])
        response = llm_with_tools.invoke(
            [SystemMessage(content=system_prompt), HumanMessage(content=user_input)]
        )

        # 检查是否有工具调用
        result_text = response.content
        if hasattr(response, "tool_calls") and response.tool_calls:
            # 执行工具调用
            for tool_call in response.tool_calls:
                if tool_call["name"] == "calculator":
                    expr = tool_call["args"]["expression"]
                    calc_result = calculator.invoke({"expression": expr})
                    result_text = f"{result_text}\n\n工具调用: {calc_result}"

        state["results"]["calculation"] = result_text
        state["current_step"] = "math_calculator"

        print(f"✅ 计算结果:\n{result_text}")
        print("=" * 70)

    except Exception as e:
        print(f"❌ 计算出错: {e}")
        state["results"]["calculation"] = f"计算失败: {str(e)}"

    return state


def math_prover_node(state: AgentState) -> AgentState:
    """数学证明节点：进行数学推理和证明"""
    print("\n" + "=" * 70)
    print("📐 [PROVER] 执行数学证明...")
    print("=" * 70)

    messages = state["messages"]
    user_input = messages[-1].content if messages else ""

    system_prompt = """你是一个数学证明专家。对用户提出的数学问题进行严谨的推理和证明。

请按照以下步骤：
1. 明确要证明的命题
2. 列出已知条件
3. 给出证明步骤（使用数学符号和逻辑推理）
4. 得出结论

要求逻辑严密，步骤清晰。"""

    try:
        response = llm.invoke(
            [SystemMessage(content=system_prompt), HumanMessage(content=user_input)]
        )

        proof_text = response.content
        state["results"]["proof"] = proof_text
        state["current_step"] = "math_prover"

        print(f"✅ 证明过程:\n{proof_text}")
        print("=" * 70)

    except Exception as e:
        print(f"❌ 证明出错: {e}")
        state["results"]["proof"] = f"证明失败: {str(e)}"

    return state


def logic_reasoner_node(state: AgentState) -> AgentState:
    """逻辑推理节点：处理逻辑问题"""
    print("\n" + "=" * 70)
    print("🧠 [LOGIC] 执行逻辑推理...")
    print("=" * 70)

    messages = state["messages"]
    user_input = messages[-1].content if messages else ""

    system_prompt = """你是一个逻辑推理专家。分析用户提出的逻辑问题，进行系统的推理。

请按照以下步骤：
1. 理解问题中的逻辑关系
2. 列出关键条件和约束
3. 进行逻辑推导（可以使用符号表示）
4. 得出结论并验证

要求推理严密，逻辑清晰。"""

    try:
        response = llm.invoke(
            [SystemMessage(content=system_prompt), HumanMessage(content=user_input)]
        )

        reasoning_text = response.content
        state["results"]["reasoning"] = reasoning_text
        state["current_step"] = "logic_reasoner"

        print(f"✅ 推理过程:\n{reasoning_text}")
        print("=" * 70)

    except Exception as e:
        print(f"❌ 推理出错: {e}")
        state["results"]["reasoning"] = f"推理失败: {str(e)}"

    return state


def summarizer_node(state: AgentState) -> AgentState:
    """总结节点：汇总所有结果并给出最终答案"""
    print("\n" + "=" * 70)
    print("📝 [SUMMARIZER] 总结结果...")
    print("=" * 70)

    messages = state["messages"]
    user_input = messages[-1].content if messages else ""
    results = state.get("results", {})
    plan = state.get("plan", "")
    task_type = state.get("task_type", "general")

    # 如果是 general 类型，直接回答
    if task_type == "general" or not results:
        system_prompt = """你是一个友好的助手。直接回答用户的问题。"""
        try:
            response = llm.invoke(
                [SystemMessage(content=system_prompt), HumanMessage(content=user_input)]
            )
            final_answer = response.content
        except Exception as e:
            final_answer = f"抱歉，处理问题时出错: {str(e)}"
    else:
        # 汇总各节点的结果
        system_prompt = """你是一个总结助手。根据之前各个节点的执行结果，给出清晰、完整的最终答案。

要求：
1. 综合所有信息
2. 突出重点结论
3. 语言简洁清晰
4. 如果有计算结果，要明确给出"""

        results_summary = "\n\n".join(
            [f"**{key}节点结果:**\n{value}" for key, value in results.items()]
        )

        try:
            response = llm.invoke(
                [
                    SystemMessage(content=system_prompt),
                    HumanMessage(
                        content=f"原始问题: {user_input}\n\n执行计划: {plan}\n\n各节点结果:\n{results_summary}\n\n请给出最终答案:"
                    ),
                ]
            )
            final_answer = response.content
        except Exception as e:
            final_answer = f"总结失败，但以下是各节点的结果：\n{results_summary}"

    state["final_answer"] = final_answer
    state["current_step"] = "summarizer"

    print(f"✅ 最终答案:\n{final_answer}")
    print("=" * 70)

    return state


# ============================================================
# 4. 定义路由函数
# ============================================================


def route_after_planner(
    state: AgentState,
) -> Literal["math_calculator", "math_prover", "logic_reasoner", "summarizer"]:
    """根据任务类型路由到不同的工作节点"""
    task_type = state.get("task_type", "general")

    print(f"\n🔀 [ROUTER] 任务类型: {task_type}, 路由到相应节点...")

    routing_map = {
        "math_calc": "math_calculator",
        "math_proof": "math_prover",
        "logic": "logic_reasoner",
        "general": "summarizer",
    }

    next_node = routing_map.get(task_type, "summarizer")
    print(f"🔀 [ROUTER] 下一个节点: {next_node}\n")

    return next_node


# ============================================================
# 5. 创建 Agent
# ============================================================


def create_agent(config):
    """创建 LangGraph Agent

    Args:
        config: 配置字典，包含 base_url 和 api_key

    Returns:
        编译好的 LangGraph 应用
    """
    global llm

    print("\n🔧 正在初始化 Agent...")
    print(f"📡 模型配置:")
    print(f"   - Base URL: {config['base_url']}")
    print(f"   - Model: {config.get('model', 'deepseek-chat')}")

    # 1. 初始化模型（原生方式）
    llm = ChatOpenAI(
        model=config.get("model", "deepseek-chat"),
        api_key=config["api_key"],
        base_url=config["base_url"],
        temperature=0.1,
    )

    # 2. 创建 StateGraph
    workflow = StateGraph(AgentState)

    # 3. 添加节点
    workflow.add_node("planner", planner_node)
    workflow.add_node("math_calculator", math_calculator_node)
    workflow.add_node("math_prover", math_prover_node)
    workflow.add_node("logic_reasoner", logic_reasoner_node)
    workflow.add_node("summarizer", summarizer_node)

    # 4. 添加边（定义工作流）
    # 开始 → planner
    workflow.add_edge(START, "planner")

    # planner → 条件路由到各个工作节点
    workflow.add_conditional_edges(
        "planner",
        route_after_planner,
        {
            "math_calculator": "math_calculator",
            "math_prover": "math_prover",
            "logic_reasoner": "logic_reasoner",
            "summarizer": "summarizer",
        },
    )

    # 所有工作节点 → summarizer
    workflow.add_edge("math_calculator", "summarizer")
    workflow.add_edge("math_prover", "summarizer")
    workflow.add_edge("logic_reasoner", "summarizer")

    # summarizer → END
    workflow.add_edge("summarizer", END)

    # 5. 编译（带内存管理）
    app = workflow.compile(checkpointer=MemorySaver())

    print("✅ Agent 初始化完成！\n")

    return app


# ============================================================
# 6. 交互模式
# ============================================================


def interactive_mode():
    """交互模式"""
    print("\n" + "=" * 70)
    print("🎯 LangGraph Agent Demo - 交互模式")
    print("=" * 70)

    # 模型配置（可手动修改）
    # DeepSeek 在线模型
    load_dotenv()
    base_url = "https://api.deepseek.com"
    api_key = os.getenv("DEEPSEEK_API_KEY")

    # 如果使用本地部署的 Qwen-7B，修改为：
    # base_url = "http://localhost:8000/v1"
    # api_key = "EMPTY"

    config = {
        "base_url": base_url,
        "api_key": api_key,
        "model": "deepseek-chat",  # 或 "qwen-7b"
    }

    agent = create_agent(config)

    print("✨ 智能助手已就绪！输入 'quit' 或 'exit' 退出")
    print("💡 提示：我可以帮你进行数学计算、数学证明、逻辑推理等\n")

    session_id = "interactive_session"

    while True:
        try:
            user_input = input("💬 你: ").strip()

            if user_input.lower() in ["quit", "exit", "退出"]:
                print("\n👋 再见！")
                break
            elif not user_input:
                continue

            # 构建初始状态
            initial_state = {
                "messages": [HumanMessage(content=user_input)],
                "task_type": "",
                "plan": "",
                "current_step": "",
                "results": {},
                "final_answer": "",
            }

            # 执行 agent
            thread_config = {"configurable": {"thread_id": session_id}}

            print("\n🤖 助手正在思考...")
            final_state = agent.invoke(initial_state, config=thread_config)

            # 显示最终答案
            print(f"\n{'='*70}")
            print("🎉 最终回答:")
            print(f"{'='*70}")
            print(final_state.get("final_answer", "抱歉，没有生成答案"))
            print(f"{'='*70}\n")

        except KeyboardInterrupt:
            print("\n\n👋 再见！")
            break
        except Exception as e:
            print(f"\n❌ 出错了: {str(e)}\n")


# ============================================================
# 7. 主程序入口
# ============================================================

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("🎊 欢迎使用 LangGraph 智能助手 Demo！")
    print("=" * 70)
    print("\n📚 这是一个基于原生 LangGraph 的演示程序")
    print("🔧 包含：规划器 + 数学计算 + 数学证明 + 逻辑推理 + 总结器")
    print("💻 使用 langgraph-agent conda 环境运行\n")

    interactive_mode()
