# AI Agent实例代码：LangGraph与Agno框架完整实现

## 概述

本文档提供两个功能完整的AI agent实例代码，分别基于**LangGraph框架**和**Agno框架**开发。两个agent都具备对话、多步推理、工具调用和任务规划能力。

## 1. LangGraph Agent实现

### 1.1 环境搭建

```bash
# 创建虚拟环境
python -m venv langgraph_env
source langgraph_env/bin/activate  # Windows: langgraph_env\Scripts\activate

# 安装依赖
pip install -U langgraph langchain-openai langchain-community
pip install tavily-python duckduckgo-search yfinance
pip install python-dotenv pydantic typing-extensions

# 设置环境变量
export OPENAI_API_KEY="sk-your-key-here"
export TAVILY_API_KEY="tvly-your-key-here"
```

### 1.2 完整代码实现

```python
"""
LangGraph智能助手 - 完整实现
功能：对话、多步推理、工具调用、任务规划
作者：AI Assistant
版本：1.0
"""

import os
from typing import List, Dict, Any, Literal, Annotated
from typing_extensions import TypedDict
from dataclasses import dataclass
import json
from datetime import datetime

# LangGraph相关导入
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

# LangChain相关导入
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool

# 工具导入
from tavily import TavilyClient
from duckduckgo_search import DDGS
import yfinance as yf

# 环境配置
from dotenv import load_dotenv
load_dotenv()

# ========== 状态定义 ==========
class AgentState(TypedDict):
    """智能体状态定义"""
    messages: List[Any]  # 消息历史
    current_task: str   # 当前任务
    task_type: str      # 任务类型：simple_chat, research, analysis, planning
    search_results: List[Dict]  # 搜索结果
    analysis_results: Dict[str, Any]  # 分析结果
    reasoning_steps: List[str]  # 推理步骤
    plan: List[Dict]    # 任务计划
    completed_steps: List[str]  # 已完成步骤
    context: Dict[str, Any]  # 上下文信息
    next_action: str    # 下一步动作


# ========== 工具定义 ==========
@tool
def web_search(query: str, max_results: int = 5) -> str:
    """
    使用Tavily搜索引擎进行网络搜索
    
    Args:
        query: 搜索查询
        max_results: 最大结果数
    
    Returns:
        str: 搜索结果摘要
    """
    try:
        client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
        response = client.search(
            query=query,
            search_depth="advanced",
            max_results=max_results,
            include_answer=True
        )
        
        results = []
        for result in response.get("results", []):
            results.append(f"标题: {result['title']}\n内容: {result['content'][:300]}...\nURL: {result['url']}")
        
        return "\n\n---\n\n".join(results)
    except Exception as e:
        return f"搜索失败: {str(e)}"


@tool
def duckduckgo_search(query: str, max_results: int = 5) -> str:
    """
    使用DuckDuckGo进行隐私友好的搜索
    
    Args:
        query: 搜索查询
        max_results: 最大结果数
    
    Returns:
        str: 搜索结果
    """
    try:
        ddgs = DDGS()
        results = list(ddgs.text(query, max_results=max_results))
        
        formatted_results = []
        for result in results:
            formatted_results.append(f"标题: {result['title']}\n摘要: {result['body']}\nURL: {result['href']}")
        
        return "\n\n---\n\n".join(formatted_results)
    except Exception as e:
        return f"DuckDuckGo搜索失败: {str(e)}"


@tool
def get_stock_info(symbol: str) -> str:
    """
    获取股票信息
    
    Args:
        symbol: 股票代码 (如: AAPL, TSLA)
    
    Returns:
        str: 股票信息
    """
    try:
        stock = yf.Ticker(symbol)
        info = stock.info
        hist = stock.history(period="1d")
        
        result = f"""
股票代码: {symbol}
公司名称: {info.get('longName', 'N/A')}
当前价格: ${info.get('currentPrice', 'N/A')}
市值: ${info.get('marketCap', 'N/A'):,} 
52周最高: ${info.get('fiftyTwoWeekHigh', 'N/A')}
52周最低: ${info.get('fiftyTwoWeekLow', 'N/A')}
行业: {info.get('industry', 'N/A')}
简介: {info.get('longBusinessSummary', 'N/A')[:200]}...
        """
        return result
    except Exception as e:
        return f"获取股票信息失败: {str(e)}"


@tool
def calculator(expression: str) -> str:
    """
    执行数学计算
    
    Args:
        expression: 数学表达式
    
    Returns:
        str: 计算结果
    """
    try:
        # 安全的数学表达式计算
        allowed_chars = set('0123456789+-*/(). ')
        if not all(c in allowed_chars for c in expression):
            return "计算表达式包含不允许的字符"
        
        result = eval(expression)
        return f"计算结果: {expression} = {result}"
    except Exception as e:
        return f"计算错误: {str(e)}"


# 工具列表
tools = [web_search, duckduckgo_search, get_stock_info, calculator]


# ========== 核心Agent类 ==========
class LangGraphAgent:
    """基于LangGraph的智能助手"""
    
    def __init__(self, model_name: str = "gpt-4o"):
        """
        初始化智能助手
        
        Args:
            model_name: 使用的模型名称
        """
        self.llm = ChatOpenAI(model=model_name, temperature=0.1)
        self.memory = MemorySaver()
        
        # 创建工具节点
        self.tool_node = ToolNode(tools)
        
        # 构建图
        self.app = self._build_graph()
        
        print(f"✅ LangGraph智能助手初始化完成，使用模型: {model_name}")
    
    def _build_graph(self) -> StateGraph:
        """构建LangGraph工作流图"""
        
        # 创建状态图
        workflow = StateGraph(AgentState)
        
        # 添加节点
        workflow.add_node("classifier", self._classify_task)          # 任务分类
        workflow.add_node("simple_chat", self._simple_chat)           # 简单对话
        workflow.add_node("planner", self._create_plan)              # 任务规划
        workflow.add_node("researcher", self._research_node)          # 研究节点
        workflow.add_node("analyzer", self._analysis_node)           # 分析节点
        workflow.add_node("reasoning", self._reasoning_node)         # 推理节点
        workflow.add_node("tools", self.tool_node)                   # 工具调用
        workflow.add_node("synthesizer", self._synthesize_results)   # 结果综合
        
        # 添加边和条件路由
        workflow.add_edge(START, "classifier")
        
        # 从分类器到不同处理路径
        workflow.add_conditional_edges(
            "classifier",
            self._route_after_classification,
            {
                "simple_chat": "simple_chat",
                "research": "planner", 
                "analysis": "planner",
                "planning": "planner"
            }
        )
        
        # 简单对话路径
        workflow.add_conditional_edges(
            "simple_chat",
            tools_condition,
            {
                "tools": "tools",
                "end": END
            }
        )
        
        # 工具调用后回到简单对话
        workflow.add_edge("tools", "simple_chat")
        
        # 规划后的路径
        workflow.add_edge("planner", "researcher")
        workflow.add_edge("researcher", "analyzer")  
        workflow.add_edge("analyzer", "reasoning")
        workflow.add_edge("reasoning", "synthesizer")
        workflow.add_edge("synthesizer", END)
        
        return workflow.compile(checkpointer=self.memory)
    
    def _classify_task(self, state: AgentState) -> Dict[str, Any]:
        """任务分类节点"""
        messages = state.get("messages", [])
        if not messages:
            return {"task_type": "simple_chat"}
            
        last_message = messages[-1].content if messages else ""
        
        # 使用LLM进行任务分类
        classification_prompt = f"""
        分析用户的请求，将其分类为以下类型之一：
        1. simple_chat - 简单对话、问候、基本问答
        2. research - 需要搜索信息的研究任务
        3. analysis - 需要深度分析的任务
        4. planning - 需要制定计划或策略的任务
        
        用户请求: {last_message}
        
        只返回类型名称，不要其他文字。
        """
        
        response = self.llm.invoke([HumanMessage(content=classification_prompt)])
        task_type = response.content.strip().lower()
        
        # 验证返回值
        valid_types = ["simple_chat", "research", "analysis", "planning"]
        if task_type not in valid_types:
            task_type = "simple_chat"
        
        return {
            "task_type": task_type,
            "current_task": last_message,
            "reasoning_steps": [f"任务分类: {task_type}"]
        }
    
    def _simple_chat(self, state: AgentState) -> Dict[str, Any]:
        """简单对话处理节点"""
        messages = state.get("messages", [])
        
        # 构建对话提示
        system_prompt = """
        你是一个友好、有用的AI助手。你可以：
        1. 进行日常对话
        2. 回答一般性问题
        3. 在需要时调用工具获取信息
        4. 进行数学计算
        5. 搜索最新信息
        
        如果用户的问题需要搜索最新信息、股票数据或数学计算，请调用相应的工具。
        """
        
        full_messages = [{"role": "system", "content": system_prompt}] + [
            {"role": "human" if isinstance(msg, HumanMessage) else "ai", 
             "content": msg.content} for msg in messages
        ]
        
        response = self.llm.bind_tools(tools).invoke(full_messages)
        
        return {
            "messages": messages + [response]
        }
    
    def _create_plan(self, state: AgentState) -> Dict[str, Any]:
        """任务规划节点"""
        current_task = state.get("current_task", "")
        task_type = state.get("task_type", "")
        
        planning_prompt = f"""
        为以下{task_type}任务制定详细的执行计划：
        
        任务: {current_task}
        
        请将任务分解为具体的步骤，每个步骤包括：
        1. 步骤名称
        2. 具体行动
        3. 预期结果
        
        以JSON格式返回计划，格式如下：
        {{
            "steps": [
                {{
                    "name": "步骤名称",
                    "action": "具体行动", 
                    "expected_result": "预期结果"
                }}
            ]
        }}
        """
        
        response = self.llm.invoke([HumanMessage(content=planning_prompt)])
        
        try:
            plan_data = json.loads(response.content)
            plan = plan_data.get("steps", [])
        except:
            # 如果JSON解析失败，创建默认计划
            plan = [
                {
                    "name": "信息收集",
                    "action": "搜索相关信息",
                    "expected_result": "获得背景信息"
                },
                {
                    "name": "深度分析", 
                    "action": "分析收集到的信息",
                    "expected_result": "形成初步结论"
                },
                {
                    "name": "推理综合",
                    "action": "进行逻辑推理",
                    "expected_result": "得出最终答案"
                }
            ]
        
        reasoning_steps = state.get("reasoning_steps", [])
        reasoning_steps.append(f"制定了包含{len(plan)}个步骤的执行计划")
        
        return {
            "plan": plan,
            "reasoning_steps": reasoning_steps
        }
    
    def _research_node(self, state: AgentState) -> Dict[str, Any]:
        """研究信息收集节点"""
        current_task = state.get("current_task", "")
        plan = state.get("plan", [])
        
        # 执行研究步骤
        search_queries = self._generate_search_queries(current_task)
        search_results = []
        
        for query in search_queries[:3]:  # 限制搜索查询数量
            try:
                # 使用Tavily搜索
                result = web_search.invoke({"query": query, "max_results": 3})
                search_results.append({
                    "query": query,
                    "results": result
                })
            except:
                # 备用DuckDuckGo搜索
                try:
                    result = duckduckgo_search.invoke({"query": query, "max_results": 3})
                    search_results.append({
                        "query": query, 
                        "results": result
                    })
                except:
                    pass
        
        reasoning_steps = state.get("reasoning_steps", [])
        reasoning_steps.append(f"完成信息收集，获得{len(search_results)}组搜索结果")
        
        return {
            "search_results": search_results,
            "reasoning_steps": reasoning_steps
        }
    
    def _analysis_node(self, state: AgentState) -> Dict[str, Any]:
        """分析处理节点"""
        search_results = state.get("search_results", [])
        current_task = state.get("current_task", "")
        
        # 整合搜索结果
        all_info = ""
        for result in search_results:
            all_info += f"查询: {result['query']}\n结果: {result['results']}\n\n"
        
        analysis_prompt = f"""
        基于收集到的信息，对以下任务进行深度分析：
        
        任务: {current_task}
        
        收集到的信息:
        {all_info}
        
        请进行多角度分析：
        1. 关键信息总结
        2. 重要发现和趋势
        3. 不同观点对比
        4. 潜在影响和意义
        
        请以结构化方式组织分析结果。
        """
        
        response = self.llm.invoke([HumanMessage(content=analysis_prompt)])
        
        analysis_results = {
            "summary": "分析完成",
            "content": response.content,
            "timestamp": datetime.now().isoformat()
        }
        
        reasoning_steps = state.get("reasoning_steps", [])
        reasoning_steps.append("完成信息分析，形成结构化见解")
        
        return {
            "analysis_results": analysis_results,
            "reasoning_steps": reasoning_steps
        }
    
    def _reasoning_node(self, state: AgentState) -> Dict[str, Any]:
        """推理思考节点"""
        analysis_results = state.get("analysis_results", {})
        current_task = state.get("current_task", "")
        
        reasoning_prompt = f"""
        基于分析结果，请进行逐步推理来回答用户的问题：
        
        原始问题: {current_task}
        分析结果: {analysis_results.get('content', '')}
        
        请使用以下推理框架：
        
        1. 问题理解：重新阐述核心问题
        2. 关键因素：识别影响答案的关键因素  
        3. 逻辑推理：
           - 步骤1：[基于证据A得出结论1]
           - 步骤2：[基于证据B得出结论2]
           - 步骤3：[综合结论1和2得出最终结论]
        4. 验证检查：检验结论的合理性
        5. 最终答案：清晰明确的回答
        
        请严格按照这个结构进行推理。
        """
        
        response = self.llm.invoke([HumanMessage(content=reasoning_prompt)])
        
        reasoning_steps = state.get("reasoning_steps", [])
        reasoning_steps.append("完成逻辑推理分析")
        
        return {
            "reasoning_steps": reasoning_steps,
            "reasoning_content": response.content
        }
    
    def _synthesize_results(self, state: AgentState) -> Dict[str, Any]:
        """结果综合节点"""
        current_task = state.get("current_task", "")
        analysis_results = state.get("analysis_results", {})
        reasoning_content = state.get("reasoning_content", "")
        reasoning_steps = state.get("reasoning_steps", [])
        
        synthesis_prompt = f"""
        请将所有分析和推理结果综合成一个完整、清晰的最终回答：
        
        原始问题: {current_task}
        
        分析结果: {analysis_results.get('content', '')}
        
        推理过程: {reasoning_content}
        
        请提供：
        1. 直接明确的答案
        2. 支持证据和理由
        3. 相关的背景信息
        4. 如果有的话，建议下一步行动
        
        请以自然、易懂的方式组织回答。
        """
        
        final_response = self.llm.invoke([HumanMessage(content=synthesis_prompt)])
        
        reasoning_steps.append("完成结果综合，生成最终回答")
        
        messages = state.get("messages", [])
        messages.append(AIMessage(content=final_response.content))
        
        return {
            "messages": messages,
            "reasoning_steps": reasoning_steps
        }
    
    def _generate_search_queries(self, task: str) -> List[str]:
        """生成搜索查询"""
        query_prompt = f"""
        为以下任务生成2-3个有效的搜索查询：
        
        任务: {task}
        
        请生成能够获得相关、准确信息的搜索查询。
        每行一个查询，不要编号。
        """
        
        response = self.llm.invoke([HumanMessage(content=query_prompt)])
        queries = [q.strip() for q in response.content.split('\n') if q.strip()]
        
        return queries[:3]  # 限制查询数量
    
    def _route_after_classification(self, state: AgentState) -> str:
        """分类后的路由决策"""
        task_type = state.get("task_type", "simple_chat")
        return task_type
    
    def chat(self, message: str, session_id: str = "default") -> str:
        """
        与智能助手对话
        
        Args:
            message: 用户消息
            session_id: 会话ID，用于维护对话历史
        
        Returns:
            str: 助手回复
        """
        try:
            # 准备初始状态
            initial_state = {
                "messages": [HumanMessage(content=message)],
                "current_task": "",
                "task_type": "",
                "search_results": [],
                "analysis_results": {},
                "reasoning_steps": [],
                "plan": [],
                "completed_steps": [],
                "context": {},
                "next_action": ""
            }
            
            # 执行图处理
            config = {"configurable": {"thread_id": session_id}}
            final_state = self.app.invoke(initial_state, config=config)
            
            # 提取回复
            messages = final_state.get("messages", [])
            if messages and isinstance(messages[-1], AIMessage):
                return messages[-1].content
            else:
                return "抱歉，处理过程中出现了问题。"
                
        except Exception as e:
            return f"对话处理失败: {str(e)}"
    
    def get_reasoning_steps(self, session_id: str = "default") -> List[str]:
        """获取推理步骤"""
        try:
            config = {"configurable": {"thread_id": session_id}}
            state = self.app.get_state(config)
            return state.values.get("reasoning_steps", [])
        except:
            return []


# ========== 使用示例 ==========
def demo_langgraph_agent():
    """LangGraph Agent 演示"""
    print("\n" + "="*60)
    print("🚀 LangGraph智能助手演示")
    print("="*60)
    
    # 创建智能助手
    agent = LangGraphAgent()
    
    # 测试场景
    test_cases = [
        {
            "name": "简单对话",
            "message": "你好，请介绍一下自己的能力"
        },
        {
            "name": "数学计算", 
            "message": "请帮我计算 (125 + 75) * 2 - 50"
        },
        {
            "name": "信息研究",
            "message": "请调研一下2024年人工智能的最新发展趋势"
        },
        {
            "name": "股票分析",
            "message": "分析一下特斯拉(TSLA)股票的投资价值"
        }
    ]
    
    for i, test in enumerate(test_cases, 1):
        print(f"\n📋 测试{i}：{test['name']}")
        print(f"问题：{test['message']}")
        print("\n💭 处理中...")
        
        response = agent.chat(test['message'], session_id=f"demo_{i}")
        
        print(f"\n🤖 回答：")
        print(response)
        
        # 显示推理步骤
        steps = agent.get_reasoning_steps(f"demo_{i}")
        if steps:
            print(f"\n🧠 推理步骤：")
            for step in steps:
                print(f"  • {step}")
        
        print("\n" + "-"*40)
    
    print("\n✅ LangGraph Agent演示完成！")


if __name__ == "__main__":
    demo_langgraph_agent()
```

---

## 2. Agno Agent实现

### 2.1 环境搭建

```bash
# 创建虚拟环境
python -m venv agno_env
source agno_env/bin/activate  # Windows: agno_env\Scripts\activate

# 安装依赖
pip install -U agno openai anthropic
pip install tavily-python duckduckgo-search yfinance
pip install python-dotenv

# 设置环境变量
export OPENAI_API_KEY="sk-your-key-here"
export ANTHROPIC_API_KEY="sk-ant-your-key-here"
export TAVILY_API_KEY="tvly-your-key-here"
```

### 2.2 完整代码实现

```python
"""
Agno智能助手 - 完整实现
功能：对话、多步推理、工具调用、任务规划
作者：AI Assistant
版本：1.0
"""

import os
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import json
from datetime import datetime

# Agno框架导入
from agno.agent import Agent
from agno.team import Team  
from agno.workflow import Workflow
from agno.models.openai import OpenAIChat
from agno.models.anthropic import Claude
from agno.tools import Tool
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.tools.yfinance import YFinanceTools
from agno.tools.reasoning import ReasoningTools

# 其他导入
from tavily import TavilyClient
import yfinance as yf

# 环境配置
from dotenv import load_dotenv
load_dotenv()


# ========== 自定义工具类 ==========
class TavilySearchTool(Tool):
    """Tavily搜索工具"""
    
    def __init__(self):
        super().__init__(
            name="tavily_search",
            description="使用Tavily搜索引擎进行高质量网络搜索，专为AI优化",
        )
        self.client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
    
    def run(self, query: str, max_results: int = 5) -> str:
        """执行搜索"""
        try:
            response = self.client.search(
                query=query,
                search_depth="advanced", 
                max_results=max_results,
                include_answer=True
            )
            
            results = []
            for result in response.get("results", []):
                results.append(
                    f"标题: {result['title']}\n"
                    f"内容: {result['content'][:300]}...\n"
                    f"URL: {result['url']}\n"
                )
            
            return "\n---\n".join(results)
        except Exception as e:
            return f"Tavily搜索失败: {str(e)}"


class AdvancedCalculatorTool(Tool):
    """高级计算器工具"""
    
    def __init__(self):
        super().__init__(
            name="advanced_calculator",
            description="执行数学计算、统计分析等运算",
        )
    
    def run(self, expression: str) -> str:
        """执行计算"""
        try:
            # 安全计算
            allowed_chars = set('0123456789+-*/().eE ')
            allowed_words = {'sin', 'cos', 'tan', 'log', 'exp', 'sqrt', 'pi', 'abs'}
            
            # 基本安全检查
            clean_expr = expression.replace(' ', '')
            if not all(c in allowed_chars or c.isalpha() for c in clean_expr):
                return "表达式包含不允许的字符"
            
            # 简单数学运算
            import math
            safe_dict = {
                "__builtins__": {},
                "sin": math.sin, "cos": math.cos, "tan": math.tan,
                "log": math.log, "exp": math.exp, "sqrt": math.sqrt,
                "pi": math.pi, "abs": abs, "pow": pow
            }
            
            result = eval(expression, safe_dict)
            return f"计算结果: {expression} = {result}"
        except Exception as e:
            return f"计算错误: {str(e)}"


class EnhancedStockTool(Tool):
    """增强股票分析工具"""
    
    def __init__(self):
        super().__init__(
            name="enhanced_stock_analysis",
            description="获取详细的股票信息、财务数据和技术分析",
        )
    
    def run(self, symbol: str) -> str:
        """获取股票详细信息"""
        try:
            stock = yf.Ticker(symbol)
            info = stock.info
            hist = stock.history(period="1mo")
            
            # 基本信息
            basic_info = f"""
📊 股票代码: {symbol}
🏢 公司名称: {info.get('longName', 'N/A')}
💰 当前价格: ${info.get('currentPrice', 'N/A')}
📈 今日涨跌: {info.get('regularMarketChangePercent', 'N/A')}%
🏆 市值: ${info.get('marketCap', 0):,}
📊 市盈率: {info.get('trailingPE', 'N/A')}
🎯 52周最高: ${info.get('fiftyTwoWeekHigh', 'N/A')}
📉 52周最低: ${info.get('fiftyTwoWeekLow', 'N/A')}
🏭 行业: {info.get('industry', 'N/A')}
🌍 板块: {info.get('sector', 'N/A')}
            """
            
            # 技术指标
            if not hist.empty:
                current_price = hist['Close'][-1]
                sma_20 = hist['Close'].rolling(window=20).mean()[-1]
                volatility = hist['Close'].pct_change().std() * 100
                
                technical_info = f"""
📈 技术指标:
  - 20日均线: ${sma_20:.2f}
  - 当前价格vs均线: {'上方' if current_price > sma_20 else '下方'}
  - 波动率: {volatility:.2f}%
                """
            else:
                technical_info = "📈 技术指标: 数据不足"
            
            # 公司描述
            description = info.get('longBusinessSummary', 'N/A')
            if len(description) > 300:
                description = description[:300] + "..."
            
            return f"{basic_info}\n{technical_info}\n\n📝 公司简介:\n{description}"
            
        except Exception as e:
            return f"获取股票信息失败: {str(e)}"


# ========== 多智能体团队 ==========
class ResearchTeam:
    """研究团队 - 多智能体协作"""
    
    def __init__(self, model_provider: str = "openai"):
        """
        初始化研究团队
        
        Args:
            model_provider: 模型提供商 ("openai" 或 "anthropic")
        """
        # 选择基础模型
        if model_provider == "anthropic":
            base_model = Claude(id="claude-3-5-sonnet-20241022")
        else:
            base_model = OpenAIChat(id="gpt-4o")
        
        # 创建专门化的智能体
        self.web_researcher = Agent(
            name="网络研究专家",
            role="负责搜索和收集网络信息",
            model=base_model,
            tools=[
                TavilySearchTool(),
                DuckDuckGoTools(),
            ],
            instructions=[
                "专注于搜索高质量、权威的信息源",
                "总是提供信息来源链接",
                "优先使用最新的信息",
                "对搜索结果进行初步筛选和整理"
            ],
            markdown=True,
        )
        
        self.financial_analyst = Agent(
            name="金融分析师", 
            role="负责金融数据分析和股票研究",
            model=base_model,
            tools=[
                EnhancedStockTool(),
                YFinanceTools(
                    stock_price=True,
                    company_info=True,
                    analyst_recommendations=True,
                    company_news=True
                ),
                AdvancedCalculatorTool(),
            ],
            instructions=[
                "提供深度的金融分析",
                "使用表格展示关键数据",
                "包含风险评估",
                "给出明确的投资建议"
            ],
            markdown=True,
        )
        
        self.reasoning_expert = Agent(
            name="推理分析专家",
            role="负责逻辑推理和综合分析",
            model=base_model,
            tools=[ReasoningTools(add_instructions=True)],
            instructions=[
                "使用逐步推理解决复杂问题",
                "展示完整的思考过程",
                "从多个角度分析问题", 
                "得出有逻辑支撑的结论"
            ],
            markdown=True,
            show_tool_calls=True,
        )
        
        # 创建协作团队
        self.team = Team(
            model=base_model,
            members=[self.web_researcher, self.financial_analyst, self.reasoning_expert],
            instructions=[
                "团队协作完成复杂任务",
                "每个成员发挥自己的专长",
                "最终提供全面、准确的答案",
                "保持逻辑清晰和结构化"
            ],
            show_tool_calls=True,
            markdown=True,
        )
        
        print(f"✅ Agno研究团队初始化完成，使用{model_provider}模型")
    
    def research(self, topic: str) -> str:
        """执行研究任务"""
        return self.team.run(f"请深入研究: {topic}")


# ========== 智能工作流 ==========
class IntelligentWorkflow(Workflow):
    """智能工作流 - 任务规划和执行"""
    
    def __init__(self, model_provider: str = "openai"):
        super().__init__(name="intelligent_workflow")
        
        # 选择模型
        if model_provider == "anthropic":
            model = Claude(id="claude-3-5-sonnet-20241022")
        else:
            model = OpenAIChat(id="gpt-4o")
        
        # 任务分类器
        self.classifier = Agent(
            name="任务分类器",
            role="分析和分类用户请求",
            model=model,
            instructions=[
                "分析用户请求的类型和复杂度",
                "决定最适合的处理方式",
                "提供清晰的分类结果"
            ]
        )
        
        # 通用对话助手
        self.chat_assistant = Agent(
            name="对话助手", 
            role="处理日常对话和简单问答",
            model=model,
            tools=[AdvancedCalculatorTool()],
            instructions=[
                "提供友好、有用的回答",
                "对于简单问题直接回答",
                "必要时使用工具辅助"
            ]
        )
        
        # 研究团队
        self.research_team = ResearchTeam(model_provider)
        
        print(f"✅ 智能工作流初始化完成")
    
    def run(self, user_request: str) -> str:
        """执行工作流"""
        
        # Step 1: 任务分类
        classification = self.classifier.run(f"""
        请分析以下用户请求，并分类：
        
        用户请求: {user_request}
        
        分类选项:
        1. simple_chat - 简单对话、问候、基本问答
        2. calculation - 数学计算
        3. research - 需要搜索信息的研究任务
        4. financial_analysis - 金融、股票相关分析
        5. complex_analysis - 需要深度分析和推理的复杂任务
        
        请只返回分类名称，不要其他内容。
        """)
        
        task_type = classification.strip().lower()
        
        # Step 2: 根据分类执行相应处理
        if task_type in ["simple_chat", "calculation"]:
            return self.chat_assistant.run(user_request)
        
        elif task_type in ["research", "financial_analysis", "complex_analysis"]:
            return self.research_team.research(user_request)
        
        else:
            # 默认使用聊天助手
            return self.chat_assistant.run(user_request)


# ========== 主要Agent类 ==========
class AgnoAgent:
    """基于Agno的智能助手"""
    
    def __init__(self, model_provider: str = "openai"):
        """
        初始化Agno智能助手
        
        Args:
            model_provider: 模型提供商 ("openai" 或 "anthropic")
        """
        self.model_provider = model_provider
        
        # 创建工作流
        self.workflow = IntelligentWorkflow(model_provider)
        
        # 会话历史
        self.chat_history: Dict[str, List[str]] = {}
        
        print(f"🎉 Agno智能助手初始化完成！")
        print(f"📍 模型提供商: {model_provider}")
        print(f"🔧 功能: 对话、搜索、分析、推理、金融数据")
    
    def chat(self, message: str, session_id: str = "default") -> str:
        """
        与智能助手对话
        
        Args:
            message: 用户消息
            session_id: 会话ID
        
        Returns:
            str: 助手回复
        """
        try:
            # 维护会话历史
            if session_id not in self.chat_history:
                self.chat_history[session_id] = []
            
            # 记录用户消息
            self.chat_history[session_id].append(f"用户: {message}")
            
            # 执行工作流处理
            response = self.workflow.run(message)
            
            # 记录助手回复
            self.chat_history[session_id].append(f"助手: {response}")
            
            return response
            
        except Exception as e:
            error_msg = f"处理失败: {str(e)}"
            print(f"❌ {error_msg}")
            return error_msg
    
    def get_chat_history(self, session_id: str = "default") -> List[str]:
        """获取聊天历史"""
        return self.chat_history.get(session_id, [])
    
    def clear_history(self, session_id: str = "default"):
        """清除聊天历史"""
        if session_id in self.chat_history:
            del self.chat_history[session_id]
            print(f"✅ 已清除会话 {session_id} 的历史记录")
    
    def switch_model(self, new_provider: str):
        """切换模型提供商"""
        if new_provider in ["openai", "anthropic"]:
            self.model_provider = new_provider
            self.workflow = IntelligentWorkflow(new_provider)
            print(f"✅ 已切换到 {new_provider} 模型")
        else:
            print("❌ 不支持的模型提供商，支持: openai, anthropic")


# ========== 使用示例和演示 ==========
def demo_agno_agent():
    """Agno Agent 演示"""
    print("\n" + "="*60)
    print("🚀 Agno智能助手演示") 
    print("="*60)
    
    # 创建智能助手 - 可以选择不同模型
    print("选择模型提供商：")
    print("1. OpenAI (gpt-4o)")
    print("2. Anthropic (claude-3-5-sonnet)")
    
    choice = input("请输入选择 (1 或 2，默认1): ").strip()
    model_provider = "anthropic" if choice == "2" else "openai"
    
    agent = AgnoAgent(model_provider=model_provider)
    
    # 测试场景
    test_cases = [
        {
            "name": "简单对话",
            "message": "你好！请介绍一下你的能力",
        },
        {
            "name": "数学计算",
            "message": "计算复合增长率：初值100，年增长率15%，5年后的值是多少？"
        },
        {
            "name": "信息研究", 
            "message": "2024年AI大模型的最新发展趋势是什么？"
        },
        {
            "name": "股票分析",
            "message": "分析微软(MSFT)的股票投资价值，包括基本面和技术面"
        },
        {
            "name": "复杂推理",
            "message": "如果要在2025年开始投资AI相关股票，应该考虑哪些因素？请给出详细的分析框架"
        }
    ]
    
    for i, test in enumerate(test_cases, 1):
        print(f"\n{'='*40}")
        print(f"📋 测试 {i}: {test['name']}")
        print(f"{'='*40}")
        print(f"❓ 问题: {test['message']}")
        print("\n💭 处理中...")
        
        # 执行测试
        response = agent.chat(test['message'], session_id=f"demo_{i}")
        
        print(f"\n🤖 回答:")
        print("-" * 40)
        print(response)
        print("-" * 40)
        
        # 询问是否继续
        if i < len(test_cases):
            continue_test = input(f"\n继续下一个测试？(y/n, 默认y): ").strip().lower()
            if continue_test == 'n':
                break
    
    print(f"\n✅ Agno Agent演示完成！")
    
    # 显示会话历史示例
    print(f"\n📚 会话历史示例 (最后一个会话):")
    history = agent.get_chat_history("demo_1")
    for entry in history[-4:]:  # 显示最后4条记录
        print(f"  {entry[:100]}...")


def interactive_mode():
    """交互模式"""
    print("\n" + "="*60)
    print("🎯 进入交互模式")
    print("="*60)
    
    # 选择模型
    print("选择模型提供商：")
    print("1. OpenAI (gpt-4o)")  
    print("2. Anthropic (claude-3-5-sonnet)")
    
    choice = input("请输入选择 (1 或 2，默认1): ").strip()
    model_provider = "anthropic" if choice == "2" else "openai"
    
    agent = AgnoAgent(model_provider=model_provider)
    
    print("\n✨ 智能助手已就绪！输入 'quit' 退出，'clear' 清除历史")
    print(f"💡 提示：我可以帮你搜索信息、分析数据、计算数学题、分析股票等")
    
    session_id = "interactive"
    
    while True:
        try:
            user_input = input(f"\n💬 你: ").strip()
            
            if user_input.lower() == 'quit':
                print("👋 再见！")
                break
            elif user_input.lower() == 'clear':
                agent.clear_history(session_id)
                continue
            elif user_input.lower() == 'history':
                history = agent.get_chat_history(session_id)
                print("\n📚 对话历史:")
                for entry in history:
                    print(f"  {entry}")
                continue
            elif not user_input:
                continue
            
            print(f"\n🤖 助手: 思考中...")
            response = agent.chat(user_input, session_id)
            print(f"\n🤖 助手: {response}")
            
        except KeyboardInterrupt:
            print(f"\n\n👋 再见！")
            break
        except Exception as e:
            print(f"❌ 出错了: {str(e)}")


if __name__ == "__main__":
    print("🎊 欢迎使用Agno智能助手！")
    print("\n选择运行模式:")
    print("1. 演示模式 (自动运行测试案例)")
    print("2. 交互模式 (手动对话)")
    
    mode = input("\n请选择模式 (1 或 2，默认1): ").strip()
    
    if mode == "2":
        interactive_mode()
    else:
        demo_agno_agent()
```

---

## 3. 环境配置文件

### 3.1 .env 环境变量文件

```bash
# API密钥配置
OPENAI_API_KEY=sk-your-openai-key-here
ANTHROPIC_API_KEY=sk-ant-your-anthropic-key-here  
TAVILY_API_KEY=tvly-your-tavily-key-here

# 可选配置
GROQ_API_KEY=gsk-your-groq-key-here

# Agno配置
AGNO_TELEMETRY=false
```

### 3.2 requirements.txt 依赖文件

#### LangGraph版本:
```
langgraph>=0.2.76
langchain-openai>=0.2.0
langchain-community>=0.3.0  
langchain-core>=0.3.0
tavily-python>=0.5.0
duckduckgo-search>=6.0.0
yfinance>=0.2.0
python-dotenv>=1.0.0
pydantic>=2.5.0
typing-extensions>=4.8.0
```

#### Agno版本:
```
agno>=1.7.7
openai>=1.50.0
anthropic>=0.34.0
tavily-python>=0.5.0
duckduckgo-search>=6.0.0
yfinance>=0.2.0
python-dotenv>=1.0.0
```

---

## 4. 运行说明

### 4.1 快速开始

1. **克隆或创建项目目录**
```bash
mkdir ai_agents
cd ai_agents
```

2. **创建环境和安装依赖**
```bash
# LangGraph环境
python -m venv langgraph_env
source langgraph_env/bin/activate
pip install -r requirements_langgraph.txt

# Agno环境  
python -m venv agno_env
source agno_env/bin/activate
pip install -r requirements_agno.txt
```

3. **配置环境变量**
```bash
# 创建.env文件并添加API密钥
cp .env.example .env
# 编辑.env文件添加你的API密钥
```

4. **运行示例**
```bash
# 运行LangGraph示例
python langgraph_agent.py

# 运行Agno示例
python agno_agent.py
```

### 4.2 功能特性对比

| 功能特性 | LangGraph Agent | Agno Agent |
|---------|----------------|------------|
| 对话功能 | ✅ 支持多轮对话 | ✅ 支持多轮对话 |
| 工具调用 | ✅ 灵活的工具系统 | ✅ 丰富的预建工具 |
| 多步推理 | ✅ 复杂的图状推理 | ✅ 内置推理工具 |
| 任务规划 | ✅ 动态计划生成 | ✅ 智能体团队协作 |
| 搜索集成 | ✅ 多搜索引擎 | ✅ 多搜索引擎 |
| 性能 | 🔶 中等 | ✅ 高性能 |
| 学习曲线 | 🔶 较陡峭 | ✅ 相对简单 |
| 模型支持 | ✅ 多模型 | ✅ 23+模型提供商 |

### 4.3 使用建议

**选择LangGraph的情况：**
- 需要复杂的工作流控制
- 需要精细的状态管理
- 需要自定义复杂的推理路径
- 有图结构化处理需求

**选择Agno的情况：**
- 需要高性能和低延迟
- 希望快速原型开发
- 需要多智能体协作
- 希望使用预建组件

---

## 5. 扩展和定制

### 5.1 添加新工具

**LangGraph中添加工具：**
```python
@tool
def custom_tool(param: str) -> str:
    """自定义工具描述"""
    # 工具逻辑
    return "结果"

# 添加到工具列表
tools.append(custom_tool)
```

**Agno中添加工具：**
```python
class CustomTool(Tool):
    def __init__(self):
        super().__init__(name="custom_tool", description="自定义工具")
    
    def run(self, param: str) -> str:
        # 工具逻辑
        return "结果"
```

### 5.2 自定义推理模式

两个框架都支持自定义推理模式，可以根据具体需求实现特定的推理逻辑。

### 5.3 部署建议

- 使用Docker容器化部署
- 配置适当的资源限制
- 实现请求缓存机制
- 添加监控和日志记录

---

## 6. 总结

本文档提供了两个完整、可运行的AI agent实现，展示了LangGraph和Agno框架的强大功能。两个框架各有优势，可以根据具体需求选择使用。代码包含详细的中文注释和完整的使用示例，便于理解和扩展。