import os
from typing import List, Dict, Any
import json
from datetime import datetime

from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from dotenv import load_dotenv

from states.agent_state import AgentState
from tools import web_search, duckduckgo_search, get_stock_info, calculator
from utils import create_langgraph_model

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))

from model_config import ModelManager, display_available_models, ModelProvider

load_dotenv()

tools = [web_search, duckduckgo_search, get_stock_info, calculator]

class LangGraphAgent:
    """基于LangGraph的智能助手"""
    
    def __init__(self, model_key: str = "gpt-4o"):
        """
        初始化智能助手
        
        Args:
            model_key: 使用的模型键值
        """
        self.model_key = model_key

        # 检查模型是否可用
        if not ModelManager.is_model_available(model_key):
            print(f"❌ 模型 {model_key} 不可用")
            available_models = ModelManager.get_models_by_availability()
            if available_models:
                fallback_key = list(available_models.keys())[0]
                print(f"🔄 使用回退模型: {fallback_key}")
                self.model_key = fallback_key
            else:
                print("❌ 没有可用的模型，请检查API密钥配置")
                display_available_models()
                raise ValueError("没有可用的模型")
        
        # 使用模型工厂创建模型
        try:
            # 为HTTP模型设置较短的超时时间
            model_config = ModelManager.get_model_config(self.model_key)
            if model_config and model_config.provider == ModelProvider.HTTP:
                self.llm = create_langgraph_model(self.model_key, temperature=0.1, timeout=30)
                print(f"🌐 HTTP模型，设置超时时间: 30秒")
            else:
                self.llm = create_langgraph_model(self.model_key, temperature=0.1)
        except Exception as e:
            print(f"❌ 创建模型失败: {e}")
            raise
            
        self.memory = MemorySaver()
        
        # 创建工具节点
        self.tool_node = ToolNode(tools)
        
        # 构建图
        self.app = self._build_graph()
        
        # 获取模型配置信息
        model_config = ModelManager.get_model_config(self.model_key)
        
        print(f"✅ LangGraph智能助手初始化完成")
        print(f"📍 使用模型: {self.model_key}")
        if model_config:
            print(f"🏷️  模型名称: {model_config.name}")
            print(f"🏢 提供商: {model_config.provider.value}")
    
    def _build_graph(self) -> StateGraph:
        """构建LangGraph工作流图"""
        
        # 创建状态图
        workflow = StateGraph(AgentState)
        
        # 添加节点
        workflow.add_node("classifier", self._classify_task)          
        workflow.add_node("simple_chat", self._simple_chat)           
        workflow.add_node("planner", self._create_plan)              
        workflow.add_node("researcher", self._research_node)          
        workflow.add_node("analyzer", self._analysis_node)           
        workflow.add_node("reasoning", self._reasoning_node)         
        workflow.add_node("tools", self.tool_node)                   
        workflow.add_node("synthesizer", self._synthesize_results)   
        
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
            self._should_use_tools,
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
        
        # 保持LangGraph的消息格式
        # 构建完整消息列表，保持LangGraph消息格式
        system_message = SystemMessage(content=system_prompt)
        full_messages = [system_message] + messages
        
        # 检查是否是DeepSeek模型或HTTP模型，如果是，则使用不同的处理方式
        model_config = ModelManager.get_model_config(self.model_key)
        
        if model_config and (model_config.provider.value == "deepseek" or model_config.provider == ModelProvider.HTTP):
            # DeepSeek模型和HTTP模型直接处理，不使用复杂的工具调用
            response = self.llm.invoke(full_messages)
            
            # 检查是否需要工具调用
            content = response.content.lower()
            if any(keyword in content for keyword in ["计算", "搜索", "股票", "数学"]):
                # 尝试提取并执行计算
                if "计算" in content or "数学" in content:
                    try:
                        # 提取数学表达式
                        import re
                        user_message = messages[-1].content if messages else ""
                        
                        # 更强大的数学表达式提取
                        math_patterns = [
                            r'\([\d\s+\-*/]+\)\s*[\+\-\*/]\s*\d+',  # (125 + 75) * 2
                            r'\d+\s*[\+\-\*/]\s*\d+[\s\+\-\*/\d]*',  # 125 + 75 * 2
                            r'\([\d\s+\-*/]+\)',  # (125 + 75)
                        ]
                        
                        for pattern in math_patterns:
                            math_match = re.search(pattern, user_message)
                            if math_match:
                                expression = math_match.group(0).strip()
                                # 如果找到完整表达式，使用它；否则使用整个数学部分
                                if not expression:
                                    continue
                                
                                # 对于题目"(125 + 75) * 2 - 50"，提取完整表达式
                                full_match = re.search(r'([\d\s+\-*/\(\)]+)', user_message)
                                if full_match:
                                    full_expr = full_match.group(0).strip()
                                    if len(full_expr) > len(expression):
                                        expression = full_expr
                                
                                calc_result = calculator.invoke({"expression": expression})
                                response.content = f"{response.content}\n\n{calc_result}"
                                break
                    except:
                        pass
        else:
            # 其他模型使用标准的工具绑定
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
        
        # 执行研究步骤
        search_queries = self._generate_search_queries(current_task)
        search_results = []
        
        for query in search_queries[:3]:  
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
        
        return queries[:3]
    
    def _should_use_tools(self, state: AgentState) -> str:
        """决定是否使用工具"""
        messages = state.get("messages", [])
        if not messages:
            return "end"
        
        # 检查是否是DeepSeek模型或HTTP模型
        model_config = ModelManager.get_model_config(self.model_key)
        if model_config and (model_config.provider.value == "deepseek" or model_config.provider == ModelProvider.HTTP):
            # DeepSeek模型和HTTP模型不使用LangGraph的工具调用机制
            return "end"
        
        # 其他模型使用标准的tools_condition
        return tools_condition(state)
    
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
            error_msg = f"对话处理失败: {str(e)}"
            print(f"❌ 错误: {error_msg}")
            return "抱歉，处理您的请求时遇到了问题，请稍后再试。"
    
    def get_reasoning_steps(self, session_id: str = "default") -> List[str]:
        """获取推理步骤"""
        try:
            config = {"configurable": {"thread_id": session_id}}
            state = self.app.get_state(config)
            return state.values.get("reasoning_steps", [])
        except:
            return []
    
    def switch_model(self, new_model_key: str):
        """切换模型"""
        if ModelManager.is_model_available(new_model_key):
            try:
                # 创建新模型
                new_llm = create_langgraph_model(new_model_key, temperature=0.1)
                
                # 更新模型
                self.model_key = new_model_key
                self.llm = new_llm
                
                # 重新构建图
                self.app = self._build_graph()
                
                model_config = ModelManager.get_model_config(new_model_key)
                print(f"✅ 已切换到模型: {new_model_key}")
                if model_config:
                    print(f"🏷️  模型名称: {model_config.name}")
                    print(f"🏢 提供商: {model_config.provider.value}")
            except Exception as e:
                print(f"❌ 切换模型失败: {e}")
        else:
            print(f"❌ 模型 {new_model_key} 不可用")
            print("🤖 可用模型:")
            available_models = ModelManager.get_models_by_availability()
            for key, config in available_models.items():
                print(f"  - {key}: {config.name}")
    
    def list_available_models(self):
        """列出可用模型"""
        display_available_models()