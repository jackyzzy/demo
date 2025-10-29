from agno.agent import Agent
from agno.team import Team  
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.tools.yfinance import YFinanceTools
from agno.tools.reasoning import ReasoningTools

from tools import TavilySearchTool, AdvancedCalculatorTool, EnhancedStockTool
from utils import create_agno_model

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))

from model_config import ModelManager

class ResearchTeam:
    """研究团队 - 多智能体协作"""
    
    def __init__(self, model_key: str = "gpt-4o"):
        """
        初始化研究团队
        
        Args:
            model_key: 模型键值
        """
        # 使用模型工厂创建基础模型
        try:
            base_model = create_agno_model(model_key)
            print(f"✅ 研究团队使用模型: {model_key}")
        except Exception as e:
            print(f"❌ 创建模型失败: {e}")
            # fallback到第一个可用模型
            available_models = ModelManager.get_models_by_availability()
            if available_models:
                fallback_key = list(available_models.keys())[0]
                base_model = create_agno_model(fallback_key)
                print(f"🔄 研究团队回退到模型: {fallback_key}")
            else:
                raise ValueError("没有可用的模型，请检查API密钥配置")
        
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
        
        print(f"✅ Agno研究团队初始化完成，使用{model_key}模型")
    
    def research(self, topic: str) -> str:
        """执行研究任务"""
        return self.team.run(f"请深入研究: {topic}")