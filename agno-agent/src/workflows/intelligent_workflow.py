from agno.agent import Agent
from agno.workflow import Workflow

from teams import ResearchTeam
from tools import AdvancedCalculatorTool
from utils import create_agno_model

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))

from model_config import ModelManager, display_available_models

class IntelligentWorkflow(Workflow):
    """智能工作流 - 任务规划和执行"""
    
    def __init__(self, model_key: str = "gpt-4o"):
        super().__init__(name="intelligent_workflow")
        
        # 保存模型键值
        self.model_key = model_key
        
        # 使用模型工厂创建模型
        try:
            model = create_agno_model(model_key)
            print(f"✅ 使用模型: {model_key}")
        except Exception as e:
            print(f"❌ 创建模型失败: {e}")
            # fallback到第一个可用模型
            available_models = ModelManager.get_models_by_availability()
            if available_models:
                fallback_key = list(available_models.keys())[0]
                model = create_agno_model(fallback_key)
                print(f"🔄 回退到模型: {fallback_key}")
            else:
                raise ValueError("没有可用的模型，请检查API密钥配置")
        
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
        self.research_team = ResearchTeam(model_key)
        
        print(f"✅ 智能工作流初始化完成")
    
    def run(self, user_request: str) -> str:
        """执行工作流"""
        return self._execute_workflow(user_request)
    
    def run_workflow(self, user_request: str = None, **kwargs) -> str:
        """兼容Agno框架的工作流执行方法"""
        # 处理不同的调用方式
        if user_request is None and kwargs:
            # 如果没有user_request，尝试从kwargs中获取
            user_request = kwargs.get('message', kwargs.get('input', kwargs.get('request', '')))
        
        if not user_request:
            return "请提供有效的用户请求"
        
        return self._execute_workflow(user_request)
    
    def _execute_workflow(self, user_request: str) -> str:
        """内部工作流执行逻辑"""
        
        # 检查是否是DeepSeek模型，如果是则使用特殊处理
        model_config = ModelManager.get_model_config(self.model_key)
        if model_config and model_config.provider.value == "deepseek":
            return self._handle_deepseek_request(user_request)
        
        # 其他模型使用标准流程
        try:
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
            
            # 正确提取响应内容
            if hasattr(classification, 'content'):
                task_type = str(classification.content).strip().lower()
            elif hasattr(classification, 'text'):
                task_type = str(classification.text).strip().lower()
            else:
                task_type = str(classification).strip().lower()
            
            # Step 2: 根据分类执行相应处理
            if task_type in ["simple_chat", "calculation"]:
                return self.chat_assistant.run(user_request)
            
            elif task_type in ["research", "financial_analysis", "complex_analysis"]:
                return self.research_team.research(user_request)
            
            else:
                # 默认使用聊天助手
                return self.chat_assistant.run(user_request)
                
        except Exception as e:
            # 如果出现角色相关错误，回退到DeepSeek特殊处理
            if "developer" in str(e) and "unknown variant" in str(e):
                return self._handle_deepseek_request(user_request)
            else:
                raise e
    
    def _handle_deepseek_request(self, user_request: str) -> str:
        """处理DeepSeek模型的请求，避免角色兼容性问题"""
        import requests
        import json
        import time
        
        # 获取DeepSeek API配置
        model_config = ModelManager.get_model_config(self.model_key)
        api_key = model_config.api_key if model_config else None
        base_url = model_config.base_url if model_config else "https://api.deepseek.com"
        model_id = model_config.model_id if model_config else "deepseek-chat"
        temperature = model_config.temperature if model_config else 0.1
        
        if not api_key:
            return "DeepSeek API密钥未配置"
        
        # 确保base_url以正确格式结尾
        if not base_url.endswith('/chat/completions'):
            if base_url.endswith('/'):
                base_url += 'chat/completions'
            else:
                base_url += '/chat/completions'
        
        # 直接调用DeepSeek API，增加重试逻辑
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {api_key}'
        }
        
        # 根据查询类型调整参数
        is_complex_query = any(keyword in user_request.lower() for keyword in 
                              ["分析", "股票", "趋势", "投资", "风险", "研究", "调研"])
        
        # 构建请求，使用配置中的参数
        data = {
            "model": model_id,
            "messages": [
                {"role": "system", "content": "你是一个专业的AI助手，可以回答问题、进行计算、分析数据、提供投资建议等。请提供准确、详细的回答。"},
                {"role": "user", "content": user_request}
            ],
            "temperature": temperature,
            "max_tokens": 4000 if is_complex_query else 2000  # 复杂查询使用更多tokens
        }
        
        # 重试逻辑
        max_retries = 2
        base_timeout = 60 if is_complex_query else 30  # 复杂查询使用更长的超时
        
        for attempt in range(max_retries + 1):
            try:
                timeout_value = base_timeout + (attempt * 30)  # 每次重试增加30秒
                
                if attempt > 0:
                    print(f"🔄 DeepSeek重试中... (第{attempt}次，超时设置: {timeout_value}秒)")
                    time.sleep(2)  # 重试前短暂等待
                
                response = requests.post(
                    base_url,
                    headers=headers,
                    json=data,
                    timeout=timeout_value
                )
                
                if response.status_code == 200:
                    result = response.json()
                    if 'choices' in result and len(result['choices']) > 0:
                        answer = result['choices'][0]['message']['content']
                        
                        # 对于数学计算，尝试提供更精确的结果
                        if any(keyword in user_request for keyword in ["计算", "数学", "+", "-", "*", "/", "="]):
                            import re
                            # 简单的数学表达式提取和计算
                            math_expr = re.search(r'[\d\s\+\-\*/\(\)]+', user_request)
                            if math_expr:
                                expr = math_expr.group(0).strip()
                                try:
                                    # 安全计算
                                    if all(c in '0123456789+-*/(). ' for c in expr):
                                        calc_result = eval(expr)
                                        answer += f"\n\n计算结果: {expr} = {calc_result}"
                                except:
                                    pass
                        
                        # 成功时显示处理信息
                        if attempt > 0:
                            print(f"✅ DeepSeek重试成功!")
                        
                        return answer
                    else:
                        return "DeepSeek API返回格式错误"
                        
                elif response.status_code == 429:  # 速率限制错误，需要重试
                    if attempt < max_retries:
                        print(f"⏳ DeepSeek API速率限制，等待重试...")
                        time.sleep(5)  # 等待更长时间
                        continue
                    else:
                        return f"DeepSeek API速率限制，请稍后重试"
                        
                else:
                    return f"DeepSeek API调用失败: {response.status_code} - {response.text}"
                    
            except requests.exceptions.Timeout:
                if attempt < max_retries:
                    print(f"⏰ DeepSeek API超时，准备重试... (已等待{timeout_value}秒)")
                    continue
                else:
                    return f"DeepSeek API请求超时，已重试{max_retries}次。请检查网络连接或尝试简化查询内容。"
                    
            except requests.exceptions.ConnectionError:
                if attempt < max_retries:
                    print(f"🔌 DeepSeek API连接错误，准备重试...")
                    time.sleep(3)
                    continue
                else:
                    return f"DeepSeek API连接失败，请检查网络连接。"
                    
            except Exception as e:
                if attempt < max_retries:
                    print(f"⚠️  DeepSeek请求异常，准备重试: {str(e)}")
                    time.sleep(2)
                    continue
                else:
                    return f"DeepSeek处理失败: {str(e)}"
        
        return "DeepSeek处理失败，已达到最大重试次数"