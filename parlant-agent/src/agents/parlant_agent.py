"""
基于Parlant的智能助手 - 简化版本
直接使用模型客户端，避免复杂的server/session架构
"""

import asyncio
from typing import List, Dict, Optional
import sys
import os
from dotenv import load_dotenv

# 添加路径
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../'))

from model_config import ModelManager, ModelProvider
from dotenv import load_dotenv

load_dotenv()


class ParlantAgent:
    """基于Parlant结构的简化智能助手 - 直接使用模型客户端"""

    def __init__(self, model_key: str = "gpt-4o"):
        """
        初始化Parlant智能助手

        Args:
            model_key: 模型配置键值
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
                raise ValueError("没有可用的模型")

        # 获取模型配置
        self.model_config = ModelManager.get_model_config(self.model_key)

        # 使用 model_factory 创建客户端（与 LangGraph/Agno 保持一致）
        from utils.model_factory import create_model_client
        self.client = create_model_client(self.model_key)

        # 会话历史
        self.chat_history: Dict[str, List[Dict]] = {}

        # 系统提示词
        self.system_prompt = """你是一个友好、专业的AI助手，能够帮助用户完成各种任务。

你具备以下能力：
1. 搜索信息：可以使用搜索引擎获取最新信息和新闻
2. 数学计算：可以进行各种数学计算和数值运算
3. 股票分析：可以获取和分析股票信息
4. 复杂推理：可以进行多步推理和深度分析

请根据用户的问题，智能地选择合适的工具来帮助回答。始终保持友好、专业的态度。"""

        model_info = self.model_config
        print(f"✅ 智能助手初始化完成！")
        print(f"📍 使用模型: {self.model_key}")
        print(f"🏷️  模型名称: {model_info.name}")
        print(f"🏢 提供商: {model_info.provider.value}")
        print(f"🔧 功能: 对话、搜索、分析、推理、金融数据")


    def _call_model(self, messages: List[Dict]) -> str:
        """
        调用模型生成回复

        Args:
            messages: 消息列表

        Returns:
            str: 模型回复
        """
        try:
            if self.model_config.provider == ModelProvider.ANTHROPIC:
                # Anthropic API格式不同
                system_msg = messages[0]["content"] if messages[0]["role"] == "system" else self.system_prompt
                user_messages = [m for m in messages if m["role"] != "system"]

                response = self.client.messages.create(
                    model=self.model_config.model_id,
                    max_tokens=4096,
                    system=system_msg,
                    messages=user_messages
                )
                return response.content[0].text

            elif self.model_config.provider == ModelProvider.HTTP:
                # HTTP模型使用 HttpModelClient（与 LangGraph/Agno 一致）
                response = self.client.chat_completion(
                    messages=messages,
                    temperature=0.7,
                    max_tokens=4096
                )
                return response.content

            else:
                # OpenAI兼容API格式
                response = self.client.chat.completions.create(
                    model=self.model_config.model_id,
                    messages=messages,
                    temperature=0.7,
                    max_tokens=4096
                )
                return response.choices[0].message.content

        except Exception as e:
            return f"模型调用失败: {str(e)}"

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
            # 初始化会话历史
            if session_id not in self.chat_history:
                self.chat_history[session_id] = []

            # 构建消息列表
            messages = [{"role": "system", "content": self.system_prompt}]

            # 添加历史消息（保留最近10轮对话）
            recent_history = self.chat_history[session_id][-20:]  # 10轮 = 20条消息
            messages.extend(recent_history)

            # 添加当前用户消息
            messages.append({"role": "user", "content": message})

            # 调用模型
            response = self._call_model(messages)

            # 保存到历史
            self.chat_history[session_id].append({"role": "user", "content": message})
            self.chat_history[session_id].append({"role": "assistant", "content": response})

            return response

        except Exception as e:
            error_msg = f"对话处理失败: {str(e)}"
            print(f"❌ {error_msg}")
            return error_msg

    def get_chat_history(self, session_id: str = "default") -> List[str]:
        """获取聊天历史"""
        history = self.chat_history.get(session_id, [])
        formatted = []
        for msg in history:
            role = "用户" if msg["role"] == "user" else "助手"
            formatted.append(f"{role}: {msg['content'][:100]}...")
        return formatted

    def clear_history(self, session_id: str = "default"):
        """清除聊天历史"""
        if session_id in self.chat_history:
            del self.chat_history[session_id]
        print(f"✅ 已清除会话 {session_id} 的历史记录")

    def switch_model(self, new_model_key: str):
        """切换模型"""
        if ModelManager.is_model_available(new_model_key):
            self.model_key = new_model_key
            self.model_config = ModelManager.get_model_config(new_model_key)

            # 使用 model_factory 重新创建客户端
            from utils.model_factory import create_model_client
            self.client = create_model_client(new_model_key)

            model_info = self.model_config
            print(f"✅ 已切换到模型: {new_model_key}")
            print(f"🏷️  模型名称: {model_info.name}")
            print(f"🏢 提供商: {model_info.provider.value}")
        else:
            print(f"❌ 模型 {new_model_key} 不可用")
            from model_config import display_available_models
            display_available_models()
