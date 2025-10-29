from typing import List, Dict
from dotenv import load_dotenv

from workflows import IntelligentWorkflow

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))

from model_config import ModelManager, display_available_models

load_dotenv()

class AgnoAgent:
    """基于Agno的智能助手"""
    
    def __init__(self, model_key: str = "gpt-4o"):
        """
        初始化Agno智能助手
        
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
                display_available_models()
                raise ValueError("没有可用的模型")
        
        # 创建工作流
        self.workflow = IntelligentWorkflow(self.model_key)
        
        # 会话历史
        self.chat_history: Dict[str, List[str]] = {}
        
        # 获取模型配置信息
        model_config = ModelManager.get_model_config(self.model_key)
        
        print(f"🎉 Agno智能助手初始化完成！")
        print(f"📍 使用模型: {self.model_key}")
        if model_config:
            print(f"🏷️  模型名称: {model_config.name}")
            print(f"🏢 提供商: {model_config.provider.value}")
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
    
    def switch_model(self, new_model_key: str):
        """切换模型"""
        if ModelManager.is_model_available(new_model_key):
            self.model_key = new_model_key
            self.workflow = IntelligentWorkflow(new_model_key)
            model_config = ModelManager.get_model_config(new_model_key)
            print(f"✅ 已切换到模型: {new_model_key}")
            if model_config:
                print(f"🏷️  模型名称: {model_config.name}")
                print(f"🏢 提供商: {model_config.provider.value}")
        else:
            print(f"❌ 模型 {new_model_key} 不可用")
            print("🤖 可用模型:")
            available_models = ModelManager.get_models_by_availability()
            for key, config in available_models.items():
                print(f"  - {key}: {config.name}")
    
    def list_available_models(self):
        """列出可用模型"""
        display_available_models()