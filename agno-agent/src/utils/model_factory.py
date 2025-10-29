"""
Agno模型工厂
用于创建不同提供商的模型实例
"""

from typing import Any, Optional
import os

from agno.models.openai import OpenAIChat
from agno.models.anthropic import Claude

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../'))

from model_config import ModelManager, ModelConfig, ModelProvider

# HTTP模型适配器
try:
    sys.path.append(os.path.join(os.path.dirname(__file__), '../../../'))
    from http_agno_adapter import create_http_agno_model
    HTTP_ADAPTER_AVAILABLE = True
except ImportError:
    HTTP_ADAPTER_AVAILABLE = False
    print("⚠️ HTTP模型适配器不可用，请检查依赖")


class DeepSeekChatWrapper(OpenAIChat):
    """DeepSeek聊天模型包装器，继承并重写OpenAIChat"""
    
    def __init__(self, **kwargs):
        # 确保正确初始化父类
        super().__init__(**kwargs)
        # 在父类初始化完成后再包装client
        if hasattr(self, 'client') and self.client:
            self._original_client = self.client
            # 重写client来拦截API调用
            self.client = DeepSeekAPIClient(self._original_client)

class DeepSeekAPIClient:
    """DeepSeek API客户端包装器，处理消息角色映射"""
    
    def __init__(self, original_client):
        self.original_client = original_client
        # 代理所有属性到原始客户端
        for attr in dir(original_client):
            if not attr.startswith('_') and attr != 'chat':
                setattr(self, attr, getattr(original_client, attr))
    
    @property
    def chat(self):
        return DeepSeekChatCompletions(self.original_client.chat)
    
    def __getattr__(self, name):
        return getattr(self.original_client, name)

class DeepSeekChatCompletions:
    """处理聊天补全的角色映射"""
    
    def __init__(self, original_chat):
        self.original_chat = original_chat
        # 代理completions属性
        self.completions = DeepSeekCompletionsWrapper(original_chat.completions)
    
    def __getattr__(self, name):
        return getattr(self.original_chat, name)

class DeepSeekCompletionsWrapper:
    """包装completions.create方法以处理角色映射"""
    
    def __init__(self, original_completions):
        self.original_completions = original_completions
    
    def create(self, messages=None, **kwargs):
        """创建聊天补全，处理消息角色映射"""
        if messages:
            # 处理消息列表，映射不支持的角色
            processed_messages = []
            has_developer_role = False
            
            for msg in messages:
                if isinstance(msg, dict):
                    processed_msg = msg.copy()
                    if 'role' in processed_msg:
                        if processed_msg['role'] == 'developer':
                            processed_msg['role'] = 'system'
                            has_developer_role = True
                        elif processed_msg['role'] not in ['system', 'user', 'assistant', 'tool']:
                            processed_msg['role'] = 'user'
                    processed_messages.append(processed_msg)
                else:
                    processed_messages.append(msg)
            
            # Debug输出
            if has_developer_role:
                print("🔧 DeepSeek: 映射了developer角色到system角色")
            
            kwargs['messages'] = processed_messages
        
        return self.original_completions.create(**kwargs)
    
    def __getattr__(self, name):
        return getattr(self.original_completions, name)


class AgnoModelFactory:
    """Agno框架模型工厂"""
    
    @staticmethod
    def create_model(model_key: str, **kwargs) -> Any:
        """
        创建模型实例
        
        Args:
            model_key: 模型配置键值
            **kwargs: 额外的模型参数
            
        Returns:
            模型实例
        """
        config = ModelManager.get_model_config(model_key)
        if not config:
            raise ValueError(f"未找到模型配置: {model_key}")
        
        # 检查模型是否可用
        if not ModelManager.is_model_available(model_key):
            raise ValueError(f"模型 {model_key} 不可用，请检查API密钥配置")
        
        # 合并配置参数
        model_params = {
            'id': config.model_id,
            'temperature': config.temperature,
            'max_tokens': config.max_tokens,
            **kwargs  # 允许覆盖默认参数
        }
        
        # 根据提供商创建相应的模型实例
        if config.provider == ModelProvider.OPENAI:
            return AgnoModelFactory._create_openai_model(config, model_params)
        
        elif config.provider == ModelProvider.ANTHROPIC:
            return AgnoModelFactory._create_anthropic_model(config, model_params)
        
        elif config.provider == ModelProvider.GROQ:
            return AgnoModelFactory._create_groq_model(config, model_params)
        
        elif config.provider == ModelProvider.OLLAMA:
            return AgnoModelFactory._create_ollama_model(config, model_params)
        
        elif config.provider == ModelProvider.DEEPSEEK:
            return AgnoModelFactory._create_deepseek_model(config, model_params)
        
        elif config.provider == ModelProvider.HTTP:
            return AgnoModelFactory._create_http_model(config, model_params)
        
        else:
            raise ValueError(f"不支持的模型提供商: {config.provider}")
    
    @staticmethod
    def _create_openai_model(config: ModelConfig, params: dict) -> OpenAIChat:
        """创建OpenAI模型"""
        # 使用配置中的API密钥
        if config.api_key:
            params['api_key'] = config.api_key
        
        return OpenAIChat(**params)
    
    @staticmethod
    def _create_anthropic_model(config: ModelConfig, params: dict) -> Claude:
        """创建Anthropic模型"""
        # 使用配置中的API密钥
        if config.api_key:
            params['api_key'] = config.api_key
        
        return Claude(**params)
    
    @staticmethod
    def _create_groq_model(config: ModelConfig, params: dict) -> Any:
        """创建Groq模型（通过OpenAI兼容接口）"""
        try:
            # Groq使用OpenAI兼容的接口
            from agno.models.openai import OpenAIChat
            
            # 设置Groq特定参数
            groq_params = params.copy()
            groq_params['base_url'] = "https://api.groq.com/openai/v1"
            
            # 使用配置中的API密钥
            if config.api_key:
                groq_params['api_key'] = config.api_key
            
            return OpenAIChat(**groq_params)
            
        except ImportError:
            raise ValueError("创建Groq模型失败，请确认Agno版本支持")
    
    @staticmethod
    def _create_ollama_model(config: ModelConfig, params: dict) -> Any:
        """创建Ollama本地模型"""
        try:
            from agno.models.ollama import Ollama
            
            # 设置Ollama参数
            ollama_params = params.copy()
            if config.base_url:
                ollama_params['host'] = config.base_url
            
            # Ollama不需要API密钥
            ollama_params.pop('api_key', None)
            
            return Ollama(**ollama_params)
            
        except ImportError:
            # 如果没有Ollama支持，尝试通过OpenAI接口
            print("⚠️ 未找到Ollama模块，尝试通过OpenAI兼容接口连接")
            
            ollama_params = params.copy()
            ollama_params['base_url'] = config.base_url or "http://localhost:11434/v1"
            ollama_params['api_key'] = "ollama"  # Ollama不验证API密钥
            
            from agno.models.openai import OpenAIChat
            return OpenAIChat(**ollama_params)
    
    @staticmethod
    def _create_deepseek_model(config: ModelConfig, params: dict) -> Any:
        """创建DeepSeek模型（通过OpenAI兼容接口）"""
        # DeepSeek使用OpenAI兼容的接口
        deepseek_params = params.copy()
        deepseek_params['base_url'] = config.base_url or "https://api.deepseek.com"
        
        # 使用配置中的API密钥
        if config.api_key:
            deepseek_params['api_key'] = config.api_key
        
        # 为DeepSeek模型添加角色映射功能
        model = OpenAIChat(**deepseek_params)
        
        # 设置角色映射
        if not hasattr(model, '_role_map'):
            model._role_map = {
                'developer': 'system',  # 映射developer角色到system
                'function': 'assistant',  # 映射function角色到assistant
            }
        
        return model
    
    @staticmethod
    def _create_http_model(config: ModelConfig, params: dict) -> Any:
        """创建HTTP模型"""
        if not HTTP_ADAPTER_AVAILABLE:
            raise ValueError("HTTP模型适配器不可用，请检查依赖安装")
        
        if not config.base_url:
            raise ValueError(f"HTTP模型 {config.name} 缺少base_url配置")
        
        if not config.api_key:
            raise ValueError(f"HTTP模型 {config.name} 的API密钥未配置: {config.api_key_env}")
        
        # 提取参数
        http_params = {
            'url': config.base_url,
            'api_key': config.api_key,
            'model_id': config.model_id,
            'vendor': config.vendor or "generic",
            'temperature': params.get('temperature', config.temperature),
            'max_tokens': params.get('max_tokens', config.max_tokens)
        }
        
        # 添加自定义头部
        if config.headers:
            http_params['headers'] = config.headers
        
        import logging
        logging.info(f"🌐 创建HTTP模型: {config.name} (供应商: {config.vendor})")
        logging.info(f"📡 端点: {config.base_url}")
        
        # 创建HTTP Agno模型
        return create_http_agno_model(**http_params)
    
    @staticmethod
    def list_supported_models() -> list:
        """获取支持的模型列表"""
        return ModelManager.list_available_models()
    
    @staticmethod
    def get_available_models() -> dict:
        """获取当前可用的模型"""
        return ModelManager.get_models_by_availability()


# 便捷函数
def create_agno_model(model_key: str = "gpt-4o", **kwargs) -> Any:
    """
    便捷函数：创建Agno模型实例
    
    Args:
        model_key: 模型键值，默认gpt-4o
        **kwargs: 额外参数
    
    Returns:
        模型实例
    """
    return AgnoModelFactory.create_model(model_key, **kwargs)


if __name__ == "__main__":
    # 测试模型工厂
    print("🧪 测试Agno模型工厂")
    
    # 显示可用模型
    available = AgnoModelFactory.get_available_models()
    print(f"可用模型数量: {len(available)}")
    
    for key, config in available.items():
        print(f"- {key}: {config.name}")
    
    # 测试创建模型（选择第一个可用模型）
    if available:
        test_model_key = list(available.keys())[0]
        print(f"\n🔧 测试创建模型: {test_model_key}")
        
        try:
            model = create_agno_model(test_model_key)
            print(f"✅ 成功创建模型: {model}")
        except Exception as e:
            print(f"❌ 创建模型失败: {e}")
    else:
        print("❌ 没有可用的模型进行测试")