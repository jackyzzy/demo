"""
LangGraph模型工厂
用于创建不同提供商的模型实例
"""

from typing import Any, Optional
import os

from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../'))

from model_config import ModelManager, ModelConfig, ModelProvider

# HTTP模型适配器
try:
    sys.path.append(os.path.join(os.path.dirname(__file__), '../../../'))
    from http_langchain_adapter import create_http_chat_model
    HTTP_ADAPTER_AVAILABLE = True
except ImportError:
    HTTP_ADAPTER_AVAILABLE = False
    print("⚠️ HTTP模型适配器不可用，请检查依赖")


class LangGraphModelFactory:
    """LangGraph框架模型工厂"""
    
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
            'model': config.model_id,
            'temperature': config.temperature,
            'max_tokens': config.max_tokens,
            **kwargs  # 允许覆盖默认参数
        }
        
        # 根据提供商创建相应的模型实例
        if config.provider == ModelProvider.OPENAI:
            return LangGraphModelFactory._create_openai_model(config, model_params)
        
        elif config.provider == ModelProvider.ANTHROPIC:
            return LangGraphModelFactory._create_anthropic_model(config, model_params)
        
        elif config.provider == ModelProvider.GROQ:
            return LangGraphModelFactory._create_groq_model(config, model_params)
        
        elif config.provider == ModelProvider.OLLAMA:
            return LangGraphModelFactory._create_ollama_model(config, model_params)
        
        elif config.provider == ModelProvider.DEEPSEEK:
            return LangGraphModelFactory._create_deepseek_model(config, model_params)
        
        elif config.provider == ModelProvider.HTTP:
            return LangGraphModelFactory._create_http_model(config, model_params)
        
        else:
            raise ValueError(f"不支持的模型提供商: {config.provider}")
    
    @staticmethod
    def _create_openai_model(config: ModelConfig, params: dict) -> ChatOpenAI:
        """创建OpenAI模型"""
        # 使用配置中的API密钥
        if config.api_key:
            params['api_key'] = config.api_key
        
        return ChatOpenAI(**params)
    
    @staticmethod
    def _create_anthropic_model(config: ModelConfig, params: dict) -> ChatAnthropic:
        """创建Anthropic模型"""
        # 使用配置中的API密钥
        if config.api_key:
            params['anthropic_api_key'] = config.api_key
        
        return ChatAnthropic(**params)
    
    @staticmethod
    def _create_groq_model(config: ModelConfig, params: dict) -> ChatOpenAI:
        """创建Groq模型（通过OpenAI兼容接口）"""
        # Groq使用OpenAI兼容的接口
        groq_params = params.copy()
        groq_params['base_url'] = "https://api.groq.com/openai/v1"
        
        # 使用配置中的API密钥
        if config.api_key:
            groq_params['api_key'] = config.api_key
        
        return ChatOpenAI(**groq_params)
    
    @staticmethod
    def _create_ollama_model(config: ModelConfig, params: dict) -> ChatOpenAI:
        """创建Ollama本地模型"""
        # Ollama通过OpenAI兼容接口
        ollama_params = params.copy()
        ollama_params['base_url'] = config.base_url or "http://localhost:11434/v1"
        ollama_params['api_key'] = "ollama"  # Ollama不验证API密钥，但需要提供
        
        return ChatOpenAI(**ollama_params)
    
    @staticmethod
    def _create_deepseek_model(config: ModelConfig, params: dict) -> ChatOpenAI:
        """创建DeepSeek模型"""
        # DeepSeek使用OpenAI兼容的接口
        deepseek_params = params.copy()
        deepseek_params['base_url'] = config.base_url or "https://api.deepseek.com"
        
        # 使用配置中的API密钥
        if config.api_key:
            deepseek_params['api_key'] = config.api_key
        
        # 对于自定义部署的DeepSeek-R1，创建兼容的ChatOpenAI实例
        if "custom" in config.model_id.lower() or (config.base_url and "modelarts" in config.base_url):
            # 移除可能不支持的参数
            deepseek_params.pop('max_completion_tokens', None)
            
            # 创建ChatOpenAI实例并禁用max_completion_tokens
            chat_model = ChatOpenAI(**deepseek_params)
            
            # 重写_get_request_payload方法来阻止max_completion_tokens的使用
            original_get_request_payload = chat_model._get_request_payload
            
            def custom_get_request_payload(input_, **kwargs):
                payload = original_get_request_payload(input_, **kwargs)
                # 移除max_completion_tokens，使用max_tokens
                if 'max_completion_tokens' in payload:
                    max_val = payload.pop('max_completion_tokens')
                    payload['max_tokens'] = max_val
                return payload
            
            chat_model._get_request_payload = custom_get_request_payload
            return chat_model
        
        return ChatOpenAI(**deepseek_params)
    
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
        
        # 创建HTTP聊天模型
        return create_http_chat_model(**http_params)
    
    @staticmethod
    def list_supported_models() -> list:
        """获取支持的模型列表"""
        return ModelManager.list_available_models()
    
    @staticmethod
    def get_available_models() -> dict:
        """获取当前可用的模型"""
        return ModelManager.get_models_by_availability()


# 便捷函数
def create_langgraph_model(model_key: str = "gpt-4o", **kwargs) -> Any:
    """
    便捷函数：创建LangGraph模型实例
    
    Args:
        model_key: 模型键值，默认gpt-4o
        **kwargs: 额外参数
    
    Returns:
        模型实例
    """
    return LangGraphModelFactory.create_model(model_key, **kwargs)


if __name__ == "__main__":
    # 测试模型工厂
    print("🧪 测试LangGraph模型工厂")
    
    # 显示可用模型
    available = LangGraphModelFactory.get_available_models()
    print(f"可用模型数量: {len(available)}")
    
    for key, config in available.items():
        print(f"- {key}: {config.name}")
    
    # 测试创建模型（选择第一个可用模型）
    if available:
        test_model_key = list(available.keys())[0]
        print(f"\n🔧 测试创建模型: {test_model_key}")
        
        try:
            model = create_langgraph_model(test_model_key)
            print(f"✅ 成功创建模型: {model}")
        except Exception as e:
            print(f"❌ 创建模型失败: {e}")
    else:
        print("❌ 没有可用的模型进行测试")