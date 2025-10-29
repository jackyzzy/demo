"""
Parlant模型工厂
支持多模型提供商的统一创建接口
"""

import os
import sys
from typing import Optional, Dict, Any

# 添加根目录到Python路径以访问model_config
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../'))

from model_config import ModelManager, ModelConfig, ModelProvider

# HTTP模型适配器
try:
    from http_parlant_adapter import create_http_parlant_client
    HTTP_ADAPTER_AVAILABLE = True
except ImportError:
    HTTP_ADAPTER_AVAILABLE = False
    print("⚠️ HTTP模型适配器不可用，请检查依赖")


def create_parlant_model(
    model_key: str,
    temperature: float = 0.1,
    **kwargs
) -> Dict[str, Any]:
    """
    为Parlant创建模型配置

    Parlant使用配置字典而非直接的模型实例

    Args:
        model_key: 模型键值（如 "gpt-4o", "claude-3.5-sonnet"）
        temperature: 温度参数
        **kwargs: 其他模型参数

    Returns:
        Dict[str, Any]: Parlant模型配置字典

    Raises:
        ValueError: 如果模型不可用
    """
    # 获取模型配置
    config = ModelManager.get_model_config(model_key)

    if not config:
        raise ValueError(f"未找到模型配置: {model_key}")

    if not config.is_available:
        raise ValueError(f"模型不可用（请检查API密钥）: {model_key}")

    # 根据提供商创建配置
    if config.provider == ModelProvider.OPENAI:
        return _create_openai_config(config, temperature, **kwargs)
    elif config.provider == ModelProvider.ANTHROPIC:
        return _create_anthropic_config(config, temperature, **kwargs)
    elif config.provider == ModelProvider.DEEPSEEK:
        return _create_deepseek_config(config, temperature, **kwargs)
    elif config.provider == ModelProvider.HTTP:
        return _create_http_config(config, temperature, **kwargs)
    elif config.provider == ModelProvider.GROQ:
        return _create_groq_config(config, temperature, **kwargs)
    else:
        raise ValueError(f"不支持的模型提供商: {config.provider}")


def _create_openai_config(
    config: ModelConfig,
    temperature: float,
    **kwargs
) -> Dict[str, Any]:
    """创建OpenAI模型配置"""
    model_config = {
        "provider": "openai",
        "model": config.model_id,
        "api_key": config.api_key,
        "temperature": temperature,
        "max_tokens": kwargs.get("max_tokens", config.max_tokens),
    }

    if config.base_url:
        model_config["base_url"] = config.base_url

    return model_config


def _create_anthropic_config(
    config: ModelConfig,
    temperature: float,
    **kwargs
) -> Dict[str, Any]:
    """创建Anthropic模型配置"""
    model_config = {
        "provider": "anthropic",
        "model": config.model_id,
        "api_key": config.api_key,
        "temperature": temperature,
        "max_tokens": kwargs.get("max_tokens", config.max_tokens),
    }

    return model_config


def _create_deepseek_config(
    config: ModelConfig,
    temperature: float,
    **kwargs
) -> Dict[str, Any]:
    """创建DeepSeek模型配置（OpenAI兼容）"""
    model_config = {
        "provider": "openai",  # DeepSeek使用OpenAI兼容API
        "model": config.model_id,
        "api_key": config.api_key,
        "base_url": config.base_url or "https://api.deepseek.com",
        "temperature": temperature,
        "max_tokens": kwargs.get("max_tokens", config.max_tokens),
    }

    return model_config


def _create_http_config(
    config: ModelConfig,
    temperature: float,
    **kwargs
) -> Dict[str, Any]:
    """创建HTTP模型配置（OpenAI兼容）"""
    model_config = {
        "provider": "openai",  # HTTP模型使用OpenAI兼容API
        "model": config.model_id,
        "api_key": config.api_key,
        "base_url": config.base_url,
        "temperature": temperature,
        "max_tokens": kwargs.get("max_tokens", config.max_tokens),
    }

    if config.headers:
        model_config["headers"] = config.headers

    return model_config


def _create_groq_config(
    config: ModelConfig,
    temperature: float,
    **kwargs
) -> Dict[str, Any]:
    """创建Groq模型配置（OpenAI兼容）"""
    model_config = {
        "provider": "openai",  # Groq使用OpenAI兼容API
        "model": config.model_id,
        "api_key": config.api_key,
        "base_url": "https://api.groq.com/openai/v1",
        "temperature": temperature,
        "max_tokens": kwargs.get("max_tokens", config.max_tokens),
    }

    return model_config


def get_model_info(model_key: str) -> Optional[Dict[str, str]]:
    """
    获取模型信息

    Args:
        model_key: 模型键值

    Returns:
        Optional[Dict[str, str]]: 模型信息字典
    """
    config = ModelManager.get_model_config(model_key)
    if config:
        return config.get_model_info()
    return None


def create_model_client(model_key: str, **kwargs) -> Any:
    """
    创建模型客户端实例（用于Parlant Agent）

    这是与 LangGraph/Agno 的 model_factory 保持一致的接口。
    返回配置好的客户端实例（OpenAI 或 Anthropic），可以直接使用。

    Args:
        model_key: 模型键值（如 "gpt-4o", "claude-3.5-sonnet", "http-deepseek-r1-huawei"）
        **kwargs: 额外的配置参数

    Returns:
        客户端实例 (OpenAI 或 Anthropic)

    Raises:
        ValueError: 如果模型不存在或不可用

    Example:
        >>> client = create_model_client("gpt-4o")
        >>> response = client.chat.completions.create(
        ...     model="gpt-4o",
        ...     messages=[{"role": "user", "content": "Hello"}]
        ... )
    """
    # 获取模型配置
    config = ModelManager.get_model_config(model_key)

    if not config:
        raise ValueError(f"未找到模型配置: {model_key}")

    if not config.is_available:
        raise ValueError(f"模型不可用（请检查API密钥）: {model_key}")

    # 根据提供商创建客户端实例
    if config.provider == ModelProvider.OPENAI:
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("请安装openai: pip install openai")

        return OpenAI(api_key=config.api_key)

    elif config.provider == ModelProvider.ANTHROPIC:
        try:
            from anthropic import Anthropic
        except ImportError:
            raise ImportError("请安装anthropic: pip install anthropic")

        return Anthropic(api_key=config.api_key)

    elif config.provider == ModelProvider.DEEPSEEK:
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("请安装openai: pip install openai")

        return OpenAI(
            api_key=config.api_key,
            base_url=config.base_url or "https://api.deepseek.com"
        )

    elif config.provider == ModelProvider.GROQ:
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("请安装openai: pip install openai")

        return OpenAI(
            api_key=config.api_key,
            base_url="https://api.groq.com/openai/v1"
        )

    elif config.provider == ModelProvider.OLLAMA:
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("请安装openai: pip install openai")

        return OpenAI(
            api_key="ollama",  # Ollama不验证API密钥
            base_url=config.base_url or "http://localhost:11434/v1"
        )

    elif config.provider == ModelProvider.HTTP:
        if not HTTP_ADAPTER_AVAILABLE:
            raise ValueError("HTTP模型适配器不可用，请检查http_parlant_adapter.py")

        if not config.base_url:
            raise ValueError(f"HTTP模型 {config.name} 缺少base_url配置")

        if not config.api_key:
            raise ValueError(f"HTTP模型 {config.name} 的API密钥未配置: {config.api_key_env}")

        import logging
        logging.info(f"🌐 创建HTTP模型: {config.name} (供应商: {config.vendor})")
        logging.info(f"📡 端点: {config.base_url}")

        return create_http_parlant_client(
            url=config.base_url,
            api_key=config.api_key,
            model_id=config.model_id,
            vendor=config.vendor or "generic"
        )

    else:
        raise ValueError(f"不支持的模型提供商: {config.provider}")
