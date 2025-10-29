"""
HTTP模型的Parlant适配器
为Parlant Agent提供HTTP模型支持
"""

from typing import Dict, Optional
from http_model_client import HttpModelClient, create_http_client


def create_http_parlant_client(
    url: str,
    api_key: str,
    model_id: str,
    vendor: str = "generic",
    headers: Optional[Dict[str, str]] = None,
    timeout: int = 60,
    max_retries: int = 3,
    **kwargs
) -> HttpModelClient:
    """
    创建HTTP模型客户端（用于Parlant）

    与 LangGraph/Agno 保持一致，返回 HttpModelClient 实例。
    HttpModelClient 直接向指定 URL 发送请求，不会修改路径。

    Args:
        url: API端点URL（完整路径，包含 /chat/completions）
        api_key: API密钥
        model_id: 模型ID
        vendor: 供应商名称（如：huawei, alibaba等）
        headers: 额外的HTTP头部
        timeout: 请求超时时间（秒）
        max_retries: 最大重试次数
        **kwargs: 其他参数

    Returns:
        HttpModelClient: HTTP模型客户端实例

    Example:
        >>> client = create_http_parlant_client(
        ...     url="https://api.example.com/v1/chat/completions",
        ...     api_key="your-api-key",
        ...     model_id="model-name",
        ...     vendor="huawei"
        ... )
        >>> response = client.chat_completion(
        ...     messages=[{"role": "user", "content": "Hello"}]
        ... )
        >>> print(response.content)
    """
    # 使用统一的 HttpModelClient（与 LangGraph/Agno 一致）
    return create_http_client(
        url=url,
        api_key=api_key,
        model_id=model_id,
        vendor=vendor,
        headers=headers,
        timeout=timeout,
        max_retries=max_retries
    )


if __name__ == "__main__":
    # 测试HTTP Parlant适配器
    print("🧪 测试HTTP Parlant适配器")

    # 示例配置
    test_config = {
        "url": "https://api.example.com/v1/chat/completions",
        "api_key": "test-key",
        "model_id": "test-model",
        "vendor": "generic"
    }

    try:
        # 创建HTTP客户端
        http_client = create_http_parlant_client(**test_config)
        print(f"✅ 成功创建HTTP Parlant客户端")
        print(f"🏷️  客户端类型: {type(http_client).__name__}")
        print(f"📡 URL: {http_client.url}")
        print(f"🏢 供应商: {http_client.vendor}")
        print(f"🆔 模型ID: {http_client.model_id}")

    except Exception as e:
        print(f"❌ 测试失败: {e}")
