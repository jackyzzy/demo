"""
HTTP模型的Agno适配器
将HTTP模型客户端适配为Agno兼容的模型
"""

from typing import Any, Dict, List, Optional, Iterator
from http_model_client import HttpModelClient, create_http_client, HttpModelResponse


class HttpAgnoModel:
    """HTTP模型的Agno模型适配器"""
    
    def __init__(self,
                 url: str,
                 api_key: str,
                 id: str,  # Agno使用id而不是model_id
                 vendor: str = "generic",
                 headers: Optional[Dict[str, str]] = None,
                 temperature: float = 0.7,
                 max_tokens: Optional[int] = None,
                 timeout: int = 60,
                 max_retries: int = 3,
                 **kwargs):
        """
        初始化HTTP Agno模型
        
        Args:
            url: API端点URL
            api_key: API密钥
            id: 模型ID（Agno参数）
            vendor: 供应商名称
            headers: 额外的HTTP头部
            temperature: 温度参数
            max_tokens: 最大令牌数
            timeout: 请求超时时间
            max_retries: 最大重试次数
            **kwargs: 其他参数
        """
        # Agno模型的标准属性
        self.id = id
        self.name = f"HTTP_{vendor}_{id}"
        self.provider = "http"
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        # HTTP客户端相关属性
        self.url = url
        self.api_key = api_key
        self.vendor = vendor
        self.headers = headers
        self.timeout = timeout
        self.max_retries = max_retries
        
        # 创建HTTP客户端
        self.client = create_http_client(
            url=url,
            api_key=api_key,
            model_id=id,
            vendor=vendor,
            headers=headers,
            timeout=timeout,
            max_retries=max_retries
        )
        
        # Agno兼容的属性
        self.response_model = None
        self.structured_outputs = False
        
        # Agno消息角色配置
        self.assistant_message_role = "assistant"
        self.tool_message_role = "tool"
        self.role_map = None  # 通常为None，除非特别指定
        self.default_role_map = {
            'system': 'developer', 
            'user': 'user', 
            'assistant': 'assistant', 
            'tool': 'tool', 
            'model': 'assistant'
        }
        
        # 其他Agno标准属性
        self.instructions = None
        self.name = f"HttpAgnoModel_{vendor}"
        self.provider = "HTTP"
        
        # 存储其他配置
        self._config = kwargs
    
    def _convert_agno_messages_to_http(self, messages: List[Any]) -> List[Dict[str, str]]:
        """
        将Agno消息格式转换为HTTP API格式
        
        Args:
            messages: Agno消息列表
            
        Returns:
            HTTP API格式的消息列表
        """
        http_messages = []
        
        for message in messages:
            if hasattr(message, 'role') and hasattr(message, 'content'):
                # 处理Agno Message对象
                role = str(message.role)
                content = str(message.content)
                
                # 角色映射
                if role == "developer":
                    role = "system"
                elif role not in ["system", "user", "assistant", "tool"]:
                    role = "user"
                
                http_messages.append({
                    "role": role,
                    "content": content
                })
            elif isinstance(message, dict):
                # 处理字典格式的消息
                role = message.get("role", "user")
                content = message.get("content", str(message))
                
                # 角色映射
                if role == "developer":
                    role = "system"
                elif role not in ["system", "user", "assistant", "tool"]:
                    role = "user"
                
                http_messages.append({
                    "role": role,
                    "content": content
                })
            else:
                # 处理字符串或其他格式
                http_messages.append({
                    "role": "user",
                    "content": str(message)
                })
        
        return http_messages
    
    def invoke(self, messages: List[Any], **kwargs) -> Any:
        """
        Agno标准的invoke方法
        
        Args:
            messages: 输入消息
            **kwargs: 额外参数
            
        Returns:
            模型响应
        """
        try:
            # 移除不能序列化的参数（如tools对象）
            clean_kwargs = {}
            for key, value in kwargs.items():
                if key in ['temperature', 'max_tokens', 'stream']:
                    clean_kwargs[key] = value
                # 跳过tools等复杂对象
            
            # 转换消息格式
            http_messages = self._convert_agno_messages_to_http(messages)
            
            # 准备参数
            temperature = clean_kwargs.get("temperature", self.temperature)
            max_tokens = clean_kwargs.get("max_tokens", self.max_tokens)
            
            # 调用HTTP客户端
            response = self.client.chat_completion(
                messages=http_messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            # 创建Agno兼容的响应对象
            return HttpAgnoResponse(
                content=response.content,
                finish_reason=response.finish_reason,
                usage=response.usage,
                model=response.model or self.id
            )
            
        except Exception as e:
            # 返回错误响应，但不抛出异常
            return HttpAgnoResponse(
                content=f"HTTP模型调用失败: {str(e)}",
                finish_reason="error", 
                usage={},
                model=self.id
            )
    
    def stream(self, messages: List[Any], **kwargs) -> Iterator[Any]:
        """
        流式调用方法
        
        Args:
            messages: 输入消息
            **kwargs: 额外参数
            
        Yields:
            流式响应片段
        """
        # 转换消息格式
        http_messages = self._convert_agno_messages_to_http(messages)
        
        # 准备参数
        temperature = kwargs.get("temperature", self.temperature)
        max_tokens = kwargs.get("max_tokens", self.max_tokens)
        
        try:
            # 调用流式HTTP客户端
            for chunk in self.client.stream_chat_completion(
                messages=http_messages,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs
            ):
                yield HttpAgnoResponse(
                    content=chunk.content,
                    finish_reason=None,
                    usage={},
                    model=self.id
                )
                
        except Exception as e:
            # 生成错误响应
            yield HttpAgnoResponse(
                content=f"HTTP流式调用失败: {str(e)}",
                finish_reason="error",
                usage={},
                model=self.id
            )
    
    def get_instructions_for_model(self, tools: Optional[List[Any]] = None) -> Optional[List[str]]:
        """
        获取模型的指令（Agno兼容方法）
        
        Args:
            tools: 可选的工具列表
            
        Returns:
            指令列表或None
        """
        # HTTP模型通常不需要特殊指令，返回None
        # 这与OpenAI模型的默认行为一致
        return None
    
    def get_system_message_for_model(self, tools: Optional[List[Any]] = None) -> Optional[str]:
        """
        获取模型的系统消息（Agno兼容方法）
        
        Args:
            tools: 可选的工具列表
            
        Returns:
            系统消息字符串或None
        """
        # HTTP模型通常不需要特殊的系统消息，返回None
        # 这与OpenAI模型的默认行为一致
        return None
    
    def get_client(self):
        """
        获取HTTP客户端（Agno兼容方法）
        
        Returns:
            HTTP客户端实例
        """
        # 返回我们的HTTP客户端
        return self.client
    
    def get_async_client(self):
        """
        获取异步HTTP客户端（Agno兼容方法）
        
        Returns:
            异步HTTP客户端实例（这里返回同一个客户端）
        """
        # 对于HTTP模型，我们返回同一个客户端
        # 实际的异步功能由HTTP客户端内部处理
        return self.client
    
    def response(self, messages: List[Any], **kwargs) -> Any:
        """
        Agno标准的response方法（兼容invoke）
        
        Args:
            messages: 输入消息
            **kwargs: 额外参数
            
        Returns:
            模型响应
        """
        # response方法通常与invoke方法相同
        return self.invoke(messages, **kwargs)
    
    def get_provider(self) -> str:
        """
        获取提供商名称（Agno兼容方法）
        
        Returns:
            提供商名称
        """
        return "http"
    
    def get_request_params(self, **kwargs) -> Dict[str, Any]:
        """
        获取请求参数（Agno兼容方法）
        
        Returns:
            请求参数字典
        """
        return {
            "temperature": kwargs.get("temperature", self.temperature),
            "max_tokens": kwargs.get("max_tokens", self.max_tokens),
            "model": self.id,
            "url": self.url
        }
    
    def __str__(self) -> str:
        """字符串表示"""
        return f"HttpAgnoModel(id={self.id}, vendor={self.vendor}, url={self.url})"
    
    def __repr__(self) -> str:
        """对象表示"""
        return self.__str__()


class HttpAgnoResponse:
    """HTTP Agno响应类，完全兼容Agno ModelResponse格式"""
    
    def __init__(self, content: str, finish_reason: Optional[str] = None, 
                 usage: Optional[Dict[str, int]] = None, model: Optional[str] = None):
        # 核心内容
        self.content = content
        self.finish_reason = finish_reason
        self.usage = usage or {}
        self.model = model
        
        # Agno ModelResponse 必需属性
        self.role = "assistant"
        self.tool_calls = []
        self.tool_executions = []
        self.citations = []
        self.audio = None
        self.image = None
        self.parsed = None
        self.extra = {}
        self.provider_data = {}
        self.response_usage = usage or {}
        self.created_at = None
        self.event = None
        self.reasoning_content = None
        self.redacted_thinking = None
        self.thinking = None
        
        # 向后兼容属性
        self.message = self
        self.choices = [self]
        
        # 添加一些常用方法
        self._text = content
    
    @property
    def text(self) -> str:
        """获取文本内容（Agno兼容）"""
        return self.content
    
    def get_content(self) -> str:
        """获取内容的方法形式"""
        return self.content
    
    def strip(self) -> str:
        """字符串strip方法（兼容性）"""
        return str(self.content).strip()
    
    def __len__(self) -> int:
        """长度方法"""
        return len(str(self.content))
    
    def __contains__(self, item) -> bool:
        """包含检查"""
        return item in str(self.content)
    
    def __getitem__(self, key):
        """索引访问"""
        return str(self.content)[key]
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "content": self.content,
            "role": self.role,
            "tool_calls": self.tool_calls,
            "tool_executions": self.tool_executions,
            "finish_reason": self.finish_reason,
            "usage": self.usage,
            "model": self.model
        }
    
    def __str__(self) -> str:
        """字符串表示"""
        return self.content
    
    def __repr__(self) -> str:
        """对象表示"""
        return f"HttpAgnoResponse(content='{self.content[:50]}...', model='{self.model}')"


def create_http_agno_model(url: str,
                          api_key: str,
                          model_id: str,
                          vendor: str = "generic",
                          **kwargs) -> HttpAgnoModel:
    """
    创建HTTP Agno模型的便捷函数
    
    Args:
        url: API端点URL
        api_key: API密钥
        model_id: 模型ID
        vendor: 供应商名称
        **kwargs: 其他参数
        
    Returns:
        HTTP Agno模型实例
    """
    return HttpAgnoModel(
        url=url,
        api_key=api_key,
        id=model_id,
        vendor=vendor,
        **kwargs
    )


if __name__ == "__main__":
    # 测试HTTP Agno适配器
    print("🧪 测试HTTP Agno适配器")
    
    # 示例配置
    test_config = {
        "url": "https://api.example.com/v1/chat/completions",
        "api_key": "test-key",
        "model_id": "test-model",
        "vendor": "generic"
    }
    
    try:
        # 创建HTTP Agno模型
        http_model = create_http_agno_model(**test_config)
        print(f"✅ 成功创建HTTP Agno模型: {http_model}")
        print(f"🆔 模型ID: {http_model.id}")
        print(f"🏷️  模型名称: {http_model.name}")
        print(f"📡 提供商: {http_model.provider}")
        
        # 测试消息转换
        test_messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello!"}
        ]
        
        http_messages = http_model._convert_agno_messages_to_http(test_messages)
        print(f"📝 转换后的消息: {http_messages}")
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")