"""
HTTP模型的LangChain适配器
将HTTP模型客户端适配为LangChain兼容的聊天模型
"""

from typing import Any, Dict, List, Optional, Iterator
from pydantic import Field
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.outputs import ChatResult, ChatGeneration
from langchain_core.callbacks import CallbackManagerForLLMRun

from http_model_client import HttpModelClient, create_http_client, HttpModelResponse


class HttpChatModel(BaseChatModel):
    """HTTP模型的LangChain聊天模型适配器"""
    
    # Pydantic字段定义
    url: str = Field(description="API端点URL")
    api_key: str = Field(description="API密钥")
    model_id: str = Field(description="模型ID") 
    vendor: str = Field(default="generic", description="供应商名称")
    headers: Optional[Dict[str, str]] = Field(default=None, description="额外的HTTP头部")
    temperature: float = Field(default=0.7, description="温度参数")
    max_tokens: Optional[int] = Field(default=None, description="最大令牌数")
    timeout: int = Field(default=60, description="请求超时时间")
    max_retries: int = Field(default=3, description="最大重试次数")
    client: Optional[HttpModelClient] = Field(default=None, exclude=True, description="HTTP客户端")
    
    def __init__(self,
                 url: str,
                 api_key: str,
                 model_id: str,
                 vendor: str = "generic",
                 headers: Optional[Dict[str, str]] = None,
                 temperature: float = 0.7,
                 max_tokens: Optional[int] = None,
                 timeout: int = 60,
                 max_retries: int = 3,
                 **kwargs):
        """
        初始化HTTP聊天模型
        
        Args:
            url: API端点URL
            api_key: API密钥
            model_id: 模型ID
            vendor: 供应商名称
            headers: 额外的HTTP头部
            temperature: 温度参数
            max_tokens: 最大令牌数
            timeout: 请求超时时间
            max_retries: 最大重试次数
            **kwargs: 其他参数
        """
        # 调用父类初始化，传递所有参数
        super().__init__(
            url=url,
            api_key=api_key,
            model_id=model_id,
            vendor=vendor,
            headers=headers,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
            max_retries=max_retries,
            **kwargs
        )
        
        # 创建HTTP客户端
        self.client = create_http_client(
            url=url,
            api_key=api_key,
            model_id=model_id,
            vendor=vendor,
            headers=headers,
            timeout=timeout,
            max_retries=max_retries
        )
    
    def _convert_messages_to_http(self, messages: List[BaseMessage]) -> List[Dict[str, str]]:
        """
        将LangChain消息转换为HTTP API格式
        
        Args:
            messages: LangChain消息列表
            
        Returns:
            HTTP API格式的消息列表
        """
        http_messages = []
        
        for message in messages:
            if isinstance(message, HumanMessage):
                role = "user"
            elif isinstance(message, AIMessage):
                role = "assistant"
            elif isinstance(message, SystemMessage):
                role = "system"
            else:
                # 默认作为用户消息处理
                role = "user"
            
            http_messages.append({
                "role": role,
                "content": message.content
            })
        
        return http_messages
    
    def _convert_http_response_to_langchain(self, response: HttpModelResponse) -> ChatResult:
        """
        将HTTP响应转换为LangChain格式
        
        Args:
            response: HTTP模型响应
            
        Returns:
            LangChain聊天结果
        """
        message = AIMessage(content=response.content)
        
        generation = ChatGeneration(
            message=message,
            generation_info={
                "finish_reason": response.finish_reason,
                "model": response.model,
                "usage": response.usage
            }
        )
        
        return ChatResult(generations=[generation])
    
    def _generate(self,
                  messages: List[BaseMessage],
                  stop: Optional[List[str]] = None,
                  run_manager: Optional[CallbackManagerForLLMRun] = None,
                  **kwargs: Any) -> ChatResult:
        """
        生成聊天响应
        
        Args:
            messages: 输入消息
            stop: 停止序列
            run_manager: 回调管理器
            **kwargs: 额外参数
            
        Returns:
            聊天结果
        """
        # 转换消息格式
        http_messages = self._convert_messages_to_http(messages)
        
        # 准备参数
        temperature = kwargs.get("temperature", self.temperature)
        max_tokens = kwargs.get("max_tokens", self.max_tokens)
        
        # 调用HTTP客户端
        try:
            response = self.client.chat_completion(
                messages=http_messages,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs
            )
            
            # 转换响应
            return self._convert_http_response_to_langchain(response)
            
        except Exception as e:
            # 如果出错，返回错误消息
            error_message = AIMessage(content=f"HTTP模型调用失败: {str(e)}")
            generation = ChatGeneration(message=error_message)
            return ChatResult(generations=[generation])
    
    def _stream(self,
                messages: List[BaseMessage],
                stop: Optional[List[str]] = None,
                run_manager: Optional[CallbackManagerForLLMRun] = None,
                **kwargs: Any) -> Iterator[ChatGeneration]:
        """
        流式生成聊天响应
        
        Args:
            messages: 输入消息
            stop: 停止序列
            run_manager: 回调管理器
            **kwargs: 额外参数
            
        Yields:
            聊天生成片段
        """
        # 转换消息格式
        http_messages = self._convert_messages_to_http(messages)
        
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
                message = AIMessage(content=chunk.content)
                yield ChatGeneration(message=message)
                
        except Exception as e:
            # 如果出错，生成错误消息
            error_message = AIMessage(content=f"HTTP流式调用失败: {str(e)}")
            yield ChatGeneration(message=error_message)
    
    @property
    def _llm_type(self) -> str:
        """返回LLM类型"""
        return f"http_{self.vendor}"
    
    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """返回识别参数"""
        return {
            "model_id": self.model_id,
            "url": self.url,
            "vendor": self.vendor,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens
        }


def create_http_chat_model(url: str,
                          api_key: str,
                          model_id: str,
                          vendor: str = "generic",
                          **kwargs) -> HttpChatModel:
    """
    创建HTTP聊天模型的便捷函数
    
    Args:
        url: API端点URL
        api_key: API密钥
        model_id: 模型ID
        vendor: 供应商名称
        **kwargs: 其他参数
        
    Returns:
        HTTP聊天模型实例
    """
    return HttpChatModel(
        url=url,
        api_key=api_key,
        model_id=model_id,
        vendor=vendor,
        **kwargs
    )


if __name__ == "__main__":
    # 测试HTTP LangChain适配器
    print("🧪 测试HTTP LangChain适配器")
    
    # 示例配置
    test_config = {
        "url": "https://api.example.com/v1/chat/completions",
        "api_key": "test-key",
        "model_id": "test-model",
        "vendor": "generic"
    }
    
    try:
        # 创建HTTP聊天模型
        http_model = create_http_chat_model(**test_config)
        print(f"✅ 成功创建HTTP LangChain模型")
        print(f"🏷️  模型类型: {http_model._llm_type}")
        print(f"🆔 识别参数: {http_model._identifying_params}")
        
        # 测试消息转换
        from langchain_core.messages import HumanMessage, SystemMessage
        
        test_messages = [
            SystemMessage(content="You are a helpful assistant."),
            HumanMessage(content="Hello!")
        ]
        
        http_messages = http_model._convert_messages_to_http(test_messages)
        print(f"📝 转换后的消息: {http_messages}")
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")