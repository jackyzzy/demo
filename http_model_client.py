"""
HTTP模型客户端
用于通过HTTP API访问各种模型提供商的统一接口
"""

import json
import requests
from typing import Dict, List, Any, Optional, Iterator
from dataclasses import dataclass
import time
import logging

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class HttpModelResponse:
    """HTTP模型响应"""
    content: str
    finish_reason: Optional[str] = None
    usage: Optional[Dict[str, int]] = None
    model: Optional[str] = None


class HttpModelClient:
    """HTTP模型客户端基类"""
    
    def __init__(self, 
                 url: str,
                 api_key: str,
                 model_id: str,
                 headers: Optional[Dict[str, str]] = None,
                 vendor: str = "generic",
                 max_retries: int = 3,
                 timeout: int = 60):
        """
        初始化HTTP模型客户端
        
        Args:
            url: API端点URL
            api_key: API密钥
            model_id: 模型ID
            headers: 额外的HTTP头部
            vendor: 供应商名称
            max_retries: 最大重试次数
            timeout: 请求超时时间
        """
        self.url = url
        self.api_key = api_key
        self.model_id = model_id
        self.vendor = vendor
        self.max_retries = max_retries
        self.timeout = timeout
        
        # 设置默认头部
        self.headers = {
            "Content-Type": "application/json"
        }
        
        # 根据供应商设置认证头部
        if vendor.lower() == "huawei":
            self.headers["Authorization"] = f"Bearer {api_key}"
        elif vendor.lower() == "openai" or vendor.lower() == "generic":
            self.headers["Authorization"] = f"Bearer {api_key}"
        else:
            # 默认使用Bearer认证
            self.headers["Authorization"] = f"Bearer {api_key}"
        
        # 合并额外头部
        if headers:
            self.headers.update(headers)
    
    def _prepare_messages(self, messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """
        预处理消息格式
        
        Args:
            messages: 消息列表
            
        Returns:
            处理后的消息列表
        """
        processed_messages = []
        
        for msg in messages:
            processed_msg = msg.copy()
            
            # 处理角色映射
            if processed_msg.get("role") == "developer":
                processed_msg["role"] = "system"
            elif processed_msg.get("role") not in ["system", "user", "assistant"]:
                processed_msg["role"] = "user"
            
            processed_messages.append(processed_msg)
        
        return processed_messages
    
    def _prepare_payload(self, 
                        messages: List[Dict[str, str]], 
                        temperature: float = 0.7,
                        max_tokens: Optional[int] = None,
                        stream: bool = False,
                        **kwargs) -> Dict[str, Any]:
        """
        准备请求负载
        
        Args:
            messages: 消息列表
            temperature: 温度参数
            max_tokens: 最大令牌数
            stream: 是否流式输出
            **kwargs: 其他参数
            
        Returns:
            请求负载
        """
        payload = {
            "model": self.model_id,
            "messages": self._prepare_messages(messages),
            "temperature": temperature,
            "stream": stream
        }
        
        if max_tokens:
            payload["max_tokens"] = max_tokens
        
        # 合并其他参数
        payload.update(kwargs)
        
        return payload
    
    def _send_request(self, payload: Dict[str, Any]) -> requests.Response:
        """
        发送HTTP请求
        
        Args:
            payload: 请求负载
            
        Returns:
            HTTP响应
            
        Raises:
            Exception: 请求失败时抛出异常
        """
        for attempt in range(self.max_retries):
            try:
                logger.debug(f"发送请求到 {self.url} (尝试 {attempt + 1}/{self.max_retries})")
                
                response = requests.post(
                    self.url,
                    headers=self.headers,
                    data=json.dumps(payload),
                    timeout=self.timeout,
                    verify=False  # 可能需要处理SSL证书问题
                )
                
                if response.status_code == 200:
                    return response
                else:
                    logger.warning(f"请求失败，状态码: {response.status_code}, 响应: {response.text}")
                    if attempt < self.max_retries - 1:
                        time.sleep(2 ** attempt)  # 指数退避
                    else:
                        raise Exception(f"HTTP请求失败: {response.status_code} - {response.text}")
                        
            except requests.RequestException as e:
                logger.warning(f"请求异常: {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)
                else:
                    raise Exception(f"HTTP请求失败: {e}")
    
    def _parse_response(self, response: requests.Response) -> HttpModelResponse:
        """
        解析HTTP响应
        
        Args:
            response: HTTP响应
            
        Returns:
            解析后的模型响应
        """
        try:
            data = response.json()
            
            # 提取内容
            if "choices" in data and data["choices"]:
                choice = data["choices"][0]
                content = choice.get("message", {}).get("content", "")
                finish_reason = choice.get("finish_reason")
            else:
                content = data.get("content", str(data))
                finish_reason = None
            
            # 提取使用情况
            usage = data.get("usage", {})
            model = data.get("model", self.model_id)
            
            return HttpModelResponse(
                content=content,
                finish_reason=finish_reason,
                usage=usage,
                model=model
            )
            
        except json.JSONDecodeError as e:
            # 如果JSON解析失败，返回原始文本
            logger.warning(f"JSON解析失败: {e}，返回原始响应")
            return HttpModelResponse(content=response.text)
    
    def chat_completion(self, 
                       messages: List[Dict[str, str]], 
                       temperature: float = 0.7,
                       max_tokens: Optional[int] = None,
                       **kwargs) -> HttpModelResponse:
        """
        执行聊天补全
        
        Args:
            messages: 消息列表
            temperature: 温度参数
            max_tokens: 最大令牌数
            **kwargs: 其他参数
            
        Returns:
            模型响应
        """
        payload = self._prepare_payload(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=False,
            **kwargs
        )
        
        response = self._send_request(payload)
        return self._parse_response(response)
    
    def stream_chat_completion(self, 
                              messages: List[Dict[str, str]], 
                              temperature: float = 0.7,
                              max_tokens: Optional[int] = None,
                              **kwargs) -> Iterator[HttpModelResponse]:
        """
        执行流式聊天补全
        
        Args:
            messages: 消息列表
            temperature: 温度参数
            max_tokens: 最大令牌数
            **kwargs: 其他参数
            
        Yields:
            流式响应片段
        """
        payload = self._prepare_payload(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
            **kwargs
        )
        
        try:
            response = requests.post(
                self.url,
                headers=self.headers,
                data=json.dumps(payload),
                timeout=self.timeout,
                stream=True,
                verify=False
            )
            
            if response.status_code != 200:
                raise Exception(f"Stream request failed: {response.status_code} - {response.text}")
            
            for line in response.iter_lines():
                if line:
                    line = line.decode('utf-8')
                    if line.startswith('data: '):
                        data_str = line[6:]  # 移除 'data: ' 前缀
                        if data_str.strip() == '[DONE]':
                            break
                        
                        try:
                            data = json.loads(data_str)
                            if "choices" in data and data["choices"]:
                                choice = data["choices"][0]
                                delta = choice.get("delta", {})
                                content = delta.get("content", "")
                                if content:
                                    yield HttpModelResponse(content=content)
                        except json.JSONDecodeError:
                            continue
                            
        except Exception as e:
            logger.error(f"流式请求失败: {e}")
            raise


class HttpModelClientFactory:
    """HTTP模型客户端工厂"""
    
    @staticmethod
    def create_client(url: str, 
                     api_key: str, 
                     model_id: str, 
                     vendor: str = "generic",
                     headers: Optional[Dict[str, str]] = None,
                     **kwargs) -> HttpModelClient:
        """
        创建HTTP模型客户端
        
        Args:
            url: API端点URL
            api_key: API密钥
            model_id: 模型ID
            vendor: 供应商名称
            headers: 额外的HTTP头部
            **kwargs: 其他参数
            
        Returns:
            HTTP模型客户端实例
        """
        return HttpModelClient(
            url=url,
            api_key=api_key,
            model_id=model_id,
            vendor=vendor,
            headers=headers,
            **kwargs
        )


# 便捷函数
def create_http_client(url: str, 
                      api_key: str, 
                      model_id: str, 
                      vendor: str = "generic",
                      **kwargs) -> HttpModelClient:
    """
    创建HTTP模型客户端的便捷函数
    
    Args:
        url: API端点URL
        api_key: API密钥
        model_id: 模型ID
        vendor: 供应商名称
        **kwargs: 其他参数
        
    Returns:
        HTTP模型客户端实例
    """
    return HttpModelClientFactory.create_client(
        url=url, 
        api_key=api_key, 
        model_id=model_id, 
        vendor=vendor,
        **kwargs
    )


if __name__ == "__main__":
    # 测试HTTP客户端（使用华为云示例）
    print("🧪 测试HTTP模型客户端")
    
    # 示例配置（实际使用时需要真实的API密钥）
    test_config = {
        "url": "https://maas-cn-southwest-2.modelarts-maas.com/v1/infers/8a062fd4-7367-4ab4-a936-5eeb8fb821c4/v1/chat/completions",
        "api_key": "test-key",  # 替换为真实API密钥
        "model_id": "DeepSeek-R1",
        "vendor": "huawei"
    }
    
    try:
        client = create_http_client(**test_config)
        print(f"✅ 成功创建HTTP客户端: {client.vendor}")
        print(f"📍 端点: {client.url}")
        print(f"🤖 模型: {client.model_id}")
        
        # 测试消息
        test_messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello, how are you?"}
        ]
        
        print("\n🔧 准备测试请求...")
        # 注意：这里只是演示，实际运行需要有效的API密钥
        
    except Exception as e:
        print(f"❌ 创建客户端失败: {e}")