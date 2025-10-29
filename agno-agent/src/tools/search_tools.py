import os
from agno.tools import Function
from tavily import TavilyClient

class TavilySearchTool(Function):
    """Tavily搜索工具"""
    
    def __init__(self):
        super().__init__(
            name="tavily_search",
            description="使用Tavily搜索引擎进行高质量网络搜索，专为AI优化",
        )
        # Initialize the client in a way that doesn't conflict with Pydantic
        api_key = os.getenv("TAVILY_API_KEY")
        if api_key:
            self._client = TavilyClient(api_key=api_key)
        else:
            self._client = None
    
    def run(self, query: str, max_results: int = 5) -> str:
        """执行搜索"""
        try:
            if not self._client:
                return "Tavily搜索失败: 未配置TAVILY_API_KEY环境变量"
            
            response = self._client.search(
                query=query,
                search_depth="advanced", 
                max_results=max_results,
                include_answer=True
            )
            
            results = []
            for result in response.get("results", []):
                results.append(
                    f"标题: {result['title']}\n"
                    f"内容: {result['content'][:300]}...\n"
                    f"URL: {result['url']}\n"
                )
            
            return "\n---\n".join(results)
        except Exception as e:
            return f"Tavily搜索失败: {str(e)}"