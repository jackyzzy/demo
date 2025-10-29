"""
Parlant搜索工具
使用@p.tool装饰器定义工具
"""

import os
import parlant.sdk as p
from tavily import TavilyClient
from duckduckgo_search import DDGS


@p.tool
async def tavily_search(context: p.ToolContext, query: str, max_results: int = 5) -> p.ToolResult:
    """
    使用Tavily搜索引擎进行高质量网络搜索，专为AI优化

    Args:
        context: Parlant工具上下文
        query: 搜索查询
        max_results: 最大结果数，默认5

    Returns:
        p.ToolResult: 搜索结果
    """
    try:
        api_key = os.getenv("TAVILY_API_KEY")
        if not api_key:
            return p.ToolResult("Tavily搜索失败: 未配置TAVILY_API_KEY环境变量")

        client = TavilyClient(api_key=api_key)
        response = client.search(
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

        return p.ToolResult("\n---\n".join(results))
    except Exception as e:
        return p.ToolResult(f"Tavily搜索失败: {str(e)}")


@p.tool
async def duckduckgo_search(context: p.ToolContext, query: str, max_results: int = 5) -> p.ToolResult:
    """
    使用DuckDuckGo搜索引擎进行网络搜索

    Args:
        context: Parlant工具上下文
        query: 搜索查询
        max_results: 最大结果数，默认5

    Returns:
        p.ToolResult: 搜索结果
    """
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))

        if not results:
            return p.ToolResult("未找到搜索结果")

        formatted_results = []
        for result in results:
            formatted_results.append(
                f"标题: {result.get('title', 'N/A')}\n"
                f"内容: {result.get('body', 'N/A')[:300]}...\n"
                f"URL: {result.get('href', 'N/A')}\n"
            )

        return p.ToolResult("\n---\n".join(formatted_results))
    except Exception as e:
        return p.ToolResult(f"DuckDuckGo搜索失败: {str(e)}")
