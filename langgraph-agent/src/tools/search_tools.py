import os
from langchain_core.tools import tool
from tavily import TavilyClient
from duckduckgo_search import DDGS

@tool
def web_search(query: str, max_results: int = 5) -> str:
    """
    使用Tavily搜索引擎进行网络搜索
    
    Args:
        query: 搜索查询
        max_results: 最大结果数
    
    Returns:
        str: 搜索结果摘要
    """
    try:
        client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
        response = client.search(
            query=query,
            search_depth="advanced",
            max_results=max_results,
            include_answer=True
        )
        
        results = []
        for result in response.get("results", []):
            results.append(f"标题: {result['title']}\n内容: {result['content'][:300]}...\nURL: {result['url']}")
        
        return "\n\n---\n\n".join(results)
    except Exception as e:
        return f"搜索失败: {str(e)}"


@tool
def duckduckgo_search(query: str, max_results: int = 5) -> str:
    """
    使用DuckDuckGo进行隐私友好的搜索
    
    Args:
        query: 搜索查询
        max_results: 最大结果数
    
    Returns:
        str: 搜索结果
    """
    try:
        ddgs = DDGS()
        results = list(ddgs.text(query, max_results=max_results))
        
        formatted_results = []
        for result in results:
            formatted_results.append(f"标题: {result['title']}\n摘要: {result['body']}\nURL: {result['href']}")
        
        return "\n\n---\n\n".join(formatted_results)
    except Exception as e:
        return f"DuckDuckGo搜索失败: {str(e)}"