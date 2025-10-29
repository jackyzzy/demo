import yfinance as yf
from langchain_core.tools import tool

@tool
def get_stock_info(symbol: str) -> str:
    """
    获取股票信息
    
    Args:
        symbol: 股票代码 (如: AAPL, TSLA)
    
    Returns:
        str: 股票信息
    """
    try:
        stock = yf.Ticker(symbol)
        info = stock.info
        hist = stock.history(period="1d")
        
        result = f"""
股票代码: {symbol}
公司名称: {info.get('longName', 'N/A')}
当前价格: ${info.get('currentPrice', 'N/A')}
市值: ${info.get('marketCap', 'N/A'):,} 
52周最高: ${info.get('fiftyTwoWeekHigh', 'N/A')}
52周最低: ${info.get('fiftyTwoWeekLow', 'N/A')}
行业: {info.get('industry', 'N/A')}
简介: {info.get('longBusinessSummary', 'N/A')[:200]}...
        """
        return result
    except Exception as e:
        return f"获取股票信息失败: {str(e)}"