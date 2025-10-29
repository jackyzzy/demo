"""
Parlant金融工具
"""

import yfinance as yf
import parlant.sdk as p


@p.tool
async def get_stock_info(context: p.ToolContext, symbol: str) -> p.ToolResult:
    """
    获取详细的股票信息、财务数据和技术分析

    Args:
        context: Parlant工具上下文
        symbol: 股票代码

    Returns:
        p.ToolResult: 股票信息
    """
    try:
        stock = yf.Ticker(symbol)
        info = stock.info
        hist = stock.history(period="1mo")

        # 基本信息
        basic_info = f"""
📊 股票代码: {symbol}
🏢 公司名称: {info.get('longName', 'N/A')}
💰 当前价格: ${info.get('currentPrice', 'N/A')}
📈 今日涨跌: {info.get('regularMarketChangePercent', 'N/A')}%
🏆 市值: ${info.get('marketCap', 0):,}
📊 市盈率: {info.get('trailingPE', 'N/A')}
🎯 52周最高: ${info.get('fiftyTwoWeekHigh', 'N/A')}
📉 52周最低: ${info.get('fiftyTwoWeekLow', 'N/A')}
🏭 行业: {info.get('industry', 'N/A')}
🌍 板块: {info.get('sector', 'N/A')}
        """

        # 技术指标
        if not hist.empty:
            current_price = hist['Close'][-1]
            sma_20 = hist['Close'].rolling(window=20).mean()[-1]
            volatility = hist['Close'].pct_change().std() * 100

            technical_info = f"""
📈 技术指标:
  - 20日均线: ${sma_20:.2f}
  - 当前价格vs均线: {'上方' if current_price > sma_20 else '下方'}
  - 波动率: {volatility:.2f}%
            """
        else:
            technical_info = "📈 技术指标: 数据不足"

        # 公司描述
        description = info.get('longBusinessSummary', 'N/A')
        if len(description) > 300:
            description = description[:300] + "..."

        result = f"{basic_info}\n{technical_info}\n\n📝 公司简介:\n{description}"
        return p.ToolResult(result)

    except Exception as e:
        return p.ToolResult(f"获取股票信息失败: {str(e)}")
