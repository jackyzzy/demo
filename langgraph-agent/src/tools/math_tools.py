from langchain_core.tools import tool

@tool
def calculator(expression: str) -> str:
    """
    执行数学计算
    
    Args:
        expression: 数学表达式
    
    Returns:
        str: 计算结果
    """
    try:
        # 安全的数学表达式计算
        allowed_chars = set('0123456789+-*/(). ')
        if not all(c in allowed_chars for c in expression):
            return "计算表达式包含不允许的字符"
        
        result = eval(expression)
        return f"计算结果: {expression} = {result}"
    except Exception as e:
        return f"计算错误: {str(e)}"