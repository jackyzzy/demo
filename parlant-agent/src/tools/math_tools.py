"""
Parlant数学计算工具
"""

import math
import parlant.sdk as p


@p.tool
async def calculator(context: p.ToolContext, expression: str) -> p.ToolResult:
    """
    执行数学计算、统计分析等运算

    Args:
        context: Parlant工具上下文
        expression: 数学表达式

    Returns:
        p.ToolResult: 计算结果
    """
    try:
        # 安全计算
        allowed_chars = set('0123456789+-*/().eE ')
        allowed_words = {'sin', 'cos', 'tan', 'log', 'exp', 'sqrt', 'pi', 'abs'}

        # 基本安全检查
        clean_expr = expression.replace(' ', '')
        if not all(c in allowed_chars or c.isalpha() for c in clean_expr):
            return p.ToolResult("表达式包含不允许的字符")

        # 简单数学运算
        safe_dict = {
            "__builtins__": {},
            "sin": math.sin, "cos": math.cos, "tan": math.tan,
            "log": math.log, "exp": math.exp, "sqrt": math.sqrt,
            "pi": math.pi, "abs": abs, "pow": pow
        }

        result = eval(expression, safe_dict)
        return p.ToolResult(f"计算结果: {expression} = {result}")
    except Exception as e:
        return p.ToolResult(f"计算错误: {str(e)}")
