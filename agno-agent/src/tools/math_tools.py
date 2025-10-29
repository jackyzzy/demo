import math
from agno.tools import Function

class AdvancedCalculatorTool(Function):
    """高级计算器工具"""
    
    def __init__(self):
        super().__init__(
            name="advanced_calculator",
            description="执行数学计算、统计分析等运算",
        )
    
    def run(self, expression: str) -> str:
        """执行计算"""
        try:
            # 安全计算
            allowed_chars = set('0123456789+-*/().eE ')
            allowed_words = {'sin', 'cos', 'tan', 'log', 'exp', 'sqrt', 'pi', 'abs'}
            
            # 基本安全检查
            clean_expr = expression.replace(' ', '')
            if not all(c in allowed_chars or c.isalpha() for c in clean_expr):
                return "表达式包含不允许的字符"
            
            # 简单数学运算
            safe_dict = {
                "__builtins__": {},
                "sin": math.sin, "cos": math.cos, "tan": math.tan,
                "log": math.log, "exp": math.exp, "sqrt": math.sqrt,
                "pi": math.pi, "abs": abs, "pow": pow
            }
            
            result = eval(expression, safe_dict)
            return f"计算结果: {expression} = {result}"
        except Exception as e:
            return f"计算错误: {str(e)}"