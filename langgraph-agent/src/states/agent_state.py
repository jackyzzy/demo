from typing import List, Dict, Any
from typing_extensions import TypedDict

class AgentState(TypedDict):
    """智能体状态定义"""
    messages: List[Any]  
    current_task: str   
    task_type: str      
    search_results: List[Dict]  
    analysis_results: Dict[str, Any]  
    reasoning_steps: List[str]  
    plan: List[Dict]    
    completed_steps: List[str]  
    context: Dict[str, Any]  
    next_action: str