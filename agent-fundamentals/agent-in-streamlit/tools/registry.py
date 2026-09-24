# tools/registry.py - Dynamic Tool Registry Manager
from typing import Dict, List, Any
from tools.base import BaseTool

class ToolRegistry:
    """
    Central hub that holds registered BaseTool instances, 
    exports API schemas to Groq, and invokes tools safely.
    """
    def __init__(self):
        self._tools: Dict[str, BaseTool] = {}

    def register(self, tool_input: Any) -> None:
        # Handle function decorated with @tool
        if hasattr(tool_input, 'tool_obj'):
            tool_obj = tool_input.tool_obj
        elif isinstance(tool_input, BaseTool):
            tool_obj = tool_input
        else:
            tool_obj = BaseTool(tool_input)

        self._tools[tool_obj.name] = tool_obj
        print(f"✅ Registered tool: '{tool_obj.name}'")

    def get_groq_schemas(self) -> List[Dict[str, Any]]:
        """Returns list of schemas ready for Groq client tools parameter"""
        return [t.schema for t in self._tools.values()]

    def list_tools(self) -> Dict[str, BaseTool]:
        """Returns the dictionary mapping tool names to BaseTool instances."""
        return self._tools

    def execute(self, tool_name: str, **kwargs) -> str:
        """Executes registered tool by string name"""
        if tool_name not in self._tools:
            return f"Error: Tool '{tool_name}' is not registered."
        return self._tools[tool_name].execute(**kwargs)
