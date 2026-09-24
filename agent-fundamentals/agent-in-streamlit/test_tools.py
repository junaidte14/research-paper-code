# test_tools.py - Independent CLI Subsystem Verification
from tools.registry import ToolRegistry
from tools.math_tools import calculate_math_expression
from tools.search_tools import search_knowledge_base

if __name__ == "__main__":
    print("--- Initializing Tool Registry ---")
    registry = ToolRegistry()
    
    # Register tools
    registry.register(calculate_math_expression)
    registry.register(search_knowledge_base)

    print("\n--- Generated Schemas for Groq API ---")
    print(registry.get_groq_schemas())

    print("\n--- Testing Safe Executions ---")
    math_res = registry.execute("calculate_math_expression", expression="(25 * 4) + 150")
    print(f"Math Result: {math_res}")

    search_res = registry.execute("search_knowledge_base", query="groq")
    print(f"Search Result: {search_res}")
