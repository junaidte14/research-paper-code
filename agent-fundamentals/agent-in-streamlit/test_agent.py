# test_agent.py - Verification Script for Agent Engine
from core.client import GroqClientManager
from tools.registry import ToolRegistry
from tools.math_tools import calculate_math_expression
from tools.search_tools import search_knowledge_base
from core.agent import AgentEngine

if __name__ == "__main__ font-mono":
    print("=== Testing ReAct Agent Engine ===")
    
    # 1. Initialize Client & Registry
    client_mgr = GroqClientManager()
    registry = ToolRegistry()
    registry.register(calculate_math_expression)
    registry.register(search_knowledge_base)

    # 2. Instantiate Agent
    agent = AgentEngine(
        client_manager=client_mgr,
        tool_registry=registry,
        max_iterations=5
    )

    # 3. Test Query needing multiple tools
    query = "What is 25 multiplied by 8, and what is Groq?"
    print(f"\nUser Task: '{query}'")
    
    result = agent.run(query)

    print("\n=== Execution Result ===")
    print(f"Status: {result['status']}")
    print(f"Iterations Used: {result['iterations']}")
    print(f"Final Answer:\n{result['final_answer']}")
