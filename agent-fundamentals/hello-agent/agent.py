import os
import json
from dotenv import load_dotenv
from groq import Groq

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# 1. Define real Python functions
def get_weather(location: str) -> str:
    # Mock weather tool
    return f"The current weather in {location} is 22°C (72°F) and sunny."

def calculate_expression(expression: str) -> str:
    try:
        # Safe evaluation for basic math
        result = eval(expression, {"__builtins__": None}, {})
        return f"Result: {result}"
    except Exception as e:
        return f"Error evaluating expression: {str(e)}"

# 2. Map tool names to python functions for invocation
tool_map = {
    "get_weather": get_weather,
    "calculate_expression": calculate_expression
}

# 3. Define schema for Groq API
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather for a given city",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "City name"}
                },
                "required": ["location"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate_expression",
            "description": "Evaluate a mathematical expression",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {"type": "string", "description": "e.g., 25 * 4 + 10"}
                },
                "required": ["expression"],
            },
        },
    }
]

def run_agent(user_prompt: str):
    # Initialize conversation history with system instructions
    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful AI assistant equipped with tools. "
                "Use tools whenever necessary to fulfill the user's request accurately."
            )
        },
        {"role": "user", "content": user_prompt}
    ]

    print(f"\n--- Starting Task: '{user_prompt}' ---")

    # Execution loop (max 5 iterations to prevent infinite loops)
    for i in range(5):
        response = client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=messages,
            tools=tools,
            tool_choice="auto",
            temperature=0.0
        )

        response_message = response.choices[0].message
        messages.append(response_message)

        # Check if model wants to call a tool
        if response_message.tool_calls:
            for tool_call in response_message.tool_calls:
                function_name = tool_call.function.name
                arguments = json.loads(tool_call.function.arguments)

                print(f"-> [Agent Decision]: Call tool '{function_name}' with args {arguments}")

                # Execute local tool function
                function_to_call = tool_map[function_name]
                tool_output = function_to_call(**arguments)

                print(f"<- [Tool Result]: {tool_output}")

                # Feed tool output back into conversation history
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": function_name,
                    "content": tool_output,
                })
        else:
            # If no tool calls were requested, task is complete
            print(f"\n[Final Answer]:\n{response_message.content}")
            break

if __name__ == "__main__":
    # Test query requiring multi-tool reasoning
    run_agent("What is the weather in Tokyo, and what is 22 multiplied by 5?")