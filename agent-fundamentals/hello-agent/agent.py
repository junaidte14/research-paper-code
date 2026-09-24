"""hello-agent: a correct, honestly-labeled function-calling agent loop.

Demonstrates the structured tool-calling pattern used by OpenAI/Groq-style
APIs (the mechanism behind Schick et al. 2023, Toolformer) — this is NOT
ReAct (Yao et al. 2022). See vs-literature.md for why that distinction
matters and what a real ReAct loop would need instead.
"""

import os
import json
from dotenv import load_dotenv
from groq import Groq

from tools import TOOL_MAP, TOOL_SCHEMAS, validate_arguments

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

SYSTEM_PROMPT = (
    "You are a helpful AI assistant equipped with tools. "
    "Use tools whenever necessary to fulfill the user's request accurately."
)


class FunctionCallingAgent:
    """Minimal structured tool-calling loop with validated arguments,
    safe execution, and an explicit (non-silent) iteration-limit outcome."""

    def __init__(self, model="openai/gpt-oss-20b", max_iterations=5, temperature=0.0):
        self.model = model
        self.max_iterations = max_iterations
        self.temperature = temperature

    def run(self, user_prompt: str) -> dict:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]
        steps = []

        for iteration in range(1, self.max_iterations + 1):
            response = client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=TOOL_SCHEMAS,
                tool_choice="auto",
                temperature=self.temperature,
            )
            msg = response.choices[0].message
            messages.append(msg)

            if not msg.tool_calls:
                return {
                    "status": "success",
                    "final_answer": msg.content,
                    "iterations": iteration,
                    "steps": steps,
                }

            for call in msg.tool_calls:
                name = call.function.name
                try:
                    args = json.loads(call.function.arguments)
                except json.JSONDecodeError as e:
                    result = f"Error: malformed arguments ({e})"
                else:
                    error = validate_arguments(name, args)
                    result = f"Error: {error}" if error else TOOL_MAP[name](**args)

                steps.append({
                    "iteration": iteration,
                    "tool": name,
                    "arguments": call.function.arguments,
                    "result": result,
                })
                messages.append({
                    "role": "tool",
                    "tool_call_id": call.id,
                    "name": name,
                    "content": str(result),
                })

        # Explicit outcome instead of the original's silent drop-off.
        return {
            "status": "max_iterations_reached",
            "final_answer": f"Stopped after {self.max_iterations} iterations without a final answer.",
            "iterations": self.max_iterations,
            "steps": steps,
        }


def run_agent(user_prompt: str) -> dict:
    """CLI-friendly wrapper, kept for parity with the original script's entry point."""
    agent = FunctionCallingAgent()
    result = agent.run(user_prompt)

    print(f"\n--- Task: '{user_prompt}' ---")
    for step in result["steps"]:
        print(f"[iter {step['iteration']}] {step['tool']}({step['arguments']}) -> {step['result']}")
    print(f"\n[{result['status']}]\n{result['final_answer']}")
    return result


if __name__ == "__main__":
    run_agent("What is the weather in Tokyo, and what is 22 multiplied by 5?")