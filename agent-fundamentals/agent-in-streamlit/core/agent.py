# core/agent.py - Main ReAct Thought-Action Engine
import json
from typing import Dict, List, Any, Optional
from core.client import GroqClientManager
from core.memory import MemoryManager
from tools.registry import ToolRegistry

class AgentEngine:
    """
    Autonomous LLM Agent executing the ReAct (Reason + Act) loop 
    with Groq API, dynamic tools, and loop safeguards.
    """
    def __init__(
        self,
        client_manager: GroqClientManager,
        tool_registry: ToolRegistry,
        system_prompt: Optional[str] = None,
        max_iterations: int = 5,
        model: Optional[str] = None,
        temperature: float = 0.0
    ):
        self.client_mgr = client_manager
        self.registry = tool_registry
        self.max_iterations = max_iterations
        self.system_prompt = system_prompt or (
            "You are a helpful AI Agent equipped with tools. "
            "Use available tools whenever precise math, database lookups, or external facts are needed."
            """Formatting Guidelines for Responses:
                1. Use standard Markdown for headings (##, ###), bold text, bullet lists, and tables.
                2. For mathematical expressions:
                - Use single dollar signs for inline math: $E = mc^2$
                - Use double dollar signs for standalone block equations:
                    $$\\frac{a}{b} = c$$
                3. For code snippets, always specify the language inside fenced blocks:
                ```python
                import streamlit as st"""
        )
        self.model = model if model is not None else getattr(self.client_mgr, 'model_name', "llama3-8b-8192")
        self.temperature = temperature

    def run(self, user_prompt: str, memory: Optional[MemoryManager] = None) -> Dict[str, Any]:
        """
        Executes the ReAct loop for a given user prompt.
        Returns a dict containing final response, execution steps, and message history.
        """
        # If a MemoryManager instance is provided, register the user input in memory
        if memory is not None:
            memory.add_message(role="user", content=user_prompt)
            messages = memory.get_pruned_messages()
        else:
            # Fallback if no memory manager is passed
            messages = [
                {"role": "system", "content": self.system_prompt or "You are a helpful assistant."},
                {"role": "user", "content": user_prompt}
            ]

        available_tools = self.registry.get_groq_schemas()
        client = self.client_mgr.client
        steps_log = []

        for iteration in range(1, self.max_iterations + 1):
            print(f"--- ReAct Iteration {iteration}/{self.max_iterations} ---")

            # Send prompt and tool definitions to Groq
            response = client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=available_tools if available_tools else None,
                tool_choice="auto" if available_tools else None,
                temperature=self.temperature
            )

            response_message = response.choices[0].message
            
            # Convert Groq response message to dict and append to history
            msg_dict = {"role": "assistant"}
            if response_message.content:
                msg_dict["content"] = response_message.content
            if response_message.tool_calls:
                msg_dict["tool_calls"] = [tc.model_dump() for tc in response_message.tool_calls]

            messages.append(msg_dict)

            # Check if LLM requested tool execution
            if response_message.tool_calls:

                for tool_call in response_message.tool_calls:
                    fn_name = tool_call.function.name
                    fn_args = json.loads(tool_call.function.arguments)
                    
                    print(f"🛠️ Executing Tool: '{fn_name}' with args {fn_args}")
                    
                    # Run local tool safely
                    tool_output = self.registry.execute(fn_name, **fn_args)
                    
                    steps_log.append({
                        "iteration": iteration,
                        "tool": fn_name,
                        "arguments": fn_args,
                        "output": tool_output
                    })

                    # Append tool response to memory / message history
                    if memory is not None:
                        memory.add_message(
                            role="tool",
                            content=str(tool_output),
                            name=fn_name,
                            tool_call_id=tool_call.id
                        )
                        messages = memory.get_pruned_messages()
                    else:
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "name": fn_name,
                            "content": str(tool_output)
                        })

                # Refresh message context after ALL tools in this turn are executed
                if memory is not None:
                    messages = memory.get_pruned_messages()
            else:
                # No tool calls: LLM has finished reasoning
                if memory is not None:
                    memory.add_message(role="assistant", content=response_message.content)
                
                return {
                    "status": "success",
                    "final_answer": response_message.content,
                    "iterations": iteration,
                    "steps": steps_log,
                    "messages": messages
                }

        # Safeguard limit reached
        return {
            "status": "max_iterations_reached",
            "final_answer": "Reached maximum iteration limit before completing request.",
            "iterations": self.max_iterations,
            "steps": steps_log,
            "messages": messages
        }
