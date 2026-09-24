import re
import streamlit as st

# Import attributes from your config.py
from config import (
    DEFAULT_MODEL,
    AVAILABLE_MODELS,
    GROQ_API_KEY,
    DEFAULT_TEMPERATURE,
    MAX_TOKENS,
    DEFAULT_SYSTEM_PROMPT
)

from core.client import GroqClientManager
from core.memory import MemoryManager
from core.agent import AgentEngine

# Import ToolRegistry and specific tool functions directly
from tools.registry import ToolRegistry
from tools.math_tools import calculate_math_expression
from tools.search_tools import search_knowledge_base

# Page Setup
st.set_page_config(
    page_title="Groq AI Agent",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------
# Helper Functions & Session State
# ---------------------------------------------------------
@st.cache_resource
def get_tool_registry():
    """Instantiates and registers all active agent tools once."""
    registry = ToolRegistry()
    registry.register(calculate_math_expression)
    registry.register(search_knowledge_base)
    return registry

def init_session_state():
    """Initializes persistent Streamlit session variables using config defaults."""
    if "memory" not in st.session_state:
        st.session_state.memory = MemoryManager(
            system_prompt=DEFAULT_SYSTEM_PROMPT
        )
    if "active_tools" not in st.session_state:
        # Default all registered tools as active
        st.session_state.active_tools = list(get_tool_registry().list_tools().keys())

init_session_state()
tool_registry = get_tool_registry()

def sanitize_math_delimiters(text: str) -> str:
    if not text:
        return ""
    # Convert display math \[ ... \] to $$ ... $$
    text = re.sub(r'\\\[(.*?)\\\]', r'$$\1$$', text, flags=re.DOTALL)
    # Convert inline math \( ... \) to $ ... $
    text = re.sub(r'\\\((.*?)\\\)', r'$\1$', text, flags=re.DOTALL)
    return text

def format_agent_response(text: str) -> str:
    """
    Cleans up HTML artifact tags, converts LaTeX delimiters, 
    and formats inline code blocks for clean Streamlit rendering.
    """
    if not text:
        return ""
    
    # 1. Replace <br> or <br/> tags with standard Markdown newlines
    text = re.sub(r'<br\s*/?>', '\n', text, flags=re.IGNORECASE)
    
    # 2. Convert LaTeX display math \[ ... \] to $$ ... $$
    text = re.sub(r'\\\[(.*?)\\\]', r'$$\1$$', text, flags=re.DOTALL)
    
    # 3. Convert LaTeX inline math \( ... \) to $ ... $
    text = re.sub(r'\\\((.*?)\\\)', r'$\1$', text, flags=re.DOTALL)
    
    return text.strip()
# ---------------------------------------------------------
# Sidebar Controls
# ---------------------------------------------------------
with st.sidebar:
    st.title("⚙️ Agent Settings")
    
    # 1. API Key Input (Prefills from config/env)
    api_key_input = st.text_input(
        "Groq API Key",
        value=GROQ_API_KEY,
        type="password",
        help="Get your key at console.groq.com"
    )
    
    # 2. Model Selector
    default_index = AVAILABLE_MODELS.index(DEFAULT_MODEL) if DEFAULT_MODEL in AVAILABLE_MODELS else 0
    selected_model = st.selectbox(
        "LLM Model",
        options=AVAILABLE_MODELS,
        index=default_index
    )
    
    # 3. Model Parameters (Defaults driven by config.py)
    temperature = st.slider(
        "Temperature", 
        min_value=0.0, 
        max_value=1.0, 
        value=float(DEFAULT_TEMPERATURE), 
        step=0.1
    )
    
    max_context_tokens = st.slider(
        "Max Context Memory (Tokens)", 
        min_value=1000, 
        max_value=16000, 
        value=MAX_TOKENS * 4,  # Context buffer size
        step=1000
    )
    
    st.markdown("---")
    st.subheader("🛠️ Active Tools")
    
    # 4. Dynamic Tool Toggles
    all_registered_tools = tool_registry.list_tools()
    enabled_tools = []
    
    for tool_name, tool_obj in all_registered_tools.items():
        is_enabled = st.checkbox(
            f"**{tool_name}**", 
            value=(tool_name in st.session_state.active_tools),
            help=tool_obj.description
        )
        if is_enabled:
            enabled_tools.append(tool_name)
            
    st.session_state.active_tools = enabled_tools
    
    st.markdown("---")
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.memory.clear()
        st.rerun()

# ---------------------------------------------------------
# Main Chat Interface
# ---------------------------------------------------------
st.title("🤖 Groq Autonomous Agent")
st.caption("First-principles ReAct Agent running locally with Streamlit & Groq API.")

# Render Chat History
for msg in st.session_state.memory.messages:
    if msg.role == "system":
        continue
        
    with st.chat_message(msg.role):
        st.write(msg.content)
        
        # Display executed tools if attached
        if msg.tool_calls:
            with st.expander("🛠️ Executed Tools Log"):
                st.json(msg.tool_calls)

# User Query Processing
user_input = st.chat_input("Ask a question or request a tool task...")

if user_input:
    if not api_key_input:
        st.error("Missing Groq API Key! Please enter your key in the sidebar.")
        st.stop()
        
    # Render user prompt
    with st.chat_message("user"):
        st.write(user_input)
        
    # Process agent loop
    with st.chat_message("assistant"):
        with st.spinner("Thinking and executing actions..."):
            try:
                # Build runtime registry containing selected tools
                active_registry = ToolRegistry()
                for t_name in st.session_state.active_tools:
                    if t_name in all_registered_tools:
                        active_registry.register(all_registered_tools[t_name].func)

                # Initialize Client Manager & Agent Engine
                client_manager = GroqClientManager(api_key=api_key_input)
                agent = AgentEngine(
                    client_manager=client_manager,
                    tool_registry=active_registry,
                    model=selected_model,
                    temperature=temperature
                )
                
                # Execute ReAct Loop
                result = agent.run(
                    user_prompt=user_input,
                    memory=st.session_state.memory
                )
                
                final_response = result.get("final_answer", "")
                thought_steps = result.get("steps", [])
                
                # Render Thought Trace
                if thought_steps:
                    with st.expander("💭 ReAct Thought Process", expanded=False):
                        for step in thought_steps:
                            iteration = step.get("iteration", 1)
                            tool_name = step.get("tool", "Unknown Tool")
                            tool_args = step.get("arguments", {})
                            tool_output = step.get("output", "")

                            st.markdown(f"**Step {iteration}:** Invoked `{tool_name}`")
                            st.code(f"Tool Call: {tool_name}({tool_args})")
                            st.info(f"Result: {tool_output}")
                
                # Output final answer
                if final_response:
                    formatted_text = format_agent_response(final_response)
                    st.markdown(formatted_text)
                else:
                    st.warning("No final response returned from the agent.")
                
            except Exception as e:
                st.error(f"Execution Error: {str(e)}")