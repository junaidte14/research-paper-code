# config.py - Central Configuration & Environment Secrets
import os
from dotenv import load_dotenv

# Load variables from local .env file
load_dotenv()

# Available Groq LLM Models
DEFAULT_MODEL = "openai/gpt-oss-20b"
AVAILABLE_MODELS = [
    "openai/gpt-oss-20b",
    "mixtral-8x7b-32768",
]

# API Key Credentials
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
MODEL_NAME = os.getenv("MODEL_NAME", DEFAULT_MODEL)

# Hyperparameters
DEFAULT_TEMPERATURE = 0.0  # Deterministic for agent function calling
MAX_TOKENS = 2048

# System Instructions
DEFAULT_SYSTEM_PROMPT = (
    "You are an expert AI Assistant equipped with function tools. "
    "Reason step-by-step before invoking any tools."
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
