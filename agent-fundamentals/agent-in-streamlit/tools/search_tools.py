# tools/search_tools.py - Knowledge Base & Mock Search Tool
import json
from tools.base import tool

MOCK_DATABASE = {
    "groq": "Groq is an AI infrastructure company that builds LPU Inference Engines for ultra-low latency.",
    "streamlit": "Streamlit is an open-source Python framework to turn data scripts into web applications.",
    "weather_tokyo": "Current weather in Tokyo: 22°C (72°F), Clear Skies, Humidity 45%."
}

@tool(description="Queries the internal local knowledge base for info about Groq, Streamlit, or local weather.")
def search_knowledge_base(query: str) -> str:
    query_clean = query.lower().strip()
    for key, content in MOCK_DATABASE.items():
        if key in query_clean:
            return json.dumps({"status": "found", "query": query, "data": content})
    
    return json.dumps({"status": "not_found", "message": f"No relevant records found for '{query}'"})
