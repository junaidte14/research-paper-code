# core/client.py - Secure Groq API Client Wrapper
from groq import Groq, GroqError
from typing import Optional, Tuple
from config import GROQ_API_KEY, DEFAULT_MODEL

class GroqClientManager:
    """
    Encapsulates Groq API authentication, model selection, and health checks.
    """
    def __init__(self, api_key: Optional[str] = None, model_name: str = DEFAULT_MODEL):
        self.api_key = api_key or GROQ_API_KEY
        self.model_name = model_name
        self._client: Optional[Groq] = None
        
        if self.api_key:
            self.init_client()

    def init_client(self) -> bool:
        """Instantiates the official Groq SDK client."""
        try:
            if not self.api_key:
                self._client = None
                return False
            self._client = Groq(api_key=self.api_key)
            return True
        except Exception as e:
            print(f"Client init error: {e}")
            self._client = None
            return False

    @property
    def client(self) -> Groq:
        """Returns the raw Groq client instance."""
        if not self._client:
            if not self.api_key:
                raise ValueError("Groq API key is missing. Please provide a valid key.")
            self.init_client()
            if not self._client:
                raise ValueError("Failed to initialize Groq client.")
        return self._client

    @client.setter
    def client(self, instance: Optional[Groq]) -> None:
        """Allows direct assignment or override of the client instance."""
        self._client = instance

    def get_client(self) -> Groq:
        """Helper method expected by AgentEngine to retrieve the Groq client instance."""
        return self.client

    def test_connection(self, model_name: Optional[str] = None) -> Tuple[bool, str]:
        """
        Performs a lightweight ping to verify API key validity.
        Returns: (success: bool, message: str)
        """
        target_model = model_name or self.model_name
        
        try:
            groq_client = self.client
        except Exception as e:
            return False, f"API Key is missing or invalid: {str(e)}"

        try:
            response = groq_client.chat.completions.create(
                model=target_model,
                messages=[{"role": "user", "content": "Ping"}],
                max_tokens=5
            )
            return True, f"Successfully connected to {target_model}!"
        except GroqError as e:
            return False, f"Groq API Error: {str(e)}"
        except Exception as e:
            return False, f"Unexpected Error: {str(e)}"