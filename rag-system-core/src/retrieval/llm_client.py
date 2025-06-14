"""
LLM Client
Multi-provider LLM client for text generation
"""
import logging
import os
import time
from typing import Optional, Dict, Any
from abc import ABC, abstractmethod

from ..core.error_handling import LLMError, APIKeyError

class BaseLLMClient(ABC):
    """Base class for LLM clients"""
    
    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        pass

class GroqClient(BaseLLMClient):
    """Groq LLM client"""
    
    def __init__(self, api_key: str, model_name: str = "mixtral-8x7b-32768", timeout: int = 30):
        self.api_key = api_key
        self.model_name = model_name
        self.timeout = timeout
        self.client = None
        self._initialize_client()
    
    def _initialize_client(self):
        try:
            import groq
            self.client = groq.Groq(
                api_key=self.api_key,
                timeout=self.timeout
            )
        except ImportError:
            raise LLMError("Groq package not installed")
        except Exception as e:
            raise APIKeyError("groq")
    
    def generate(self, prompt: str, max_tokens: int = 1000, temperature: float = 0.1) -> str:
        try:
            start_time = time.time()
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=temperature,
                timeout=self.timeout
            )
            elapsed = time.time() - start_time
            logging.debug(f"Groq API call took {elapsed:.2f} seconds")
            return response.choices[0].message.content
        except Exception as e:
            logging.error(f"Groq generation error: {e}")
            raise LLMError(f"Groq generation failed: {e}", provider="groq", model=self.model_name)

class OpenAIClient(BaseLLMClient):
    """OpenAI LLM client"""
    
    def __init__(self, api_key: str, model_name: str = "gpt-3.5-turbo", timeout: int = 30):
        self.api_key = api_key
        self.model_name = model_name
        self.timeout = timeout
        self.client = None
        self._initialize_client()
    
    def _initialize_client(self):
        try:
            import openai
            self.client = openai.OpenAI(
                api_key=self.api_key,
                timeout=self.timeout
            )
        except ImportError:
            raise LLMError("OpenAI package not installed")
        except Exception as e:
            raise APIKeyError("openai")
    
    def generate(self, prompt: str, max_tokens: int = 1000, temperature: float = 0.1) -> str:
        try:
            start_time = time.time()
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=temperature,
                timeout=self.timeout
            )
            elapsed = time.time() - start_time
            logging.debug(f"OpenAI API call took {elapsed:.2f} seconds")
            return response.choices[0].message.content
        except Exception as e:
            logging.error(f"OpenAI generation error: {e}")
            raise LLMError(f"OpenAI generation failed: {e}", provider="openai", model=self.model_name)

class LLMClient:
    """Main LLM client with provider switching"""
    
    def __init__(self, provider: str = "groq", model_name: str = None, 
                 api_key: str = None, temperature: float = 0.1, max_tokens: int = 1000,
                 timeout: int = 30):
        self.provider = provider
        self.model_name = model_name
        self.api_key = api_key or self._get_api_key(provider)
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.client = None
        
        self._initialize_client()
        logging.info(f"LLM client initialized: {provider} (timeout: {timeout}s)")
    
    def _get_api_key(self, provider: str) -> str:
        """Get API key from environment"""
        env_keys = {
            "groq": "GROQ_API_KEY",
            "openai": "OPENAI_API_KEY",
            "anthropic": "ANTHROPIC_API_KEY"
        }
        
        env_key = env_keys.get(provider)
        if not env_key:
            raise LLMError(f"Unknown provider: {provider}")
        
        api_key = os.getenv(env_key)
        if not api_key:
            raise APIKeyError(provider)
        
        return api_key
    
    def _initialize_client(self):
        """Initialize the appropriate client"""
        if self.provider == "groq":
            model = self.model_name or "meta-llama/llama-4-maverick-17b-128e-instruct"
            self.client = GroqClient(self.api_key, model, self.timeout)
        elif self.provider == "openai":
            model = self.model_name or "gpt-3.5-turbo"
            self.client = OpenAIClient(self.api_key, model, self.timeout)
        else:
            raise LLMError(f"Unsupported provider: {self.provider}")
    
    def generate(self, prompt: str, max_tokens: Optional[int] = None, 
                temperature: Optional[float] = None) -> str:
        """Generate text using the configured LLM"""
        max_tokens = max_tokens or self.max_tokens
        temperature = temperature or self.temperature
        
        try:
            return self.client.generate(prompt, max_tokens=max_tokens, temperature=temperature)
        except Exception as e:
            logging.error(f"LLM generation failed: {e}")
            raise
    
    def test_connection(self) -> bool:
        """Test LLM connection"""
        try:
            response = self.generate("Hello", max_tokens=5)
            return bool(response)
        except Exception as e:
            logging.error(f"LLM connection test failed: {e}")
            return False 