"""Gemini model wrapper for the multi-agent pipeline.

Supports both the deprecated google.generativeai API and the new google-genai API.
Uses Gemini 2.5 Flash by default.
"""

import os

import logging
from dataclasses import dataclass
from typing import List, Dict, Optional, Any
from PIL import Image

# Load .env file if python-dotenv is available
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

logger = logging.getLogger(__name__)


@dataclass
class GeminiGenerationConfig:
    """Generation configuration for Gemini models."""
    max_output_tokens: int = 40960
    temperature: float = 1.0


class GeminiModel:
    """Wrapper class for Gemini models using google.generativeai API."""

    # Available Gemini models
    MODELS = {
        "gemini-3-pro-preview": "gemini-3-pro-preview",
        "gemini-3-flash-preview": "gemini-3-flash-preview",
        "gemini-2.5-flash": "gemini-2.5-flash",
        "gemini-2.5-flash-lite": "gemini-2.5-flash-lite",
        "gemini-2.5-pro": "gemini-2.5-pro",
        "gemini-2.0-flash": "gemini-2.0-flash",
        "gemini-2.0-flash-lite": "gemini-2.0-flash-lite",
    }

    def __init__(
        self,
        model_name: str = "gemini-2.5-flash",
        generation_config: Optional[GeminiGenerationConfig] = None
    ):
        self.model_name = model_name
        self.generation_config = generation_config or GeminiGenerationConfig()

        # Get API key from environment (loaded from .env if available)
        self.api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Gemini API key is missing! "
                "Set GEMINI_API_KEY or GOOGLE_API_KEY environment variable or add to .env file."
            )
        
        self._model = None
        self._genai = None
        self._loaded = False
    
    def load(self):
        """Load and configure the Gemini model."""
        if self._loaded:
            return self._model
        
        try:
            import google.generativeai as genai
            self._genai = genai
            
            # Configure the API
            genai.configure(api_key=self.api_key)
            
            # Initialize the model with generation config
            gen_config = {
                "max_output_tokens": self.generation_config.max_output_tokens,
                "temperature": self.generation_config.temperature,
            }
            
            self._model = genai.GenerativeModel(
                model_name=self.model_name,
                generation_config=gen_config
            )
            self._loaded = True
            
            logger.info(f"[GeminiModel] Initialized with model: {self.model_name}")
            
        except ImportError:
            raise ImportError(
                "google-generativeai package not found. "
                "Install with: pip install google-generativeai"
            )
        
        return self._model
    
    @property
    def model(self):
        """Get the model, loading if necessary."""
        if not self._loaded:
            self.load()
        return self._model
    
    def _prepare_content(self, messages: List[Dict]) -> List[Any]:
        """Convert messages format to Gemini content format.
        
        Converts from:
        [{"role": "user", "content": [{"type": "text", "text": "..."}, {"type": "image", "image": "path"}]}]
        
        To Gemini format:
        ["text", PIL.Image, ...]
        """
        contents = []
        
        for msg in messages:
            content = msg.get("content", [])
            if isinstance(content, str):
                contents.append(content)
            elif isinstance(content, list):
                for item in content:
                    if isinstance(item, str):
                        contents.append(item)
                    elif isinstance(item, dict):
                        if item.get("type") == "text":
                            contents.append(item.get("text", ""))
                        elif item.get("type") == "image":
                            img_path = item.get("image", "")
                            if img_path and os.path.exists(img_path):
                                try:
                                    img = Image.open(img_path)
                                    contents.append(img)
                                except Exception as e:
                                    logger.warning(f"Failed to load image {img_path}: {e}")
        
        return contents
    
    def generate(
        self,
        messages: List[Dict],
        generation_config: Optional[GeminiGenerationConfig] = None,
        max_retries: int = 3,
        **kwargs
    ) -> str:
        """Generate text from messages with retry logic.

        Args:
            messages: List of message dicts with 'role' and 'content' keys
            generation_config: Override generation config for this call
            max_retries: Maximum number of retries for empty/blocked responses
            **kwargs: Additional generation parameters

        Returns:
            Generated text string
        """
        import time

        model = self.model

        # Prepare content
        contents = self._prepare_content(messages)

        # Override generation config if provided
        gen_config = None
        if generation_config or kwargs:
            config = generation_config or self.generation_config
            gen_config = {
                "temperature": kwargs.get("temperature", config.temperature),
            }

        last_error = None
        for attempt in range(max_retries):
            try:
                response = model.generate_content(
                    contents,
                    generation_config=gen_config
                )

                # Handle empty or blocked responses
                if not response.candidates:
                    logger.warning(f"[GeminiModel] No candidates in response (attempt {attempt + 1}/{max_retries})")
                    if attempt < max_retries - 1:
                        time.sleep(2 ** attempt)  # Exponential backoff
                        continue
                    return ""

                candidate = response.candidates[0]

                # Check finish reason
                # STOP=1, MAX_TOKENS=2, SAFETY=3, RECITATION=4, OTHER=5
                finish_reason = candidate.finish_reason

                # Try to extract text from parts
                if candidate.content and candidate.content.parts:
                    text_parts = []
                    for part in candidate.content.parts:
                        if hasattr(part, 'text') and part.text:
                            text_parts.append(part.text)
                    if text_parts:
                        return "".join(text_parts)

                # No text content - retry if possible
                logger.warning(f"[GeminiModel] No text in response, finish_reason={finish_reason} (attempt {attempt + 1}/{max_retries})")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                    continue

                # Final attempt failed
                if finish_reason == 3:  # SAFETY
                    return "[Response blocked by safety filters]"
                return ""

            except Exception as e:
                last_error = e
                logger.warning(f"[GeminiModel] Generation error (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                    continue
                raise

        # Should not reach here, but just in case
        if last_error:
            raise last_error
        return ""

