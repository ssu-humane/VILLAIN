"""Qwen3-VL model wrapper for the multi-agent pipeline."""

import torch
from dataclasses import dataclass
from typing import List, Dict, Optional
from transformers import AutoModelForVision2Seq, AutoProcessor


@dataclass
class GenerationConfig:
    """Generation configuration for Qwen3-VL models."""
    max_new_tokens: int = 40960
    temperature: float = 1.0
    top_p: float = 0.95
    top_k: int = 20
    repetition_penalty: float = 1.0
    do_sample: bool = True


class Qwen3VLModel:
    """Wrapper class for Qwen3-VL models."""
    
    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-VL-8B-Thinking",
        device: str = "cuda:0",
        generation_config: Optional[GenerationConfig] = None
    ):
        self.model_name = model_name
        self.device = device
        self.generation_config = generation_config or GenerationConfig()
        
        self._model = None
        self._processor = None
    
    def load(self):
        """Load the model and processor."""
        if self._model is None:
            print(f"[Qwen3VLModel] Loading model: {self.model_name}")
            # Use AutoModelForVision2Seq which will load the correct architecture from config
            self._model = AutoModelForVision2Seq.from_pretrained(
                self.model_name,
                torch_dtype=torch.bfloat16,
                device_map=self.device,
                trust_remote_code=True  # Required for Qwen3-VL
            )
            self._model.eval()

            self._processor = AutoProcessor.from_pretrained(
                self.model_name,
                trust_remote_code=True
            )
            print(f"[Qwen3VLModel] Model loaded on {self.device}")

        return self._model, self._processor
    
    @property
    def model(self):
        """Get the model, loading if necessary."""
        if self._model is None:
            self.load()
        return self._model
    
    @property
    def processor(self):
        """Get the processor, loading if necessary."""
        if self._processor is None:
            self.load()
        return self._processor
    
    def generate(
        self,
        messages: List[Dict],
        generation_config: Optional[GenerationConfig] = None,
        **kwargs
    ) -> str:
        """Generate text from messages.
        
        Args:
            messages: List of message dicts with 'role' and 'content' keys
            generation_config: Override generation config for this call
            **kwargs: Additional generation parameters to override
            
        Returns:
            Generated text string
        """
        model, processor = self.load()
        config = generation_config or self.generation_config
        
        # Prepare inputs
        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt"
        ).to(self.device)
        
        # Merge config with any overrides
        gen_kwargs = {
            "max_new_tokens": kwargs.get("max_new_tokens", config.max_new_tokens),
            "temperature": kwargs.get("temperature", config.temperature),
            "top_p": kwargs.get("top_p", config.top_p),
            "top_k": kwargs.get("top_k", config.top_k),
            "repetition_penalty": kwargs.get("repetition_penalty", config.repetition_penalty),
            "do_sample": kwargs.get("do_sample", config.do_sample),
        }
        
        with torch.no_grad():
            generated_ids = model.generate(**inputs, **gen_kwargs)
        
        # Trim input tokens from output
        generated_ids_trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        
        output_text = processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]
        
        return output_text
    
    def extract_response_after_think(self, output_text: str) -> str:
        """Extract response after </think> tag for Qwen3-VL-Thinking models."""
        if "Thinking" in self.model_name and "</think>" in output_text:
            return output_text.split("</think>")[-1].strip()
        return output_text
    
    def generate_and_extract(
        self,
        messages: List[Dict],
        generation_config: Optional[GenerationConfig] = None,
        **kwargs
    ) -> str:
        """Generate text and extract response after thinking (if applicable).
        
        Args:
            messages: List of message dicts
            generation_config: Override generation config
            **kwargs: Additional generation parameters
            
        Returns:
            Extracted response text
        """
        output = self.generate(messages, generation_config, **kwargs)
        return self.extract_response_after_think(output)

