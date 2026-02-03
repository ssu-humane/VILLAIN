"""Qwen3-Embedding text embedding model."""

from dataclasses import dataclass
from typing import List, Optional, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from transformers import AutoModel, AutoTokenizer


@dataclass
class QwenEmbeddingConfig:
    """Configuration for Qwen3 Embedding model."""
    model_name: str = "Qwen/Qwen3-Embedding-8B"
    device: str = "cuda"
    max_length: int = 8192
    task: str = "Given a web search query, retrieve relevant passages that answer the query"


class Qwen3Embedding:
    """Qwen3-Embedding model for text embeddings.
    
    Uses last-token pooling with instruction format for queries.
    """

    def __init__(self, config: QwenEmbeddingConfig):
        self.config = config
        self.device = config.device
        self.max_length = config.max_length
        self.task = config.task

        print(f"[Qwen3Embedding] Loading model: {config.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.model_name, padding_side='left'
        )
        self.model = AutoModel.from_pretrained(config.model_name)
        self.model.eval()
        
        if self.device != 'cpu':
            self.model.to(self.device)

    @staticmethod
    def _last_token_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
        """Pool the last token representation."""
        left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
        if left_padding:
            return last_hidden_states[:, -1]
        else:
            sequence_lengths = attention_mask.sum(dim=1) - 1
            batch_size = last_hidden_states.shape[0]
            return last_hidden_states[
                torch.arange(batch_size, device=last_hidden_states.device),
                sequence_lengths
            ]

    def encode_queries(
        self,
        queries: Union[str, List[str]],
        task: Optional[str] = None
    ) -> np.ndarray:
        """Encode queries with instruction prefix.
        
        Args:
            queries: Single query string or list of queries.
            task: Optional task instruction. Uses default if not provided.
            
        Returns:
            Normalized embeddings as numpy array.
        """
        if isinstance(queries, str):
            queries = [queries]
        
        task = task or self.task
        formatted_queries = [f'Instruct: {task}\nQuery:{q}' for q in queries]
        
        return self._encode(formatted_queries)

    def encode_documents(self, documents: Union[str, List[str]]) -> np.ndarray:
        """Encode documents without instruction prefix.
        
        Args:
            documents: Single document string or list of documents.
            
        Returns:
            Normalized embeddings as numpy array.
        """
        if isinstance(documents, str):
            documents = [documents]
        
        return self._encode(documents)

    def _encode(self, texts: List[str]) -> np.ndarray:
        """Encode texts and return normalized embeddings."""
        batch_dict = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        batch_dict = {k: v.to(self.device) for k, v in batch_dict.items()}

        with torch.no_grad():
            outputs = self.model(**batch_dict)
            embeddings = self._last_token_pool(
                outputs.last_hidden_state, batch_dict['attention_mask']
            )
            embeddings = F.normalize(embeddings, p=2, dim=1)

        return embeddings.float().cpu().numpy()

    def to(self, device: str) -> "Qwen3Embedding":
        """Move model to specified device."""
        self.device = device
        self.model.to(device)
        return self

