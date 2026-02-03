"""Nomic nomic-embed-text-v2-moe text embedding model."""

from dataclasses import dataclass
from typing import List, Union

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer


@dataclass
class NomicEmbeddingConfig:
    """Configuration for Nomic Embedding model."""
    model_name: str = "nomic-ai/nomic-embed-text-v2-moe"
    device: str = "cuda"
    max_length: int = 8192


class NomicEmbedding:
    """Nomic nomic-embed-text-v2-moe embedding model.
    
    Uses mean pooling with prefix-based query/document distinction.
    - Queries: prefix with 'search_query: '
    - Documents: prefix with 'search_document: '
    """

    def __init__(self, config: NomicEmbeddingConfig):
        self.config = config
        self.device = config.device
        self.max_length = config.max_length

        print(f"[NomicEmbedding] Loading model: {config.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        self.model = AutoModel.from_pretrained(
            config.model_name, 
            trust_remote_code=True
        )
        self.model.eval()
        
        if self.device != 'cpu':
            self.model.to(self.device)

    @staticmethod
    def _mean_pooling(
        model_output: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """Mean pooling over token embeddings.
        
        Args:
            model_output: Model output containing token embeddings.
            attention_mask: Attention mask from tokenizer.
            
        Returns:
            Mean pooled embeddings.
        """
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
            input_mask_expanded.sum(1), min=1e-9
        )

    def encode_queries(
        self,
        queries: Union[str, List[str]],
    ) -> np.ndarray:
        """Encode queries with 'search_query: ' prefix.
        
        Args:
            queries: Single query string or list of queries.
            
        Returns:
            Normalized embeddings as numpy array.
        """
        if isinstance(queries, str):
            queries = [queries]
        
        # Add query prefix
        prefixed_queries = [f'search_query: {q}' for q in queries]
        
        return self._encode(prefixed_queries)

    def encode_documents(self, documents: Union[str, List[str]]) -> np.ndarray:
        """Encode documents with 'search_document: ' prefix.
        
        Args:
            documents: Single document string or list of documents.
            
        Returns:
            Normalized embeddings as numpy array.
        """
        if isinstance(documents, str):
            documents = [documents]
        
        # Add document prefix
        prefixed_documents = [f'search_document: {d}' for d in documents]
        
        return self._encode(prefixed_documents)

    def _encode(self, texts: List[str]) -> np.ndarray:
        """Encode texts and return normalized embeddings."""
        encoded_input = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        encoded_input = {k: v.to(self.device) for k, v in encoded_input.items()}

        with torch.no_grad():
            model_output = self.model(**encoded_input)
            embeddings = self._mean_pooling(model_output, encoded_input['attention_mask'])
            embeddings = F.normalize(embeddings, p=2, dim=1)

        return embeddings.float().cpu().numpy()

    def to(self, device: str) -> "NomicEmbedding":
        """Move model to specified device."""
        self.device = device
        self.model.to(device)
        return self

