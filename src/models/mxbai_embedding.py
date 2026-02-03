"""MixedBread mxbai-embed-large-v1 text embedding model."""

from dataclasses import dataclass
from typing import List, Union

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer


@dataclass
class MxbaiEmbeddingConfig:
    """Configuration for MixedBread Embedding model."""
    model_name: str = "mixedbread-ai/mxbai-embed-large-v1"
    device: str = "cuda"
    max_length: int = 512
    pooling_strategy: str = "cls"  # 'cls' or 'mean'


class MxbaiEmbedding:
    """MixedBread mxbai-embed-large-v1 embedding model.
    
    Uses CLS pooling by default for best performance.
    """

    def __init__(self, config: MxbaiEmbeddingConfig):
        self.config = config
        self.device = config.device
        self.max_length = config.max_length
        self.pooling_strategy = config.pooling_strategy

        print(f"[MxbaiEmbedding] Loading model: {config.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        self.model = AutoModel.from_pretrained(config.model_name)
        self.model.eval()
        
        if self.device != 'cpu':
            self.model.to(self.device)

    @staticmethod
    def _pooling(
        outputs: torch.Tensor,
        attention_mask: torch.Tensor,
        strategy: str = 'cls'
    ) -> np.ndarray:
        """Pool hidden states to get embeddings.
        
        Args:
            outputs: Last hidden states from model.
            attention_mask: Attention mask from tokenizer.
            strategy: 'cls' for CLS token pooling, 'mean' for mean pooling.
            
        Returns:
            Pooled embeddings as numpy array.
        """
        if strategy == 'cls':
            outputs = outputs[:, 0]
        elif strategy == 'mean':
            outputs = torch.sum(
                outputs * attention_mask[:, :, None], dim=1
            ) / torch.sum(attention_mask, dim=1, keepdim=True)
        else:
            raise ValueError(f"Unknown pooling strategy: {strategy}. Use 'cls' or 'mean'.")
        return outputs.detach().cpu().numpy()

    @staticmethod
    def transform_query(query: str) -> str:
        """Transform query with retrieval prompt prefix.
        
        For retrieval, queries should use this prompt for best results.
        """
        return f'Represent this sentence for searching relevant passages: {query}'

    def encode_queries(
        self,
        queries: Union[str, List[str]],
        add_prompt: bool = True
    ) -> np.ndarray:
        """Encode queries for retrieval.
        
        Args:
            queries: Single query string or list of queries.
            add_prompt: Whether to add the retrieval prompt prefix.
            
        Returns:
            Normalized embeddings as numpy array.
        """
        if isinstance(queries, str):
            queries = [queries]
        
        if add_prompt:
            queries = [self.transform_query(q) for q in queries]
        
        return self._encode(queries)

    def encode_documents(self, documents: Union[str, List[str]]) -> np.ndarray:
        """Encode documents (no prompt prefix).
        
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
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs).last_hidden_state
            embeddings = self._pooling(
                outputs, inputs['attention_mask'], self.pooling_strategy
            )
            # Normalize embeddings
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            embeddings = embeddings / norms

        return embeddings

    def to(self, device: str) -> "MxbaiEmbedding":
        """Move model to specified device."""
        self.device = device
        self.model.to(device)
        return self

