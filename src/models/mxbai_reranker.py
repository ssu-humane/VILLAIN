"""MixedBread mxbai-rerank-large-v1 for text reranking."""


from typing import List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class MxbaiRerankerConfig:
    """Configuration for mxbai reranker."""
    model_name: str = "mixedbread-ai/mxbai-rerank-large-v1"
    batch_size: int = 32
    device: str = "cuda:0"


class MxbaiReranker:
    """MixedBread mxbai-rerank-large-v1 for reranking retrieved passages.
    
    Uses CrossEncoder from sentence_transformers for efficient reranking.
    """
    
    def __init__(self, config: Optional[MxbaiRerankerConfig] = None):
        from sentence_transformers import CrossEncoder
        
        self.config = config or MxbaiRerankerConfig()
        self.device = self.config.device
        
        print(f"[MxbaiReranker] Loading model: {self.config.model_name}")
        
        self.model = CrossEncoder(
            self.config.model_name,
            device=self.device
        )
        
        print(f"[MxbaiReranker] Model loaded successfully")
    
    def to(self, device: str) -> "MxbaiReranker":
        """Move model to specified device."""
        self.device = device
        self.model.model.to(device)
        return self
    
    def rerank(
        self, 
        query: str, 
        documents: List[str], 
        instruction: Optional[str] = None,
        top_k: Optional[int] = None
    ) -> List[Tuple[int, str, float]]:
        """Rerank documents by relevance to query.
        
        Args:
            query: The search query
            documents: List of document texts to rerank
            instruction: Optional custom instruction (not used by mxbai, kept for API compatibility)
            top_k: Return only top-k results (default: return all)
            
        Returns:
            List of (original_index, document, score) tuples, sorted by score descending
        """
        if not documents:
            return []
        
        # Use CrossEncoder's rank method
        results = self.model.rank(
            query, 
            documents, 
            return_documents=True, 
            top_k=top_k if top_k is not None else len(documents),
            batch_size=self.config.batch_size
        )
        
        # Convert to our format: (original_index, document, score)
        output = []
        for result in results:
            output.append((
                result['corpus_id'],  # original index
                result['text'],       # document text
                result['score']       # relevance score
            ))
        
        return output
    
    def rerank_with_indices(
        self,
        query: str,
        documents: List[str],
        original_indices: List[int],
        instruction: Optional[str] = None,
        top_k: Optional[int] = None
    ) -> List[Tuple[int, str, float]]:
        """Rerank documents and return original indices from a larger set.
        
        This is useful when documents are a subset from a larger corpus
        and you need to preserve the original indices.
        
        Args:
            query: The search query
            documents: List of document texts to rerank
            original_indices: Original indices of the documents in the larger corpus
            instruction: Optional custom instruction (not used by mxbai)
            top_k: Return only top-k results
            
        Returns:
            List of (original_corpus_index, document, score) tuples, sorted by score descending
        """
        if not documents:
            return []
        
        results = self.model.rank(
            query,
            documents,
            return_documents=True,
            top_k=top_k if top_k is not None else len(documents),
            batch_size=self.config.batch_size
        )
        
        # Map back to original indices
        output = []
        for result in results:
            local_idx = result['corpus_id']
            output.append((
                original_indices[local_idx],  # map to original index
                result['text'],
                result['score']
            ))
        
        return output

