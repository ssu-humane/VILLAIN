"""Qwen3-Reranker-8B for text reranking."""

import torch
from typing import List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class RerankerConfig:
    """Configuration for Qwen3-Reranker."""
    model_name: str = "Qwen/Qwen3-Reranker-8B"
    max_length: int = 8192
    batch_size: int = 8
    device: str = "cuda:0"
    torch_dtype: torch.dtype = torch.float16


class Qwen3Reranker:
    """Qwen3-Reranker-8B for reranking retrieved passages."""
    
    def __init__(self, config: Optional[RerankerConfig] = None):
        from transformers import AutoTokenizer, AutoModelForCausalLM
        
        self.config = config or RerankerConfig()
        self.device = self.config.device
        
        print(f"[Qwen3Reranker] Loading model: {self.config.model_name}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name, 
            padding_side='left'
        )
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            torch_dtype=self.config.torch_dtype,
        ).to(self.device).eval()
        
        # Token IDs for yes/no
        self.token_false_id = self.tokenizer.convert_tokens_to_ids("no")
        self.token_true_id = self.tokenizer.convert_tokens_to_ids("yes")
        
        # Prompt template
        self.prefix = '<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>\n<|im_start|>user\n'
        self.suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        self.prefix_tokens = self.tokenizer.encode(self.prefix, add_special_tokens=False)
        self.suffix_tokens = self.tokenizer.encode(self.suffix, add_special_tokens=False)
        
        print(f"[Qwen3Reranker] Model loaded successfully")
    
    def _format_instruction(self, query: str, doc: str, instruction: Optional[str] = None) -> str:
        """Format input for reranker."""
        if instruction is None:
            instruction = "Given a web search query, retrieve relevant passages that answer the query"
        return f"<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {doc}"
    
    def _process_inputs(self, pairs: List[str]):
        """Tokenize and process input pairs."""
        inputs = self.tokenizer(
            pairs, 
            padding=False, 
            truncation='longest_first',
            return_attention_mask=False, 
            max_length=self.config.max_length - len(self.prefix_tokens) - len(self.suffix_tokens)
        )
        
        for i, ele in enumerate(inputs['input_ids']):
            inputs['input_ids'][i] = self.prefix_tokens + ele + self.suffix_tokens
        
        inputs = self.tokenizer.pad(
            inputs, 
            padding=True, 
            return_tensors="pt", 
            max_length=self.config.max_length
        )
        
        for key in inputs:
            inputs[key] = inputs[key].to(self.device)
        
        return inputs
    
    @torch.no_grad()
    def _compute_logits(self, inputs) -> List[float]:
        """Compute relevance scores from model logits."""
        batch_scores = self.model(**inputs).logits[:, -1, :]
        true_vector = batch_scores[:, self.token_true_id]
        false_vector = batch_scores[:, self.token_false_id]
        batch_scores = torch.stack([false_vector, true_vector], dim=1)
        batch_scores = torch.nn.functional.log_softmax(batch_scores, dim=1)
        scores = batch_scores[:, 1].exp().tolist()
        return scores
    
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
            instruction: Optional custom instruction
            top_k: Return only top-k results (default: return all)
            
        Returns:
            List of (original_index, document, score) tuples, sorted by score descending
        """
        if not documents:
            return []
        
        all_scores = []
        
        # Process in batches
        for i in range(0, len(documents), self.config.batch_size):
            batch_docs = documents[i:i + self.config.batch_size]
            pairs = [self._format_instruction(query, doc, instruction) for doc in batch_docs]
            inputs = self._process_inputs(pairs)
            batch_scores = self._compute_logits(inputs)
            all_scores.extend(batch_scores)
        
        # Create results with original indices
        results = [(i, doc, score) for i, (doc, score) in enumerate(zip(documents, all_scores))]
        
        # Sort by score descending
        results.sort(key=lambda x: x[2], reverse=True)
        
        if top_k is not None:
            results = results[:top_k]
        
        return results

