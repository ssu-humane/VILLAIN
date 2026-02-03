"""
Agent 2: Claim-Image-Related-Text Evidence Analysis

Retrieves 10 text evidence items based on claim image content,
then uses VLM to analyze the visual content against the text sources.
"""

import os

from typing import List
import numpy as np

from .base_agent import BaseAgent, AgentConfig, AgentAnalysis, EvidenceItem, SharedModels
from .prompts import AGENT2_PROMPT1, AGENT2_PROMPT2


def cosine_similarity(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Compute cosine similarity between rows of X and rows of Y.

    Args:
        X: Shape (n_samples_X, n_features)
        Y: Shape (n_samples_Y, n_features)

    Returns:
        Similarity matrix of shape (n_samples_X, n_samples_Y)
    """
    # Normalize X and Y
    X_norm = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-10)
    Y_norm = Y / (np.linalg.norm(Y, axis=1, keepdims=True) + 1e-10)
    return X_norm @ Y_norm.T


class ImageTextAgent(BaseAgent):
    """Agent for analyzing text evidence related to claim image."""

    def __init__(
        self,
        config: AgentConfig,
        shared_models: SharedModels = None,
        num_evidence: int = 10,
        use_reranker: bool = False,
        reranker_fetch_k: int = 50,
    ):
        super().__init__(config, shared_models)
        self.num_evidence = num_evidence
        self.use_reranker = use_reranker  # Whether to use reranker
        self.reranker_fetch_k = reranker_fetch_k  # Candidates to fetch before reranking

    @property
    def name(self) -> str:
        return "image_text_agent"

    def _encode_query(self, query: str) -> np.ndarray:
        """Encode a query using the text embedding model."""
        model = self.shared_models.use_text_model()
        result = model.encode_queries(query)
        self.shared_models.release_text_model()
        return result

    def retrieve_text_evidence(
        self,
        query: str,
        embeddings: np.ndarray,
        chunks: dict,
        pos_to_id: np.ndarray,
        top_k: int = 10
    ) -> List[EvidenceItem]:
        """Retrieve top-k text evidence.

        Retrieval strategies:
        1. use_reranker=True: Get top-N candidates via embedding, then rerank
        2. Default: Simple top-k by cosine similarity
        """
        if embeddings is None or len(embeddings) == 0:
            return []

        query_embedding = self._encode_query(query)
        scores = query_embedding @ embeddings.T
        scores = scores[0]

        if self.use_reranker:
            # Step 1: Get top candidates using embedding similarity
            fetch_k = min(self.reranker_fetch_k, len(embeddings))
            candidate_indices = np.argsort(scores)[-fetch_k:][::-1]

            # Step 2: Build candidate documents for reranking
            candidate_docs = []
            candidate_meta = []
            for idx in candidate_indices:
                if idx >= len(pos_to_id):
                    continue
                chunk_id = pos_to_id[idx]
                if chunk_id not in chunks:
                    continue
                chunk = chunks[chunk_id]
                evidence_text = chunk['page_content']
                candidate_docs.append(evidence_text)
                candidate_meta.append((idx, chunk_id, chunk))

            if not candidate_docs:
                return []

            # Step 3: Rerank
            reranker = self.shared_models.use_reranker()
            rerank_top_k = top_k
            reranked = reranker.rerank(query, candidate_docs, top_k=rerank_top_k)
            self.shared_models.release_reranker()

            # Step 4: Build results from reranked order
            results = []
            for rank, (orig_idx, doc, rerank_score) in enumerate(reranked):
                idx, chunk_id, chunk = candidate_meta[orig_idx]
                evidence_text = chunk['page_content']
                context_before = chunk['metadata'].get('context_before', '')
                context_after = chunk['metadata'].get('context_after', '')
                parts = [p for p in [context_before, evidence_text, context_after] if p]
                full_evidence = ' '.join(parts)

                results.append(EvidenceItem(
                    text=full_evidence,
                    url=chunk['metadata'].get('url', ''),
                    score=rerank_score,
                    source='image_text',
                    query=query,
                    metadata={'chunk_id': chunk_id, 'rank': rank, 'embedding_score': float(scores[idx])}
                ))
            return results
        else:
            # Simple top-k by similarity
            k = min(top_k, len(embeddings))
            selected_indices = np.argsort(scores)[-k:][::-1].tolist()

        # Build results (for simple top-k)
        results = []
        for idx in selected_indices:
            if idx >= len(pos_to_id):
                continue
            chunk_id = pos_to_id[idx]
            if chunk_id not in chunks:
                continue

            chunk = chunks[chunk_id]
            evidence_text = chunk['page_content']
            context_before = chunk['metadata'].get('context_before', '')
            context_after = chunk['metadata'].get('context_after', '')
            parts = [p for p in [context_before, evidence_text, context_after] if p]
            full_evidence = ' '.join(parts)

            results.append(EvidenceItem(
                text=full_evidence,
                url=chunk['metadata'].get('url', ''),
                score=float(scores[idx]),
                source='image_text',
                query=query,
                metadata={'chunk_id': chunk_id, 'rank': len(results)}
            ))

        return results

    def _format_evidence_for_prompt(self, evidence_items: List[EvidenceItem]) -> str:
        """Format evidence items for the VLM prompt."""
        evidence_text = ""
        for local_id, e in enumerate(evidence_items, 1):
            url = e.url or ''
            content = e.text or ''
            evidence_text += f"{url}:\n{content}\n\n"
        return evidence_text

    def _generate_analysis(
        self,
        claim_text: str,
        claim_images: List[str],
        evidence_items: List[EvidenceItem],
        speaker: str = "Unknown",
        date: str = "Not Specified"
    ) -> str:
        """Generate VLM analysis for the retrieved evidence."""
        if not self.shared_models:
            return "No VLM model available for analysis."

        # Format evidence
        evidence_text = self._format_evidence_for_prompt(evidence_items)

        # Build message content
        content = [{"type": "text", "text": AGENT2_PROMPT1.format(speaker, date, claim_text)}]

        # Add claim images (important for this agent)
        valid_images = self.get_valid_claim_images(claim_images)
        for img in valid_images:
            img_path = self.get_full_image_path(img)
            if os.path.exists(img_path):
                content.append({"type": "image", "image": img_path})

        content.append({"type": "text", "text": AGENT2_PROMPT2.format(evidence_text)})
        messages = [{"role": "user", "content": content}]

        # Generate with VLM
        output_text = self.shared_models.generate_with_vlm(messages)

        return output_text

    def analyze(
        self,
        claim_id: int,
        claim_text: str,
        claim_images: List[str],
        speaker: str = "Unknown",
        date: str = "Not Specified",
        **kwargs
    ) -> AgentAnalysis:
        """Analyze claim image and retrieve related text evidence.

        Agent 2 uses image_related_store (image-text retrieval).
        This retrieves text evidence that may be related to the visual content.
        """
        valid_images = self.get_valid_claim_images(claim_images)

        if not valid_images:
            # No valid images, fall back to text-based retrieval
            return AgentAnalysis(
                agent_name=self.name,
                evidence_items=[],
                analysis_text="No valid claim images available for image-text analysis."
            )

        # Load pre-computed text embeddings from image_related store
        embeddings, chunks, pos_to_id = self.load_text_embeddings(claim_id, store_type="image_related")

        if embeddings is None:
            return AgentAnalysis(
                agent_name=self.name,
                evidence_items=[],
                analysis_text="No text embeddings available for this claim."
            )

        # Prepare metadata
        metadata = {
            'num_retrieved': 0,
            'claim_image': valid_images[0] if valid_images else None
        }

        query = claim_text

        # Retrieve text evidence
        evidence_items = self.retrieve_text_evidence(
            query=query,
            embeddings=embeddings,
            chunks=chunks,
            pos_to_id=pos_to_id,
            top_k=self.num_evidence
        )

        # Update metadata to indicate image context
        for item in evidence_items:
            item.metadata['claim_image'] = valid_images[0] if valid_images else None

        metadata['num_retrieved'] = len(evidence_items)

        # Generate VLM analysis
        analysis_text = self._generate_analysis(
            claim_text=claim_text,
            claim_images=claim_images,
            evidence_items=evidence_items,
            speaker=speaker,
            date=date
        )

        return AgentAnalysis(
            agent_name=self.name,
            evidence_items=evidence_items,
            analysis_text=analysis_text,
            metadata=metadata
        )

