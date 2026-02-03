"""
Agent 3: Cross-Modal Relationship Analysis

Retrieves image evidence items using multiple strategies:
- Top-1 image per claim image (image-to-image retrieval)
- Top-5 images using claim text query (text-to-image retrieval)

Then analyzes cross-modal relationships using outputs from Agent 1 & 2:
- Text-Text: Consistency between text sources
- Image-Text: Alignment between images and text narratives
- Image-Image: Visual consistency between claim and source images

Outputs a global context summary and conflict alerts.
"""

import os
from typing import List, Dict
import numpy as np
import torch
from PIL import Image

from .base_agent import BaseAgent, AgentConfig, AgentAnalysis, EvidenceItem, SharedModels
from .prompts import AGENT3_PROMPT1, AGENT3_PROMPT2, AGENT3_PROMPT3


class ImageImageAgent(BaseAgent):
    """Agent for analyzing image evidence related to claim image."""

    def __init__(
        self,
        config: AgentConfig,
        shared_models: SharedModels = None,
        num_evidence_image: int = 1,  # Top-k per claim image (image-to-image)
        num_evidence_text: int = 5,   # Top-k for claim text query (text-to-image)
    ):
        super().__init__(config, shared_models)
        self.num_evidence_image = num_evidence_image
        self.num_evidence_text = num_evidence_text

    @property
    def name(self) -> str:
        return "image_image_agent"

    def retrieve_similar_images(
        self,
        query_image_path: str,
        claim_id: int,
        top_k: int = 5
    ) -> List[EvidenceItem]:
        """Retrieve top-k similar images based on image embedding similarity.

        Uses pre-computed image embeddings from the vector store.
        """
        if not query_image_path or not os.path.exists(query_image_path):
            return []

        # Load pre-computed image embeddings
        evidence_embeddings, image_paths, image_ids = self.load_image_embeddings(claim_id)

        if evidence_embeddings is None or len(image_paths) == 0:
            return []

        model = self.shared_models.use_image_model()

        try:
            # Load and encode query image
            query_img = Image.open(query_image_path).convert('RGB')
            query_embedding = model.get_image_embeddings([query_img])

            # Convert to numpy if tensor
            if isinstance(query_embedding, torch.Tensor):
                query_embedding = query_embedding.float().cpu().numpy()

            # Release image model after encoding
            self.shared_models.release_image_model()

            # Normalize embeddings
            query_embedding = query_embedding / np.linalg.norm(query_embedding, axis=-1, keepdims=True)
            evidence_embeddings_norm = evidence_embeddings / np.linalg.norm(
                evidence_embeddings, axis=-1, keepdims=True
            )

            # Compute similarities
            similarities = np.dot(query_embedding, evidence_embeddings_norm.T)[0]

            # Get top-k
            ranked_indices = np.argsort(similarities)[::-1][:top_k]

            results = []
            for idx in ranked_indices:
                results.append(EvidenceItem(
                    text="",  # No text for image evidence
                    image_path=image_paths[idx],
                    score=float(similarities[idx]),
                    source='image_image',
                    query=query_image_path,
                    metadata={
                        'image_id': image_ids[idx] if idx < len(image_ids) else None
                    }
                ))

            return results

        except Exception as e:
            self.shared_models.release_image_model()
            print(f"[{self.name}] Error in image retrieval: {e}")
            import traceback
            traceback.print_exc()
            return []

    def retrieve_images_by_text(
        self,
        query_text: str,
        claim_id: int,
        top_k: int = 5
    ) -> List[EvidenceItem]:
        """Retrieve top-k similar images based on text-to-image embedding similarity.

        Uses the same multimodal embedding model to encode text query and match
        against pre-computed image embeddings.
        """
        if not query_text:
            return []

        # Load pre-computed image embeddings
        evidence_embeddings, image_paths, image_ids = self.load_image_embeddings(claim_id)

        if evidence_embeddings is None or len(image_paths) == 0:
            return []

        model = self.shared_models.use_image_model()

        try:
            # Encode query text using multimodal model
            query_embedding = model.get_text_embeddings([query_text])

            # Convert to numpy if tensor
            if isinstance(query_embedding, torch.Tensor):
                query_embedding = query_embedding.float().cpu().numpy()

            # Release image model after encoding
            self.shared_models.release_image_model()

            # Normalize embeddings
            query_embedding = query_embedding / np.linalg.norm(query_embedding, axis=-1, keepdims=True)
            evidence_embeddings_norm = evidence_embeddings / np.linalg.norm(
                evidence_embeddings, axis=-1, keepdims=True
            )

            # Compute similarities
            similarities = np.dot(query_embedding, evidence_embeddings_norm.T)[0]

            # Get top-k
            ranked_indices = np.argsort(similarities)[::-1][:top_k]

            results = []
            for idx in ranked_indices:
                results.append(EvidenceItem(
                    text="",  # No text for image evidence
                    image_path=image_paths[idx],
                    score=float(similarities[idx]),
                    source='text_image',  # Mark as text-to-image retrieval
                    query=query_text,
                    metadata={
                        'image_id': image_ids[idx] if idx < len(image_ids) else None
                    }
                ))

            return results

        except Exception as e:
            self.shared_models.release_image_model()
            print(f"[{self.name}] Error in text-to-image retrieval: {e}")
            import traceback
            traceback.print_exc()
            return []

    def _deduplicate_evidence(
        self,
        evidence_items: List[EvidenceItem]
    ) -> List[EvidenceItem]:
        """Remove duplicate evidence items based on image_path, keeping highest score."""
        seen_paths: Dict[str, EvidenceItem] = {}
        for item in evidence_items:
            path = item.image_path
            if path not in seen_paths or item.score > seen_paths[path].score:
                seen_paths[path] = item
        return list(seen_paths.values())

    def _format_text_sources(
        self,
        text_text_analysis: AgentAnalysis = None,
        image_text_analysis: AgentAnalysis = None
    ) -> str:
        """Format retrieved text sources from Agent 1 and Agent 2 for the prompt.

        Uses the same format as _format_evidence_for_prompt in text_text_agent and image_text_agent.
        """
        evidence_text = ""

        # Add text sources from Agent 1 (Text-Text)
        if text_text_analysis and text_text_analysis.evidence_items:
            for item in text_text_analysis.evidence_items:
                url = item.url or ''
                content = item.text or ''
                evidence_text += f"{url}:\n{content}\n\n"

        # Add text sources from Agent 2 (Image-Text)
        if image_text_analysis and image_text_analysis.evidence_items:
            for item in image_text_analysis.evidence_items:
                url = item.url or ''
                content = item.text or ''
                evidence_text += f"{url}:\n{content}\n\n"

        if not evidence_text:
            raise ValueError("No text sources provided for image-image analysis.")

        return evidence_text.strip()

    def _generate_analysis(
        self,
        claim_text: str,
        claim_images: List[str],
        evidence_items: List[EvidenceItem],
        speaker: str = "Unknown",
        date: str = "Not Specified",
        text_text_analysis: AgentAnalysis = None,
        image_text_analysis: AgentAnalysis = None
    ) -> str:
        """Generate VLM analysis for the retrieved image evidence."""
        if not self.shared_models:
            return "No VLM model available for analysis."

        # Build message content
        content = [{"type": "text", "text": AGENT3_PROMPT1.format(speaker, date, claim_text)}]

        # Add claim images
        valid_images = self.get_valid_claim_images(claim_images)
        for local_id, img in enumerate(valid_images, 1):
            img_path = self.get_full_image_path(img)
            if os.path.exists(img_path):
                content.append({"type": "text", "text": f"\n[CLAIM_IMG_{local_id}]\n"})
                content.append({"type": "image", "image": img_path})

        # Format text sources from Agent 1 and Agent 2
        text_sources_str = self._format_text_sources(text_text_analysis, image_text_analysis)
        content.append({"type": "text", "text": AGENT3_PROMPT2.format(text_sources_str)})

        # Add retrieved evidence images
        for local_id, e in enumerate(evidence_items, 1):
            if e.image_path and os.path.exists(e.image_path):
                content.append({"type": "text", "text": f"\n[RETRIEVED_IMG_{local_id}]\n"})
                content.append({"type": "image", "image": e.image_path})

        content.append({"type": "text", "text": AGENT3_PROMPT3})
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
        text_text_analysis: AgentAnalysis = None,
        image_text_analysis: AgentAnalysis = None,
        **kwargs
    ) -> AgentAnalysis:
        """Analyze claim images and retrieve similar image evidence.

        This agent uses multiple retrieval strategies:
        1. Image-to-image: Retrieve top-1 image for EACH claim image (N images)
        2. Text-to-image: Retrieve top-5 images using claim text as query

        Total: N + 5 images (before deduplication), where N is number of claim images.

        It also receives text evidence from Agent 1 and Agent 2 for
        cross-modal correlation analysis.
        """
        valid_images = self.get_valid_claim_images(claim_images)
        all_evidence_items = []

        # Strategy 1: Retrieve top-k image for EACH claim image (image-to-image)
        if valid_images:
            print(f"[{self.name}] Retrieving top-{self.num_evidence_image} image for each of {len(valid_images)} claim image(s)...")
            for img in valid_images:
                claim_image_path = self.get_full_image_path(img)
                items = self.retrieve_similar_images(
                    query_image_path=claim_image_path,
                    claim_id=claim_id,
                    top_k=self.num_evidence_image
                )
                # Update metadata with query image info
                for item in items:
                    item.metadata['query_image'] = img
                all_evidence_items.extend(items)
            print(f"[{self.name}] Retrieved {len(all_evidence_items)} images from image queries")

        # Strategy 2: Retrieve top-k images using claim text (text-to-image)
        print(f"[{self.name}] Retrieving top-{self.num_evidence_text} images using claim text...")
        text_items = self.retrieve_images_by_text(
            query_text=claim_text,
            claim_id=claim_id,
            top_k=self.num_evidence_text
        )
        # Update metadata with query type info
        for item in text_items:
            item.metadata['query_type'] = 'text'
        all_evidence_items.extend(text_items)
        print(f"[{self.name}] Retrieved {len(text_items)} images from text query")

        # Deduplicate evidence items (keep highest score for each image_path)
        evidence_items = self._deduplicate_evidence(all_evidence_items)
        # Sort by score descending
        evidence_items.sort(key=lambda x: x.score, reverse=True)
        print(f"[{self.name}] Total unique evidence images: {len(evidence_items)}")

        if not evidence_items:
            return AgentAnalysis(
                agent_name=self.name,
                evidence_items=[],
                analysis_text="No image evidence retrieved for this claim."
            )

        # Generate VLM analysis with text sources from Agent 1 and Agent 2
        analysis_text = self._generate_analysis(
            claim_text=claim_text,
            claim_images=claim_images,
            evidence_items=evidence_items,
            speaker=speaker,
            date=date,
            text_text_analysis=text_text_analysis,
            image_text_analysis=image_text_analysis
        )

        return AgentAnalysis(
            agent_name=self.name,
            evidence_items=evidence_items,
            analysis_text=analysis_text,
            metadata={
                'num_retrieved': len(evidence_items),
                'num_claim_images': len(valid_images) if valid_images else 0,
                'num_from_image_query': len(all_evidence_items) - len(text_items),
                'num_from_text_query': len(text_items)
            }
        )

