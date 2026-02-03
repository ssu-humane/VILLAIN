"""
Base Agent class for the multi-agent fact-checking pipeline.

Uses Qwen3-VL-8B-Thinking as VLM model for analysis and generation.
Each agent (1-3) does retrieval + VLM analysis.
Agent 4 combines analyses + generates Q&A + justification + veracity prediction.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Optional
import os
import sys
import pickle
import numpy as np
import torch


# Add src directory to path for models import
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models import (
    Qwen3VLModel, GenerationConfig, OpsMMEmbeddingV1,
    GeminiModel, GeminiGenerationConfig,
    Qwen3Reranker, RerankerConfig,
    MxbaiReranker, MxbaiRerankerConfig,
    Qwen3Embedding, QwenEmbeddingConfig,
    MxbaiEmbedding, MxbaiEmbeddingConfig,
    NomicEmbedding, NomicEmbeddingConfig
)


@dataclass
class AgentConfig:
    """Configuration for agents."""
    device: str = "cuda:0"

    # Knowledge store paths
    knowledge_store_path: str = "dataset/AVerImaTeC_Shared_Task/Knowledge_Store/val"
    # Text embedding stores (separate for Agent 1 and Agent 2)
    text_related_store_path: str = "dataset/AVerImaTeC_Shared_Task/Vector_Store/val/text_related/text_related_store_text_val_8B"
    image_related_store_path: str = "dataset/AVerImaTeC_Shared_Task/Vector_Store/val/text_related/image_related_store_text_val_8B"

    # Image embedding store (for Agent 3: image-image retrieval)
    image_embedding_store_path: str = "dataset/AVerImaTeC_Shared_Task/Vector_Store/val/image_related_7B"
    image_dir: str = "dataset/AVerImaTeC/images"

    # Target dataset (val, test, etc.)
    target: str = "val"

    # Legacy: for backward compatibility
    embedding_store_path: str = ""  # Will default to text_related_store_path if empty

    # Model configurations
    text_model: str = "Qwen/Qwen3-Embedding-8B"
    text_model_type: str = "qwen"  # 'qwen', 'mxbai', or 'nomic'
    image_model: str = "OpenSearch-AI/Ops-MM-embedding-v1-7B"
    vlm_model: str = "Qwen/Qwen3-VL-8B-Thinking"  # VLM for all agents
    reranker_model: str = "Qwen/Qwen3-Reranker-8B"  # Reranker model


@dataclass
class EvidenceItem:
    """Single evidence item matching existing pipeline format."""
    text: str = ""
    image_path: Optional[str] = None
    url: str = ""
    score: float = 0.0
    source: str = ""  # 'vqa', 'question', 'claim', 'text_text', 'image_text', 'image_image'
    query: str = ""   # The query used to retrieve this evidence
    metadata: Dict = field(default_factory=dict)


@dataclass
class AgentAnalysis:
    """Analysis result from a single agent."""
    agent_name: str
    evidence_items: List[EvidenceItem] = field(default_factory=list)
    questions: List[str] = field(default_factory=list)  # Generated questions
    answers: List[str] = field(default_factory=list)    # Generated answers
    analysis_text: str = ""
    metadata: Dict = field(default_factory=dict)


class SharedModels:
    """Shared models that are loaded once and passed to all agents.

    VLM stays on GPU always. Other models (text, image, reranker) are moved
    to GPU when used and back to CPU when done to save memory.
    """

    # Gemini model names
    GEMINI_MODELS = {"gemini-2.5-flash", "gemini-2.5-flash-lite", "gemini-2.5-pro",
                     "gemini-2.0-flash", "gemini-2.0-flash-lite",
                     "gemini-3-pro-preview", "gemini-3-flash-preview"}

    def __init__(self, config: AgentConfig):
        self.config = config
        self.device = config.device

        # Check if using Gemini
        self._is_gemini = config.vlm_model in self.GEMINI_MODELS

        # Models (loaded lazily)
        self._vlm = None  # Can be Qwen3VLModel or GeminiModel - stays on GPU
        self._text_model = None  # Loaded on CPU, moved to GPU when needed
        self._text_tokenizer = None
        self._image_model = None  # Loaded on CPU, moved to GPU when needed
        self._reranker = None  # Loaded on CPU, moved to GPU when needed

        # Track which model is currently on GPU (besides VLM)
        self._current_on_gpu = None  # 'text', 'image', 'reranker', or None

    def load_vlm_model(self):
        """Load VLM model (Qwen3-VL or Gemini) for all agents."""
        if self._vlm is None:
            if self._is_gemini:
                # Use Gemini model
                self._vlm = GeminiModel(
                    model_name=self.config.vlm_model,
                    generation_config=GeminiGenerationConfig()
                )
                self._vlm.load()
                return self._vlm.model, None  # Gemini has no processor
            else:
                # Use Qwen3VL model
                self._vlm = Qwen3VLModel(
                    model_name=self.config.vlm_model,
                    device=self.device,
                    generation_config=GenerationConfig()
                )
                self._vlm.load()
                return self._vlm.model, self._vlm.processor

        if self._is_gemini:
            return self._vlm.model, None
        return self._vlm.model, self._vlm.processor

    @property
    def vlm(self):
        """Get the VLM model wrapper (Qwen3VLModel or GeminiModel)."""
        if self._vlm is None:
            self.load_vlm_model()
        return self._vlm

    @property
    def is_gemini(self) -> bool:
        """Check if using Gemini model."""
        return self._is_gemini

    @property
    def is_qwen_thinking(self) -> bool:
        """Check if using Qwen3-VL-Thinking model (which uses <think> tags)."""
        return "Thinking" in self.config.vlm_model and not self._is_gemini

    def _offload_current_model(self):
        """Move current model off GPU to CPU."""
        if self._current_on_gpu == 'text' and self._text_model is not None:
            # Qwen3Embedding and MxbaiEmbedding use .to() method
            self._text_model.to('cpu')
            torch.cuda.empty_cache()
        elif self._current_on_gpu == 'image' and self._image_model is not None:
            # OpsMMEmbeddingV1 uses base_model
            self._image_model.base_model.cpu()
            torch.cuda.empty_cache()
        elif self._current_on_gpu == 'reranker' and self._reranker is not None:
            # MxbaiReranker uses .to(), Qwen3Reranker uses .model
            if self._is_mxbai_reranker():
                self._reranker.to('cpu')
            else:
                self._reranker.model.cpu()
            torch.cuda.empty_cache()
        self._current_on_gpu = None

    def load_text_model(self):
        """Load text embedding model (on CPU initially).

        Supports Qwen, mxbai, and nomic embedding models based on config.text_model_type.
        """
        if self._text_model is None:
            print(f"[SharedModels] Loading text model ({self.config.text_model_type}): {self.config.text_model}")

            if self.config.text_model_type == 'mxbai':
                config = MxbaiEmbeddingConfig(
                    model_name=self.config.text_model,
                    device='cpu'  # Load on CPU first
                )
                self._text_model = MxbaiEmbedding(config)
            elif self.config.text_model_type == 'nomic':
                config = NomicEmbeddingConfig(
                    model_name=self.config.text_model,
                    device='cpu'  # Load on CPU first
                )
                self._text_model = NomicEmbedding(config)
            else:  # default to qwen
                config = QwenEmbeddingConfig(
                    model_name=self.config.text_model,
                    device='cpu'  # Load on CPU first
                )
                self._text_model = Qwen3Embedding(config)

        return self._text_model

    def load_image_model(self):
        """Load multimodal embedding model (on CPU initially)."""
        if self._image_model is None:
            print(f"[SharedModels] Loading image model: {self.config.image_model}")
            # Load to CPU first
            self._image_model = OpsMMEmbeddingV1(
                self.config.image_model,
                device='cpu'
            )
        return self._image_model

    def _is_mxbai_reranker(self) -> bool:
        """Auto-detect if the reranker model is mxbai based on model name."""
        model_name = self.config.reranker_model.lower()
        return 'mxbai' in model_name or 'mixedbread' in model_name

    def load_reranker(self):
        """Load reranker model (on CPU initially).

        Auto-detects reranker type based on model name:
        - mxbai/mixedbread -> MxbaiReranker
        - otherwise -> Qwen3Reranker
        """
        if self._reranker is None:
            is_mxbai = self._is_mxbai_reranker()
            reranker_type = "mxbai" if is_mxbai else "qwen"
            print(f"[SharedModels] Loading reranker ({reranker_type}): {self.config.reranker_model}")

            if is_mxbai:
                self._reranker = MxbaiReranker(
                    MxbaiRerankerConfig(
                        model_name=self.config.reranker_model,
                        device='cpu'
                    )
                )
            else:
                self._reranker = Qwen3Reranker(
                    RerankerConfig(
                        model_name=self.config.reranker_model,
                        device='cpu'
                    )
                )
        return self._reranker

    def use_text_model(self):
        """Get text model, ensuring it's on GPU. Call release_text_model() when done."""
        self.load_text_model()
        if self._current_on_gpu != 'text':
            self._offload_current_model()
            self._text_model.to(self.device)
            self._current_on_gpu = 'text'
        return self._text_model

    def release_text_model(self):
        """Release text model from GPU."""
        if self._current_on_gpu == 'text':
            self._offload_current_model()

    def use_image_model(self):
        """Get image model, ensuring it's on GPU. Call release_image_model() when done."""
        self.load_image_model()
        if self._current_on_gpu != 'image':
            self._offload_current_model()
            # OpsMMEmbeddingV1 uses base_model
            self._image_model.base_model.to(self.device)
            self._image_model.device = self.device
            self._current_on_gpu = 'image'
        return self._image_model

    def release_image_model(self):
        """Release image model from GPU."""
        if self._current_on_gpu == 'image':
            self._offload_current_model()

    def use_reranker(self):
        """Get reranker model, ensuring it's on GPU. Call release_reranker() when done."""
        self.load_reranker()
        if self._current_on_gpu != 'reranker':
            self._offload_current_model()
            # MxbaiReranker uses .to(), Qwen3Reranker uses .model
            if self._is_mxbai_reranker():
                self._reranker.to(self.device)
            else:
                self._reranker.model.to(self.device)
            self._reranker.device = self.device
            self._current_on_gpu = 'reranker'
        return self._reranker

    def release_reranker(self):
        """Release reranker model from GPU."""
        if self._current_on_gpu == 'reranker':
            self._offload_current_model()

    @property
    def reranker(self):
        """Get the reranker model (on GPU). Remember to call release_reranker() when done."""
        return self.use_reranker()

    def generate_with_vlm(
        self,
        messages: List[Dict],
        max_new_tokens: int = 40960,
        temperature: float = 1.0,
        top_p: float = 0.95,
        top_k: int = 20
    ) -> str:
        """Generate text using the VLM model.

        Uses the Qwen3VLModel or GeminiModel wrapper.
        """
        if self._is_gemini:
            # Gemini uses different parameter names
            return self.vlm.generate(
                messages,
                max_new_tokens=max_new_tokens,
                temperature=temperature
            )
        else:
            return self.vlm.generate(
                messages,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k
            )

    def extract_response_after_think(self, output_text: str) -> str:
        """Extract response after </think> tag for Qwen3-VL-Thinking models.

        For Gemini models, returns the output as-is (no thinking tags).
        """
        if self._is_gemini:
            return output_text  # Gemini doesn't use <think> tags
        return self.vlm.extract_response_after_think(output_text)


class BaseAgent(ABC):
    """Abstract base class for all agents in the pipeline."""

    def __init__(self, config: AgentConfig, shared_models: SharedModels = None):
        self.config = config
        self.device = config.device
        self.shared_models = shared_models

    @property
    @abstractmethod
    def name(self) -> str:
        """Agent name identifier."""
        pass

    @abstractmethod
    def analyze(
        self,
        claim_id: int,
        claim_text: str,
        claim_images: List[str],
        **kwargs
    ) -> AgentAnalysis:
        """Analyze the claim and return analysis results."""
        pass

    # ========== Shared Helper Methods ==========

    def load_text_embeddings(self, claim_id: int, store_type: str = "text_related"):
        """Load pre-computed text embeddings for a claim.

        Args:
            claim_id: The claim ID
            store_type: Either "text_related" (for Agent 1) or "image_related" (for Agent 2)
        """
        if store_type == "text_related":
            store_path = self.config.text_related_store_path
        elif store_type == "image_related":
            store_path = self.config.image_related_store_path
        else:
            # Legacy fallback
            store_path = self.config.embedding_store_path or self.config.text_related_store_path

        claim_dir = os.path.join(store_path, str(claim_id))
        embeddings_path = os.path.join(claim_dir, 'embeddings.npy')
        chunks_path = os.path.join(claim_dir, 'chunks.pkl')
        pos_to_id_path = os.path.join(claim_dir, 'pos_to_id.pkl')

        if not os.path.exists(embeddings_path):
            return None, None, None

        embeddings = np.load(embeddings_path)
        with open(chunks_path, 'rb') as f:
            chunks = pickle.load(f)
        with open(pos_to_id_path, 'rb') as f:
            pos_to_id = pickle.load(f)

        return embeddings, chunks, pos_to_id

    def load_image_embeddings(self, claim_id: int):
        """Load pre-computed image embeddings for a claim."""
        embedding_dir = os.path.join(
            self.config.image_embedding_store_path,
            str(claim_id)
        )

        if not os.path.exists(embedding_dir):
            return None, [], []

        embeddings_path = os.path.join(embedding_dir, "image_embeddings.npy")
        paths_path = os.path.join(embedding_dir, "image_paths.pkl")
        ids_path = os.path.join(embedding_dir, "image_ids.pkl")

        if not all(os.path.exists(p) for p in [embeddings_path, paths_path, ids_path]):
            return None, [], []

        embeddings = np.load(embeddings_path)
        with open(paths_path, 'rb') as f:
            image_paths = pickle.load(f)
        with open(ids_path, 'rb') as f:
            image_ids = pickle.load(f)

        return embeddings, image_paths, image_ids

    def get_full_image_path(self, image_filename: str) -> str:
        """Get full path to claim image."""
        return os.path.join(self.config.image_dir, image_filename)

    def get_valid_claim_images(self, claim_images: List[str]) -> List[str]:
        """Filter claim images to only valid ones that exist."""
        return [
            img for img in claim_images
            if os.path.exists(self.get_full_image_path(img))
        ]

    def get_evidence_image_dir(self, claim_id: int) -> str:
        """Get the directory containing evidence images for a claim."""
        return os.path.join(
            self.config.knowledge_store_path,
            'image_related',
            f'image_related_store_image_{self.config.target}',
            str(claim_id)
        )

