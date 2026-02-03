"""
Multi-Agent Fact-Checking Pipeline

This module implements a 5-agent pipeline for multimodal fact-checking:
- Agent 1: Claim-Text-Related-Text Evidence Analysis (10 evidence) + VLM analysis
- Agent 2: Claim-Image-Related-Text Evidence Analysis (10 evidence) + VLM analysis
- Agent 3: Cross-Modal Relationship Analysis - Uses Agent 1 & 2 outputs + retrieved images
           to analyze text-text, image-text, and image-image relationships
- Agent 4: QA Generation Agent - Iteratively generates Q-A pairs (max 20, 5 per iteration)
- Agent 5: Verdict Agent - Selects 10 Q-A pairs, generates justification and veracity prediction

All agents use Qwen3-VL-8B-Thinking for VLM generation.
Models are loaded once and shared across all agents.

Outputs:
- generated_questions.json
- retrieved_evidence.json
- predictions.json (from Agent 5, includes veracity prediction)
"""

from .base_agent import BaseAgent, AgentConfig, EvidenceItem, AgentAnalysis, SharedModels
from .text_text_agent import TextTextAgent
from .image_text_agent import ImageTextAgent
from .image_image_agent import ImageImageAgent
from .qa_generation_agent import QAGenerationAgent, QAPair, QAGenerationResult
from .verdict_agent import VerdictAgent, VerdictResult
from .pipeline import (
    MultiAgentPipeline,
    PipelineConfig,
    PipelineResult,
    save_pipeline_outputs
)

__all__ = [
    'BaseAgent',
    'AgentConfig',
    'EvidenceItem',
    'AgentAnalysis',
    'SharedModels',
    'TextTextAgent',
    'ImageTextAgent',
    'ImageImageAgent',
    'QAGenerationAgent',
    'QAPair',
    'QAGenerationResult',
    'VerdictAgent',
    'VerdictResult',
    'MultiAgentPipeline',
    'PipelineConfig',
    'PipelineResult',
    'save_pipeline_outputs'
]

