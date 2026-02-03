"""
Multi-Agent Pipeline Orchestrator

Coordinates the 5-agent pipeline:
1. Agent 1: Text-Text Analysis (10 evidence) + VLM analysis
2. Agent 2: Image-Text Analysis (10 evidence) + VLM analysis
3. Agent 3: Image-Image Analysis (5 evidence) + VLM analysis
4. Agent 4: QA Generation - Iteratively generates Q-A pairs (max 20, 5 per iteration)
5. Agent 5: Verdict - Selects 10 Q-A pairs, generates justification + veracity prediction

All agents use Qwen3-VL-8B-Thinking for VLM generation.
Models are loaded once and shared across all agents.

Output format (official AVerImaTeC submission format):
- submission.json: List of claims with questions, evidence (Q&A), verdict, justification
"""

import json
import os
from dataclasses import dataclass, field
from typing import List, Dict
from tqdm import tqdm

from .base_agent import AgentConfig, AgentAnalysis, SharedModels
from .text_text_agent import TextTextAgent
from .image_text_agent import ImageTextAgent
from .image_image_agent import ImageImageAgent
from .qa_generation_agent import QAGenerationAgent, QAPair
from .verdict_agent import VerdictAgent


@dataclass
class PipelineConfig:
    """Configuration for the multi-agent pipeline."""
    # Agent-specific evidence counts
    num_text_text_evidence: int = 10
    num_image_text_evidence: int = 10
    # Agent 3: image evidence (separate counts for image and text queries)
    num_image_image_evidence_image: int = 1  # Top-k per claim image (image-to-image)
    num_image_image_evidence_text: int = 5   # Top-k for claim text query (text-to-image)

    # Reranker config (for text retrieval)
    use_reranker: bool = True  # Whether to use reranker for better ranking (enabled by default)
    reranker_fetch_k: int = 50  # Number of candidates to fetch before reranking

    # QA Generation config (Agent 4)
    qa_per_iteration: int = 5
    max_qa_iterations: int = 4
    max_qa_pairs: int = 20

    # Verdict config (Agent 5)
    num_qa_to_select: int = 10

    # Training data for few-shot examples
    train_data_path: str = "dataset/AVerImaTeC/train.json"

    # Base agent config
    agent_config: AgentConfig = field(default_factory=AgentConfig)


@dataclass
class PipelineResult:
    """Result from the complete pipeline."""
    claim_id: int
    claim_text: str
    claim_images: List[str]

    # Individual agent analyses
    text_text_analysis: AgentAnalysis = None
    image_text_analysis: AgentAnalysis = None
    image_image_analysis: AgentAnalysis = None
    qa_generation_analysis: AgentAnalysis = None  # Agent 4
    verdict_analysis: AgentAnalysis = None  # Agent 5

    # Generated Q&A (from Agent 4)
    all_qa_pairs: List[Dict] = field(default_factory=list)

    # Selected Q&A (from Agent 5)
    questions: List[str] = field(default_factory=list)
    answers: List[str] = field(default_factory=list)

    # Veracity prediction (from Agent 5)
    veracity_verdict: str = ""
    justification: str = ""

    # Metadata
    label: str = ""  # Ground truth if available
    speaker: str = ""
    date: str = ""
    location: str = ""


class MultiAgentPipeline:
    """Orchestrates the multi-agent fact-checking pipeline.

    Loads models once and shares them across all agents.
    """

    def __init__(self, config: PipelineConfig):
        self.config = config

        # Create shared models (loaded once)
        print("[Pipeline] Initializing shared models...")
        self.shared_models = SharedModels(config.agent_config)

        # Initialize agents with shared models
        # Agent 1: Text-Text
        self.text_text_agent = TextTextAgent(
            config.agent_config,
            shared_models=self.shared_models,
            num_evidence=config.num_text_text_evidence,
            use_reranker=config.use_reranker,
            reranker_fetch_k=config.reranker_fetch_k,
        )
        # Agent 2: Image-Text
        self.image_text_agent = ImageTextAgent(
            config.agent_config,
            shared_models=self.shared_models,
            num_evidence=config.num_image_text_evidence,
            use_reranker=config.use_reranker,
            reranker_fetch_k=config.reranker_fetch_k,
        )
        # Agent 3: Image-Image
        self.image_image_agent = ImageImageAgent(
            config.agent_config,
            shared_models=self.shared_models,
            num_evidence_image=config.num_image_image_evidence_image,
            num_evidence_text=config.num_image_image_evidence_text,
        )
        # Agent 4: QA Generation (iterative)
        self.qa_generation_agent = QAGenerationAgent(
            config.agent_config,
            shared_models=self.shared_models,
            qa_per_iteration=config.qa_per_iteration,
            max_iterations=config.max_qa_iterations,
            max_qa_pairs=config.max_qa_pairs,
            train_data_path=config.train_data_path
        )
        # Agent 5: Verdict
        self.verdict_agent = VerdictAgent(
            config.agent_config,
            shared_models=self.shared_models,
            num_qa_to_select=config.num_qa_to_select
        )

    def preload_models(self):
        """Preload all models before processing."""
        print("[Pipeline] Preloading models...")
        self.shared_models.load_vlm_model()
        self.shared_models.load_text_model()
        self.shared_models.load_image_model()
        print("[Pipeline] All models loaded.")
    
    def process_claim(
        self,
        claim_id: int,
        claim_text: str,
        claim_images: List[str],
        label: str = "",
        speaker: str = "Unknown",
        date: str = "Not Specified"
    ) -> PipelineResult:
        """Process a single claim through the entire pipeline."""
        result = PipelineResult(
            claim_id=claim_id,
            claim_text=claim_text,
            claim_images=claim_images,
            label=label,
            speaker=speaker,
            date=date
        )

        # Stage 1: Run evidence analysis agents (retrieval + VLM analysis)

        # Agent 1: Text-Text Analysis
        result.text_text_analysis = self.text_text_agent.analyze(
            claim_id=claim_id,
            claim_text=claim_text,
            claim_images=claim_images,
            speaker=speaker,
            date=date,
        )

        # Agent 2: Image-Text Analysis
        result.image_text_analysis = self.image_text_agent.analyze(
            claim_id=claim_id,
            claim_text=claim_text,
            claim_images=claim_images,
            speaker=speaker,
            date=date,
        )

        # Agent 3: Image-Image Analysis (with text sources from Agent 1 & 2)
        result.image_image_analysis = self.image_image_agent.analyze(
            claim_id=claim_id,
            claim_text=claim_text,
            claim_images=claim_images,
            speaker=speaker,
            date=date,
            text_text_analysis=result.text_text_analysis,
            image_text_analysis=result.image_text_analysis
        )

        # Stage 2: Agent 4 - Iterative QA Generation
        print(f"[Pipeline] Agent 4: Generating Q&A pairs for claim {claim_id}")
        result.qa_generation_analysis = self.qa_generation_agent.analyze(
            claim_id=claim_id,
            claim_text=claim_text,
            claim_images=claim_images,
            text_text_analysis=result.text_text_analysis,
            image_text_analysis=result.image_text_analysis,
            image_image_analysis=result.image_image_analysis,
            speaker=speaker,
            date=date
        )

        # Extract all generated QA pairs
        qa_metadata = result.qa_generation_analysis.metadata or {}
        result.all_qa_pairs = qa_metadata.get('qa_pairs', [])

        # Convert to QAPair objects for Agent 5
        qa_pairs_for_verdict = [
            QAPair(question=qa['question'], answer=qa['answer'])
            for qa in result.all_qa_pairs
        ]

        # Stage 3: Agent 5 - Verdict Generation
        print(f"[Pipeline] Agent 5: Generating verdict for claim {claim_id}")
        result.verdict_analysis = self.verdict_agent.analyze(
            claim_id=claim_id,
            claim_text=claim_text,
            claim_images=claim_images,
            qa_pairs=qa_pairs_for_verdict,
            speaker=speaker,
            date=date
        )

        # Extract selected Q&A and veracity from verdict
        if result.verdict_analysis:
            result.questions = result.verdict_analysis.questions
            result.answers = result.verdict_analysis.answers

            # Extract veracity prediction
            metadata = result.verdict_analysis.metadata or {}
            result.veracity_verdict = metadata.get('veracity_verdict', '')
            result.justification = metadata.get('justification', '')

        return result

    def process_batch(
        self,
        samples: List[Dict],
        verbose: bool = True
    ) -> List[PipelineResult]:
        """Process a batch of claims."""
        results = []

        iterator = tqdm(samples, desc="Processing claims") if verbose else samples

        for sample in iterator:
            claim_id = sample.get('claim_id', sample.get('id', len(results)))
            claim_text = sample['claim_text']
            claim_images = sample.get('claim_images', [])
            label = sample.get('label', '')
            speaker = sample.get('speaker', 'Unknown')
            date = sample.get('date', 'Not Specified')

            result = self.process_claim(
                claim_id=claim_id,
                claim_text=claim_text,
                claim_images=claim_images,
                label=label,
                speaker=speaker,
                date=date
            )
            results.append(result)

        return results


def result_to_submission_format(result: PipelineResult) -> Dict:
    """Convert a single PipelineResult to official AVerImaTeC submission format.

    Official format:
    {
        "id": claim_id,
        "questions": ["q1", "q2", ...],
        "justification": "justification text",
        "verdict": "Supported/Refuted/Not Enough Evidence",
        "evidence": [
            {"text": "evidence statement", "images": []},
            ...
        ]
    }
    """
    evidence_list = []
    for answer in result.answers:
        evidence_list.append({
            'text': answer,
            'images': []
        })

    return {
        'id': result.claim_id,
        'questions': result.questions,
        'justification': result.justification or '',
        'verdict': result.veracity_verdict or 'Not Enough Evidence',
        'evidence': evidence_list
    }


def save_pipeline_outputs(results: List[PipelineResult], output_dir: str):
    """Save pipeline outputs in official AVerImaTeC submission format.

    Args:
        results: List of PipelineResult objects
        output_dir: Directory to save outputs

    Output: submission.json
    """
    os.makedirs(output_dir, exist_ok=True)

    submission = [result_to_submission_format(r) for r in results]
    with open(os.path.join(output_dir, 'submission.json'), 'w') as f:
        json.dump(submission, f, indent=2)

    print(f"Saved {len(submission)} claims to {output_dir}/submission.json")
