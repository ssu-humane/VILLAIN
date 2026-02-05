#!/usr/bin/env python3
"""
Multi-Agent Fact-Checking Pipeline Runner
"""

import argparse
import json
import os
import sys
from typing import Any, Dict

import yaml

# Add src to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agents import MultiAgentPipeline
from agents.pipeline import PipelineConfig, result_to_submission_format
from agents.base_agent import AgentConfig


def load_yaml_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def get_nested(d: Dict, *keys, default=None):
    """Get nested dictionary value safely."""
    for key in keys:
        if isinstance(d, dict):
            d = d.get(key, default)
        else:
            return default
    return d


def to_json_serializable(obj):
    """Convert numpy/torch types to JSON serializable Python types."""
    import numpy as np
    if isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: to_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [to_json_serializable(v) for v in obj]
    return obj


def evidence_item_to_dict(item) -> Dict:
    """Convert EvidenceItem to serializable dict."""
    return {
        'text': item.text,
        'image_path': item.image_path,
        'url': item.url,
        'score': to_json_serializable(item.score),
        'source': item.source,
        'query': item.query,
        'metadata': to_json_serializable(item.metadata)
    }


def agent_analysis_to_dict(analysis) -> Dict:
    """Convert AgentAnalysis to serializable dict."""
    if analysis is None:
        return None
    return {
        'agent_name': analysis.agent_name,
        'evidence_items': [evidence_item_to_dict(e) for e in analysis.evidence_items],
        'questions': analysis.questions,
        'answers': analysis.answers,
        'analysis_text': analysis.analysis_text,
        'metadata': to_json_serializable(analysis.metadata)
    }


def save_per_claim_outputs(result, output_dir: str, save_intermediate: bool = False):
    """Save outputs for a single claim in official submission format.

    Output: {output_dir}/submission/{claim_id}/submission.json
    If save_intermediate=True, also saves individual agent outputs.
    """
    claim_id = result.claim_id
    submission_dir = os.path.join(output_dir, 'submission', str(claim_id))
    os.makedirs(submission_dir, exist_ok=True)

    submission = result_to_submission_format(result)
    with open(os.path.join(submission_dir, 'submission.json'), 'w') as f:
        json.dump(submission, f, indent=2)

    # Save intermediate outputs if enabled
    if save_intermediate:
        # Agent 1: text-text
        if result.text_text_analysis:
            agent1_output = agent_analysis_to_dict(result.text_text_analysis)
            with open(os.path.join(submission_dir, 'agent1_text_text.json'), 'w') as f:
                json.dump(agent1_output, f, indent=2)

        # Agent 2: image-text
        if result.image_text_analysis:
            agent2_output = agent_analysis_to_dict(result.image_text_analysis)
            with open(os.path.join(submission_dir, 'agent2_image_text.json'), 'w') as f:
                json.dump(agent2_output, f, indent=2)

        # Agent 3: image-image
        if result.image_image_analysis:
            agent3_output = agent_analysis_to_dict(result.image_image_analysis)
            with open(os.path.join(submission_dir, 'agent3_image_image.json'), 'w') as f:
                json.dump(agent3_output, f, indent=2)

        # Agent 4: QA generation
        if result.qa_generation_analysis:
            agent4_output = agent_analysis_to_dict(result.qa_generation_analysis)
            agent4_output['all_qa_pairs'] = result.all_qa_pairs
            with open(os.path.join(submission_dir, 'agent4_qa_generation.json'), 'w') as f:
                json.dump(agent4_output, f, indent=2)

        # Agent 5: verdict
        if result.verdict_analysis:
            agent5_output = agent_analysis_to_dict(result.verdict_analysis)
            agent5_output['veracity_verdict'] = result.veracity_verdict
            agent5_output['justification'] = result.justification
            agent5_output['selected_questions'] = result.questions
            agent5_output['selected_answers'] = result.answers
            with open(os.path.join(submission_dir, 'agent5_verdict.json'), 'w') as f:
                json.dump(agent5_output, f, indent=2)


def main():
    parser = argparse.ArgumentParser(
        description="Multi-Agent Fact-Checking Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Using YAML config (recommended)
  python src/run_multi_agent.py --config scripts/cfg/default.yaml

  # Override specific options via CLI
  python src/run_multi_agent.py --config scripts/cfg/default.yaml --start_idx 0 --end_idx 10

  # Legacy: CLI arguments only (backward compatible)
  python src/run_multi_agent.py --data_path dataset/AVerImaTeC/val.json --output_dir outputs-tmp
        """
    )

    # Config file (primary way to configure)
    parser.add_argument('--config', type=str, default=None,
                        help='Path to YAML config file (e.g., scripts/cfg/default.yaml)')

    # CLI overrides (can override config file values)
    parser.add_argument('--data_path', type=str, default=None, help='Path to dataset')
    parser.add_argument('--output_dir', type=str, default=None, help='Output directory')
    parser.add_argument('--start_idx', type=int, default=None, help='Start index (inclusive)')
    parser.add_argument('--end_idx', type=int, default=None, help='End index (exclusive)')
    parser.add_argument('--max_samples', type=int, default=None, help='Max samples to process')
    parser.add_argument('--preload_models', action='store_true', help='Preload all models')
    parser.add_argument('--device', type=str, default=None, help='Device to run on')
    parser.add_argument('--target', type=str, default=None, help='Target dataset (val, test)')

    args = parser.parse_args()

    # Load config from YAML or use defaults
    if args.config:
        print(f"Loading config from: {args.config}")
        cfg = load_yaml_config(args.config)
    else:
        # Use empty config, will fall back to defaults
        cfg = {}

    # Helper to get config value with CLI override support
    def get_cfg(cli_val, *yaml_keys, default=None):
        """Get config value: CLI override > YAML config > default."""
        if cli_val is not None:
            return cli_val
        yaml_val = get_nested(cfg, *yaml_keys, default=None)
        return yaml_val if yaml_val is not None else default

    # Build configuration from YAML with CLI overrides
    # Data paths
    data_path = get_cfg(args.data_path, 'data', 'data_path', default='dataset/AVerImaTeC/val.json')
    output_dir = get_cfg(args.output_dir, 'data', 'output_dir', default='outputs-multi-agent')
    train_data_path = get_nested(cfg, 'data', 'train_data_path', default='dataset/AVerImaTeC/train.json')
    image_dir = get_nested(cfg, 'data', 'image_dir', default='dataset/AVerImaTeC/images')
    target = get_cfg(args.target, 'data', 'target', default='val')

    # Store paths
    knowledge_store_path = get_nested(cfg, 'stores', 'knowledge_store_path',
                                       default='dataset/AVerImaTeC_Shared_Task/Knowledge_Store/val')
    text_related_store_path = get_nested(cfg, 'stores', 'text_related_store_path',
                                          default='dataset/AVerImaTeC_Shared_Task/Vector_Store/val/text_related/text_related_store_text_val_8B')
    image_related_store_path = get_nested(cfg, 'stores', 'image_related_store_path',
                                           default='dataset/AVerImaTeC_Shared_Task/Vector_Store/val/text_related/image_related_store_text_val_8B')
    image_embedding_store_path = get_nested(cfg, 'stores', 'image_embedding_store_path',
                                             default='dataset/AVerImaTeC_Shared_Task/Vector_Store/val/image_related_7B')

    # Model configuration
    device = get_cfg(args.device, 'models', 'device', default='cuda:0')
    text_model = get_nested(cfg, 'models', 'text_model', default='Qwen/Qwen3-Embedding-8B')
    text_model_type = get_nested(cfg, 'models', 'text_model_type', default='qwen')
    image_model = get_nested(cfg, 'models', 'image_model', default='OpenSearch-AI/Ops-MM-embedding-v1-7B')
    vlm_model = get_nested(cfg, 'models', 'vlm_model', default='Qwen/Qwen3-VL-8B-Thinking')
    reranker_model = get_nested(cfg, 'models', 'reranker_model', default='Qwen/Qwen3-Reranker-8B')

    # Evidence configuration
    num_text_text_evidence = get_nested(cfg, 'evidence', 'num_text_text_evidence', default=10)
    num_image_text_evidence = get_nested(cfg, 'evidence', 'num_image_text_evidence', default=10)
    # Agent 3: separate counts for image and text queries
    num_image_image_evidence_image = get_nested(cfg, 'evidence', 'num_image_image_evidence_image', default=1)
    num_image_image_evidence_text = get_nested(cfg, 'evidence', 'num_image_image_evidence_text', default=5)

    # QA generation configuration
    qa_per_iteration = get_nested(cfg, 'qa_generation', 'qa_per_iteration', default=5)
    max_qa_iterations = get_nested(cfg, 'qa_generation', 'max_qa_iterations', default=4)
    max_qa_pairs = get_nested(cfg, 'qa_generation', 'max_qa_pairs', default=20)
    num_qa_to_select = get_nested(cfg, 'qa_generation', 'num_qa_to_select', default=10)

    # Retrieval options (MMR disabled by default, reranker enabled by default)
    use_reranker = get_nested(cfg, 'retrieval', 'use_reranker', default=True)
    reranker_fetch_k = get_nested(cfg, 'retrieval', 'reranker_fetch_k', default=100)

    # Processing options
    preload_models = args.preload_models or get_nested(cfg, 'processing', 'preload_models', default=False)
    max_samples = get_cfg(args.max_samples, 'processing', 'max_samples', default=None)
    start_idx = get_cfg(args.start_idx, 'processing', 'start_idx', default=None)
    end_idx = get_cfg(args.end_idx, 'processing', 'end_idx', default=None)
    save_intermediate_output = get_nested(cfg, 'processing', 'save_intermediate_output', default=False)

    # Create agent config
    agent_config = AgentConfig(
        device=device,
        knowledge_store_path=knowledge_store_path,
        text_related_store_path=text_related_store_path,
        image_related_store_path=image_related_store_path,
        image_embedding_store_path=image_embedding_store_path,
        image_dir=image_dir,
        target=target,
        text_model=text_model,
        text_model_type=text_model_type,
        image_model=image_model,
        vlm_model=vlm_model,
        reranker_model=reranker_model
    )

    # Create pipeline config
    pipeline_config = PipelineConfig(
        num_text_text_evidence=num_text_text_evidence,
        num_image_text_evidence=num_image_text_evidence,
        num_image_image_evidence_image=num_image_image_evidence_image,
        num_image_image_evidence_text=num_image_image_evidence_text,
        max_qa_pairs=max_qa_pairs,
        qa_per_iteration=qa_per_iteration,
        max_qa_iterations=max_qa_iterations,
        num_qa_to_select=num_qa_to_select,
        train_data_path=train_data_path,
        use_reranker=use_reranker,
        reranker_fetch_k=reranker_fetch_k,
        agent_config=agent_config
    )

    print("=" * 50)
    print("Multi-Agent Fact-Checking Pipeline")
    print("=" * 50)
    if args.config:
        print(f"\nConfig file: {args.config}")
    print(f"\nConfiguration:")
    print(f"  Agent 1 (Text-Text): {pipeline_config.num_text_text_evidence} evidence")
    print(f"  Agent 2 (Image-Text): {pipeline_config.num_image_text_evidence} evidence")
    print(f"  Agent 3 (Image-Image): {pipeline_config.num_image_image_evidence_image}/img + {pipeline_config.num_image_image_evidence_text}/text")
    print(f"  Agent 4 (QA Generation): {pipeline_config.qa_per_iteration} pairs × {pipeline_config.max_qa_iterations} iterations (max {pipeline_config.max_qa_pairs})")
    print(f"  Agent 5 (Verdict): select {pipeline_config.num_qa_to_select} Q-A pairs")
    print(f"\nRetrieval:")
    print(f"  Reranker: {'Enabled' if use_reranker else 'Disabled'}" + (f" (fetch_k={reranker_fetch_k})" if use_reranker else ""))
    print(f"\nProcessing:")
    print(f"  Save intermediate outputs: {'Enabled' if save_intermediate_output else 'Disabled'}")
    print(f"\nModels:")
    print(f"  Text Model: {agent_config.text_model} ({agent_config.text_model_type})")
    print(f"  Image Model: {agent_config.image_model}")
    print(f"  VLM Model: {agent_config.vlm_model}")
    if use_reranker:
        print(f"  Reranker: {agent_config.reranker_model}")
    print(f"  Device: {agent_config.device}")

    # Load dataset
    print(f"\nLoading data from {data_path}")
    with open(data_path, 'r') as f:
        all_samples = json.load(f)

    total_samples = len(all_samples)

    # Apply start_idx and end_idx for parallel processing
    proc_start_idx = start_idx if start_idx is not None else 0
    proc_end_idx = end_idx if end_idx is not None else total_samples

    # Clamp indices
    proc_start_idx = max(0, min(proc_start_idx, total_samples))
    proc_end_idx = max(proc_start_idx, min(proc_end_idx, total_samples))

    samples = all_samples[proc_start_idx:proc_end_idx]

    # Apply max_samples limit (on top of index range)
    if max_samples:
        samples = samples[:max_samples]

    print(f"Total samples in dataset: {total_samples}")
    print(f"Processing samples [{proc_start_idx}:{proc_end_idx}] = {len(samples)} samples")

    # Create pipeline
    pipeline = MultiAgentPipeline(pipeline_config)

    # Optionally preload all models
    if preload_models:
        pipeline.preload_models()

    # Process claims one by one and save per-claim outputs
    from tqdm import tqdm

    results = []
    skipped = 0
    for idx, sample in enumerate(tqdm(samples, desc="Processing claims")):
        claim_id = sample.get('claim_id', sample.get('id', idx + proc_start_idx))

        # Skip if already processed
        submission_path = os.path.join(output_dir, 'submission', str(claim_id), 'submission.json')
        if os.path.exists(submission_path):
            skipped += 1
            continue

        claim_text = sample['claim_text']
        claim_images = sample.get('claim_images', [])
        label = sample.get('label', '')
        speaker = sample['metadata'].get('speaker', 'Unknown')
        date = sample.get('date', 'Not Specified')

        result = pipeline.process_claim(
            claim_id=claim_id,
            claim_text=claim_text,
            claim_images=claim_images,
            label=label,
            speaker=speaker,
            date=date
        )
        results.append(result)

        # Save per-claim outputs immediately
        save_per_claim_outputs(result, output_dir, save_intermediate=save_intermediate_output)

    if skipped > 0:
        print(f"\nSkipped {skipped} already processed claims")

    # Print statistics
    total_questions = sum(len(r.questions) for r in results)
    total_text_evidence = sum(
        len(r.text_text_analysis.evidence_items) if r.text_text_analysis else 0
        for r in results
    )
    total_image_text_evidence = sum(
        len(r.image_text_analysis.evidence_items) if r.image_text_analysis else 0
        for r in results
    )
    total_image_evidence = sum(
        len(r.image_image_analysis.evidence_items) if r.image_image_analysis else 0
        for r in results
    )

    # Count veracity predictions
    veracity_counts = {}
    for r in results:
        verdict = r.veracity_verdict or "Unknown"
        veracity_counts[verdict] = veracity_counts.get(verdict, 0) + 1

    print(f"\n{'=' * 50}")
    print("Multi-Agent Pipeline Complete!")
    print(f"{'=' * 50}")
    print(f"Processed samples [{proc_start_idx}:{proc_end_idx}]")
    print(f"Total claims processed: {len(results)}")
    print(f"Total questions generated: {total_questions}")
    if len(results) > 0:
        print(f"Average questions per claim: {total_questions / len(results):.2f}")
    print(f"\nEvidence statistics:")
    print(f"  Text-Text evidence: {total_text_evidence}")
    print(f"  Image-Text evidence: {total_image_text_evidence}")
    print(f"  Image-Image evidence: {total_image_evidence}")
    print(f"\nVeracity predictions:")
    for verdict, count in sorted(veracity_counts.items()):
        print(f"  {verdict}: {count}")
    print(f"\nOutputs saved to {output_dir}/submission/")
    print(f"\nTo merge outputs, run:")
    print(f"  python src/utils/merge_output.py --input_dir {output_dir} --output_dir {output_dir}")


if __name__ == '__main__':
    main()

