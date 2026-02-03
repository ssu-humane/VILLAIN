#!/usr/bin/env python3
"""
Merge per-sample evaluation results into single eval_results_local.json.

This script merges the per-sample eval outputs from parallel processing runs into
a single consolidated evaluation result file.

Input structure:
    {input_dir}/eval_results_local/{sample_id}.json

Output:
    {output_dir}/eval_results_local.json

Usage:
    python src/utils/merge_eval_results.py --input_dir outputs-tmp --output_dir outputs-tmp
"""

import argparse
import json
from pathlib import Path
import numpy as np


def merge_eval_results(input_dir: str, output_dir: str, verbose: bool = True) -> int:
    """Merge per-sample eval result files into single eval_results_local.json.

    Returns:
        Number of samples merged
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    eval_results_dir = input_path / 'eval_results_local'
    if not eval_results_dir.exists():
        if verbose:
            print(f"Error: {eval_results_dir} not found")
        return 0

    # Collect all per-sample results
    all_eval_results = []
    sample_files = sorted(eval_results_dir.glob('*.json'), key=lambda x: int(x.stem))

    for sample_file in sample_files:
        with open(sample_file, 'r') as f:
            result = json.load(f)
            all_eval_results.append(result)

    if len(all_eval_results) == 0:
        if verbose:
            print("No evaluation results found to merge")
        return 0

    # Compute aggregate scores (same logic as eval_official_standalone.py)
    EVIDENCE_THRESHOLD = 0.3

    ques_scores = [r['ques_score'] for r in all_eval_results]
    evid_scores = [r['evid_score'] for r in all_eval_results]
    verdict_scores = [r['verdict_score'] for r in all_eval_results]
    justi_scores = [r['justi_score'] for r in all_eval_results]

    evidence_verdict_scores = []
    evidence_justification_scores = []
    for r in all_eval_results:
        if r['evid_score'] > EVIDENCE_THRESHOLD:
            evidence_verdict_scores.append(r['verdict_score'])
            evidence_justification_scores.append(r['justi_score'])
        else:
            evidence_verdict_scores.append(0.0)
            evidence_justification_scores.append(0.0)

    num_above_threshold = sum(1 for r in all_eval_results if r['evid_score'] > EVIDENCE_THRESHOLD)

    # Build merged results
    results = {
        'component_scores': {
            'question_generation': float(np.mean(ques_scores)),
            'evidence_retrieval': float(np.mean(evid_scores)),
            'verdict_prediction': float(np.mean(verdict_scores)),
            'justification': float(np.mean(justi_scores)),
            'evidence_verdict': float(np.mean(evidence_verdict_scores)),
            'evidence_justification': float(np.mean(evidence_justification_scores))
        },
        'num_samples': len(all_eval_results),
        'num_samples_above_evidence_threshold': num_above_threshold,
        'evidence_threshold': EVIDENCE_THRESHOLD,
        'per_sample_results': all_eval_results,
        'note': f'evidence_verdict and evidence_justification scores only count when evidence score > {EVIDENCE_THRESHOLD}'
    }

    output_file = output_path / 'eval_results_local.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    if verbose:
        print(f"Merged {len(all_eval_results)} samples -> {output_file}")
        print(f"\nComponent Scores:")
        print(f"  Question Generation:     {results['component_scores']['question_generation']:.4f}")
        print(f"  Evidence Retrieval:      {results['component_scores']['evidence_retrieval']:.4f}")
        print(f"  Verdict Prediction:      {results['component_scores']['verdict_prediction']:.4f}")
        print(f"  Justification:           {results['component_scores']['justification']:.4f}")
        print(f"  Evidence-Verdict:        {results['component_scores']['evidence_verdict']:.4f}")
        print(f"  Evidence-Justification:  {results['component_scores']['evidence_justification']:.4f}")

    return len(all_eval_results)


def main():
    parser = argparse.ArgumentParser(description="Merge per-sample eval results")
    parser.add_argument('--input_dir', type=str, required=True,
                        help='Directory containing eval_results_local/')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory (default: same as input_dir)')
    parser.add_argument('--quiet', action='store_true',
                        help='Suppress output messages')

    args = parser.parse_args()

    output_dir = args.output_dir if args.output_dir else args.input_dir

    print(f"Merging eval results from: {args.input_dir}/eval_results_local/")
    print(f"Output: {output_dir}/eval_results_local.json")
    print()

    num_samples = merge_eval_results(args.input_dir, output_dir, verbose=not args.quiet)

    print()
    print("=" * 50)
    print("Merge Complete!")
    print("=" * 50)
    print(f"  Total samples: {num_samples}")


if __name__ == '__main__':
    main()

