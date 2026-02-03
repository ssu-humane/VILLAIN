"""
Standalone Veracity Evaluation Script for AVerImaTeC Pipeline
Reuses existing question and evidence scores from official_evaluation.json
Only re-evaluates the veracity score from new predictions
"""

import json
import os
import argparse
import numpy as np


def normalize_label(label):
    """Normalize label to standard format for comparison"""
    label = label.lower().strip()
    
    if 'support' in label or 'true' in label or 'correct' in label:
        return 'supported'
    elif 'refut' in label or 'false' in label or 'incorrect' in label:
        return 'refuted'
    elif 'not enough' in label or 'insufficient' in label:
        return 'not enough evidence'
    elif 'conflict' in label or 'cherry' in label:
        return 'conflicting evidence/cherrypicking'
    else:
        return label.lower()


def parse_args():
    parser = argparse.ArgumentParser(
        description='Evaluate veracity only, reusing question and evidence scores from existing evaluation'
    )
    parser.add_argument('--predictions_path', type=str, required=True,
                        help='Path to new predictions file')
    parser.add_argument('--existing_eval_path', type=str, required=True,
                        help='Path to existing official_evaluation.json with ques_score and evid_score')
    parser.add_argument('--ground_truth_path', type=str, required=True,
                        help='Path to ground truth data')
    parser.add_argument('--output_path', type=str, required=True,
                        help='Path to save new evaluation results')
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Load data
    print("Loading data...")
    
    with open(args.predictions_path, 'r') as f:
        predictions_data = json.load(f)
    
    with open(args.existing_eval_path, 'r') as f:
        existing_eval = json.load(f)
    
    with open(args.ground_truth_path, 'r') as f:
        gt_data_list = json.load(f)
    
    # Convert ground truth to dict for easy lookup
    gt_data = {i: sample for i, sample in enumerate(gt_data_list)}
    
    # Extract predictions list
    if isinstance(predictions_data, dict) and 'predictions' in predictions_data:
        predictions = predictions_data['predictions']
    else:
        predictions = predictions_data
    
    # Get existing per-sample results
    existing_per_sample = existing_eval.get('per_sample_results', [])
    
    if len(existing_per_sample) != len(predictions):
        print(f"Warning: Number of samples mismatch!")
        print(f"  Existing eval: {len(existing_per_sample)} samples")
        print(f"  New predictions: {len(predictions)} samples")
        print("  Proceeding with min of both...")
    
    num_samples = min(len(existing_per_sample), len(predictions))
    
    print(f"\nEvaluating veracity for {num_samples} samples...")
    print("  ✓ Reusing ques_score and evid_score from existing evaluation")
    print("  → Only verdict_score will be recalculated\n")
    
    # Evaluate each sample
    all_eval_results = []
    
    for i in range(num_samples):
        prediction = predictions[i]
        existing_result = existing_per_sample[i]
        sample_id = prediction['claim_id']
        gt_sample = gt_data[sample_id]
        
        # Get predicted and ground truth verdicts
        pred_verdict = prediction.get('predicted_verdict', '')
        gt_verdict = gt_sample.get('label', '')
        
        # Normalize and compare
        pred_normalized = normalize_label(pred_verdict)
        gt_normalized = normalize_label(gt_verdict)
        
        verdict_acc = 1.0 if pred_normalized == gt_normalized else 0.0
        
        # Create new result with existing ques_score and evid_score
        new_result = {
            'ques_score': existing_result['ques_score'],
            'evid_score': existing_result['evid_score'],
            'verdict_score': verdict_acc,
            'justi_score': existing_result.get('justi_score', 0.0),
            'intermediate_info': existing_result.get('intermediate_info', {}),
            'new_pred_verdict': pred_verdict,
            'gt_verdict': gt_verdict
        }
        
        all_eval_results.append(new_result)
        
        # Print progress every 10 samples
        if (i + 1) % 10 == 0 or i == num_samples - 1:
            print(f"  Processed {i+1}/{num_samples} samples...")
    
    # Compute aggregate scores
    ques_scores = [r['ques_score'] for r in all_eval_results]
    evid_scores = [r['evid_score'] for r in all_eval_results]
    verdict_scores = [r['verdict_score'] for r in all_eval_results]
    justi_scores = [r['justi_score'] for r in all_eval_results]
    
    # Compute conditional evidence-verdict scores
    EVIDENCE_THRESHOLD = 0.3
    
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
    
    # Save results
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
        'note': 'Veracity-only evaluation. ques_score and evid_score reused from existing evaluation.'
    }
    
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    with open(args.output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    print("\n" + "="*70)
    print("VERACITY-ONLY EVALUATION RESULTS")
    print("="*70)
    print(f"\nComponent Scores (0-1 scale):")
    print(f"  Question Generation Score:     {results['component_scores']['question_generation']:.4f} (reused)")
    print(f"  Evidence Retrieval Score:      {results['component_scores']['evidence_retrieval']:.4f} (reused)")
    print(f"  Verdict Prediction Score:      {results['component_scores']['verdict_prediction']:.4f} (NEW)")
    print(f"  Justification Score:           {results['component_scores']['justification']:.4f} (reused)")
    print(f"  Evidence-Verdict Score:        {results['component_scores']['evidence_verdict']:.4f} (conditional)")
    print(f"  Evidence-Justification Score:  {results['component_scores']['evidence_justification']:.4f} (conditional)")
    print(f"\nEvaluation Details:")
    print(f"  Samples evaluated:          {len(all_eval_results)}")
    print(f"  Samples with E > {EVIDENCE_THRESHOLD}:      {num_above_threshold} ({num_above_threshold/len(all_eval_results)*100:.1f}%)")
    print(f"\nNote: Only verdict_score was recalculated. Other scores reused from existing evaluation.")
    print(f"\nResults saved to: {args.output_path}")
    print("="*70)


if __name__ == '__main__':
    main()

