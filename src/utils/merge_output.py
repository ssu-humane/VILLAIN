#!/usr/bin/env python3
"""
Merge per-claim output files into single submission.json.

This script merges the per-claim outputs from parallel processing runs into
a single consolidated submission.json for evaluation.

Input structure:
    {input_dir}/submission/{claim_id}/submission.json

Output:
    {output_dir}/submission.json

Usage:
    python src/utils/merge_output.py --input_dir outputs-tmp --output_dir outputs-tmp
"""

import argparse
import json
from pathlib import Path


def merge_outputs(input_dir: str, output_dir: str, verbose: bool = True) -> int:
    """Merge per-claim submission.json files into single submission.json.

    Returns:
        Number of claims merged
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    submission_dir = input_path / 'submission'
    if not submission_dir.exists():
        if verbose:
            print(f"Error: {submission_dir} not found")
        return 0

    submission = []
    claim_ids = sorted([int(d.name) for d in submission_dir.iterdir() if d.is_dir()])

    for claim_id in claim_ids:
        sub_file = submission_dir / str(claim_id) / 'submission.json'
        if sub_file.exists():
            with open(sub_file, 'r') as f:
                submission.append(json.load(f))

    with open(output_path / 'submission.json', 'w') as f:
        json.dump(submission, f, indent=2)

    if verbose:
        print(f"Merged {len(submission)} claims -> {output_path / 'submission.json'}")

    return len(submission)


def main():
    parser = argparse.ArgumentParser(description="Merge per-claim outputs into single submission.json")
    parser.add_argument('--input_dir', type=str, required=True,
                        help='Directory containing per-claim outputs')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory (default: same as input_dir)')
    parser.add_argument('--quiet', action='store_true',
                        help='Suppress output messages')

    args = parser.parse_args()

    output_dir = args.output_dir if args.output_dir else args.input_dir

    print(f"Merging outputs from: {args.input_dir}")
    print(f"Output directory: {output_dir}")
    print()

    num_claims = merge_outputs(args.input_dir, output_dir, verbose=not args.quiet)

    print()
    print("=" * 50)
    print("Merge Complete!")
    print("=" * 50)
    print(f"  Total claims: {num_claims}")


if __name__ == '__main__':
    main()

