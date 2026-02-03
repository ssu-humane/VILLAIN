#!/bin/bash
# Merge per-sample local evaluation results into single eval_results_local.json
# Usage: ./scripts/merge_eval_results_local.sh [config_file]

set -e

# Default config file
CONFIG_FILE="${1:-scripts/cfg/default.yaml}"

echo "=========================================="
echo "Merge Local Evaluation Results"
echo "=========================================="
echo "Config: $CONFIG_FILE"
echo ""

# Parse YAML config using Python
read_yaml() {
    python3 -c "
import yaml
with open('$CONFIG_FILE', 'r') as f:
    cfg = yaml.safe_load(f)
def get_nested(d, *keys, default=''):
    for key in keys:
        if isinstance(d, dict):
            d = d.get(key, default)
        else:
            return default
    return d if d is not None else default
print(get_nested(cfg, $1))
"
}

# Read output directory from config
OUTPUT_DIR=$(read_yaml "'data', 'output_dir'")

echo "Output dir: $OUTPUT_DIR"
echo ""

# Run merge script
python src/utils/merge_eval_results.py \
    --input_dir "$OUTPUT_DIR" \
    --output_dir "$OUTPUT_DIR"

echo ""
echo "Done! Results saved to: $OUTPUT_DIR/eval_results_local.json"

