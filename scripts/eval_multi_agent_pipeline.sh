#!/bin/bash
# Simple shell script to merge and evaluate Multi-Agent Pipeline outputs
# Usage: ./scripts/eval_multi_agent_pipeline.sh [config_file]

set -e

# Load environment variables from .env file
if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
fi

# Default config file
CONFIG_FILE="${1:-scripts/cfg/default.yaml}"

echo "=========================================="
echo "Merge & Evaluate Multi-Agent Pipeline"
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

# Read configuration from YAML
DATA_PATH=$(read_yaml "'data', 'data_path'")
IMAGE_DIR=$(read_yaml "'data', 'image_dir'")
OUTPUT_DIR=$(read_yaml "'data', 'output_dir'")
DEVICE=$(read_yaml "'models', 'device'")
EVAL_API_MODEL=$(read_yaml "'evaluation', 'api_model'")
EVAL_LOCAL_MODEL=$(read_yaml "'evaluation', 'eval_model'")
EVAL_JUSTIFICATION=$(read_yaml "'evaluation', 'eval_justification'")

echo "Data path: $DATA_PATH"
echo "Output dir: $OUTPUT_DIR"
if [ -n "$EVAL_API_MODEL" ]; then
    echo "Eval model: $EVAL_API_MODEL (API)"
else
    echo "Eval model: $EVAL_LOCAL_MODEL (local)"
fi
echo ""

# ============================================================================
# Step 1: Merge per-claim outputs
# ============================================================================
echo "=========================================="
echo "Step 1: Merging per-claim outputs"
echo "=========================================="

python src/utils/merge_output.py \
    --input_dir "$OUTPUT_DIR" \
    --output_dir "$OUTPUT_DIR"

echo ""
echo "Step 1 Complete: Merged outputs -> $OUTPUT_DIR/submission.json"

# ============================================================================
# Step 2: Official Evaluation
# ============================================================================
echo ""
echo "=========================================="
echo "Step 2: Running Official Evaluation"
echo "=========================================="

# Determine output file name based on model type
if [ -n "$EVAL_API_MODEL" ]; then
    EVAL_OUTPUT="$OUTPUT_DIR/eval_results_api.json"
    EVAL_CMD="python src/evaluation/eval_official_standalone.py \
        --submission_path \"$OUTPUT_DIR/submission.json\" \
        --ground_truth_path \"$DATA_PATH\" \
        --image_dir \"$IMAGE_DIR\" \
        --output_path \"$EVAL_OUTPUT\" \
        --api_model \"$EVAL_API_MODEL\" \
        --gemini_api_key \"$GOOGLE_API_KEY\" \
        --device \"$DEVICE\""
else
    EVAL_OUTPUT="$OUTPUT_DIR/eval_results_local.json"
    EVAL_CMD="python src/evaluation/eval_official_standalone.py \
        --submission_path \"$OUTPUT_DIR/submission.json\" \
        --ground_truth_path \"$DATA_PATH\" \
        --image_dir \"$IMAGE_DIR\" \
        --output_path \"$EVAL_OUTPUT\" \
        --eval_model \"$EVAL_LOCAL_MODEL\" \
        --device \"$DEVICE\""
fi

# Add justification flag if enabled
if [ "$EVAL_JUSTIFICATION" = "True" ] || [ "$EVAL_JUSTIFICATION" = "true" ]; then
    EVAL_CMD="$EVAL_CMD --justification"
fi

eval $EVAL_CMD

echo ""
echo "=========================================="
echo "Merge & Evaluation Complete!"
echo "=========================================="
echo "Output: $EVAL_OUTPUT"

