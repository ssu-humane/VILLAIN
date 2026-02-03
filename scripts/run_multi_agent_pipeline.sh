#!/bin/bash
# Simple shell script to run the Multi-Agent Fact-Checking Pipeline
# Usage: ./scripts/run_multi_agent_pipeline.sh [config_file] [start_idx] [end_idx]

set -e

# Default config file
CONFIG_FILE="${1:-scripts/cfg/default.yaml}"

# Optional start/end indices for processing a subset
START_IDX="${2:-}"
END_IDX="${3:-}"

echo "=========================================="
echo "Multi-Agent Fact-Checking Pipeline"
echo "=========================================="
echo "Config: $CONFIG_FILE"

# Build command
CMD="python src/run_multi_agent.py --config $CONFIG_FILE"

# Add optional index arguments
if [ -n "$START_IDX" ]; then
    CMD="$CMD --start_idx $START_IDX"
    echo "Start index: $START_IDX"
fi

if [ -n "$END_IDX" ]; then
    CMD="$CMD --end_idx $END_IDX"
    echo "End index: $END_IDX"
fi

echo ""
echo "Running: $CMD"
echo ""

eval $CMD

