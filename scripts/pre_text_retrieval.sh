#!/bin/bash

# Configuration
SPLIT="val"           # Options: train, val, test
MODEL_TYPE="mxbai"     # Options: qwen, mxbai
MODEL_NAME="mixedbread-ai/mxbai-embed-large-v1"  # For qwen: Qwen/Qwen3-Embedding-8B, For mxbai: mixedbread-ai/mxbai-embed-large-v1
MODEL_SUFFIX="0d3B"     # Model size suffix for output folder (e.g., 8B, mxbai)

START_IDX=0
END_IDX=152

echo "Split: $SPLIT, Model type: $MODEL_TYPE, Model suffix: $MODEL_SUFFIX"

INPUT_BASE="dataset/AVerImaTeC_Shared_Task/Knowledge_Store/${SPLIT}/text_related"
OUTPUT_BASE="dataset/AVerImaTeC_Shared_Task/Vector_Store/${SPLIT}/text_related"

# 1. Process text_related_store_text
echo ""
echo "=== [1/4] Processing text_related_store_text_${SPLIT} ==="
python src/retrieval/pre_text_retrieval.py \
    --input_dir ${INPUT_BASE}/text_related_store_text_${SPLIT} \
    --output_dir ${OUTPUT_BASE} \
    --model_name $MODEL_NAME \
    --model_type $MODEL_TYPE \
    --model_suffix $MODEL_SUFFIX \
    --start_idx $START_IDX \
    --end_idx $END_IDX

# 2. Process image_related_store_text
echo ""
echo "=== [2/4] Processing image_related_store_text_${SPLIT} ==="
python src/retrieval/pre_text_retrieval.py \
    --input_dir ${INPUT_BASE}/image_related_store_text_${SPLIT} \
    --output_dir ${OUTPUT_BASE} \
    --model_name $MODEL_NAME \
    --model_type $MODEL_TYPE \
    --model_suffix $MODEL_SUFFIX \
    --start_idx $START_IDX \
    --end_idx $END_IDX

echo ""
echo "=== All 2 stores processed ==="
