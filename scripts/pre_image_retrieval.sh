#!/bin/bash

# Configuration
SPLIT="val"           # Options: train, val, test
MODEL_NAME="OpenSearch-AI/Ops-MM-embedding-v1-7B"
MODEL_SUFFIX="7B"      # Model size suffix for output folder

# Calculate samples per job (ceil division)
START_IDX=0
END_IDX=152

echo "Job $SLURM_ARRAY_TASK_ID: Processing claims $START_IDX to $END_IDX"
echo "Split: $SPLIT, Model: $MODEL_NAME, Model suffix: $MODEL_SUFFIX"

INPUT_BASE="dataset/AVerImaTeC_Shared_Task/Knowledge_Store/${SPLIT}/image_related"
OUTPUT_BASE="dataset/AVerImaTeC_Shared_Task/Vector_Store/${SPLIT}/image_related"

echo ""
echo "=== Processing image_related_store_image_${SPLIT} ==="
python src/retrieval/pre_image_retrieval.py \
    --input_dir ${INPUT_BASE}/image_related_store_image_${SPLIT} \
    --output_dir ${OUTPUT_BASE}/image_related_${MODEL_SUFFIX} \
    --model_name $MODEL_NAME \
    --start_idx $START_IDX \
    --end_idx $END_IDX

echo ""
echo "=== Image embedding complete ==="
