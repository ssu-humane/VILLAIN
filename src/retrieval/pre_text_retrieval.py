"""
Pre-calculate text embeddings for the knowledge store.
Supports Qwen3-Embedding, MixedBread mxbai-embed-large-v1, and Nomic nomic-embed-text-v2-moe models.
This script pre-builds document embeddings to avoid re-calculating them during retrieval.
"""

import json
import os
import sys
import argparse
import uuid
import pickle
import torch
from tqdm import tqdm
import numpy as np

# Add src directory to path for models import
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models import (
    Qwen3Embedding, QwenEmbeddingConfig,
    MxbaiEmbedding, MxbaiEmbeddingConfig,
    NomicEmbedding, NomicEmbeddingConfig
)


def parse_args():
    parser = argparse.ArgumentParser(description="Pre-calculate text embeddings for knowledge store")
    parser.add_argument('--input_dir', type=str,
                        default='dataset/AVerImaTeC_Shared_Task/Knowledge_Store/val/text_related',
                        help='Path to input knowledge store directory (contains text_related_store_text_val and image_related_store_text_val)')
    parser.add_argument('--output_dir', type=str,
                        default='dataset/AVerImaTeC_Shared_Task/Vector_Store/val/text_related',
                        help='Path to output embedding directory')
    parser.add_argument('--model_name', type=str, default='Qwen/Qwen3-Embedding-0.6B',
                        help='Embedding model name (Qwen, mxbai, or nomic)')
    parser.add_argument('--model_type', type=str, default='qwen', choices=['qwen', 'mxbai', 'nomic'],
                        help='Embedding model type: qwen, mxbai, or nomic')
    parser.add_argument('--model_suffix', type=str, default='8B',
                        help='Model size suffix for output folder name (e.g., 0.6B, 4B, 8B, mxbai)')
    parser.add_argument('--max_length', type=int, default=20000,
                        help='Maximum token length (Qwen3 supports up to 32k, mxbai up to 512)')
    parser.add_argument('--chunk_size', type=int, default=2048,
                        help='Chunk size in characters (default: 2048 = 512 tokens / 0.25 tokens per char)')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size for embedding')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='Device to run model on')
    parser.add_argument('--use_flash_attention', action='store_true',
                        help='Use flash attention for better performance (Qwen only)')
    parser.add_argument('--start_idx', type=int, default=0,
                        help='Start claim index (inclusive)')
    parser.add_argument('--end_idx', type=int, default=152,
                        help='End claim index (exclusive)')
    return parser.parse_args()


def load_model(model_name, model_type, device, max_length):
    """Load embedding model based on type."""
    print(f"Loading {model_type} model: {model_name}")

    if model_type == 'qwen':
        config = QwenEmbeddingConfig(
            model_name=model_name,
            device=device,
            max_length=max_length
        )
        return Qwen3Embedding(config)
    elif model_type == 'mxbai':
        config = MxbaiEmbeddingConfig(
            model_name=model_name,
            device=device,
            max_length=min(max_length, 512)  # mxbai max is 512
        )
        return MxbaiEmbedding(config)
    elif model_type == 'nomic':
        config = NomicEmbeddingConfig(
            model_name=model_name,
            device=device,
            max_length=min(max_length, 8192)  # nomic max is 8192
        )
        return NomicEmbedding(config)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def embed_texts(texts, model, batch_size=8, desc=None):
    """Embed a list of texts using the embedding model in batches."""
    if not texts:
        return np.array([])

    # Replace newlines with spaces
    texts = [t.replace("\n", " ") for t in texts]

    all_embeddings = []

    batch_iter = range(0, len(texts), batch_size)
    if desc:
        batch_iter = tqdm(batch_iter, desc=desc, leave=False)

    for i in batch_iter:
        batch_texts = texts[i:i + batch_size]

        # Use encode_documents (no query prompt prefix for documents)
        embeddings = model.encode_documents(batch_texts)
        all_embeddings.append(embeddings)

        # Clear GPU cache periodically
        torch.cuda.empty_cache()

    return np.vstack(all_embeddings) if all_embeddings else np.array([])


def chunk_documents(docstore, chunk_size):
    """
    Chunk documents into smaller pieces with context tracking.
    Similar to build_vecstores.py logic.

    Args:
        docstore: List of document dicts with 'url' and 'url2text'
        chunk_size: Maximum chunk size in characters

    Returns:
        chunks: Dict mapping chunk_id to chunk data
    """
    chunks = {}

    for doc in docstore:
        if not doc.get("url2text"):
            continue

        buffer = ""

        for i, sentence in enumerate(doc["url2text"]):
            # Add sentence to buffer first
            buffer += sentence + " "

            # Check if we should create a new chunk
            is_last_sentence = (i == len(doc["url2text"]) - 1)
            buffer_full = (len(buffer) >= chunk_size)

            if is_last_sentence or buffer_full:
                if buffer.strip():
                    context_before = ""

                    # Link to previous chunk if same URL
                    if chunks:
                        last_key = next(reversed(chunks))
                        last_chunk = chunks[last_key]

                        if last_chunk["metadata"]["url"] == doc.get("url", ""):
                            last_chunk["metadata"]["context_after"] = buffer
                            context_before = last_chunk["page_content"]

                    # Generate unique chunk ID
                    while True:
                        chunk_id = uuid.uuid4().int & ((1 << 63) - 1)
                        if chunk_id not in chunks:
                            break

                    chunks[chunk_id] = {
                        "page_content": buffer.strip(),
                        "metadata": {
                            "url": doc.get("url", ""),
                            "context_before": context_before,
                            "context_after": "",
                        },
                    }

                    buffer = ""

    return chunks


def save_vecstore(embeddings, chunks, pos_to_id, output_dir):
    """Save embeddings and chunks in the same format as build_vecstores.py"""
    os.makedirs(output_dir, exist_ok=True)

    # Save embeddings as numpy array
    embeddings_path = os.path.join(output_dir, "embeddings.npy")
    np.save(embeddings_path, embeddings)

    # Save chunks
    chunks_path = os.path.join(output_dir, "chunks.pkl")
    with open(chunks_path, "wb") as f:
        pickle.dump(chunks, f)

    # Save position to ID mapping
    pos_to_id_path = os.path.join(output_dir, "pos_to_id.pkl")
    with open(pos_to_id_path, "wb") as f:
        pickle.dump(pos_to_id, f)


def process_store(store_name, input_path, output_dir, model, args, claim_range):
    """Process a single store (text_related or image_related) for all claims."""
    processed_count = 0
    skipped_count = 0

    for claim_id in tqdm(claim_range, desc=f"Processing {store_name}"):
        json_path = os.path.join(input_path, f"{claim_id}.json")

        if not os.path.exists(json_path):
            skipped_count += 1
            continue

        # Load documents
        docstore = []
        with open(json_path, 'r') as f:
            for line in f:
                try:
                    doc = json.loads(line.strip())
                    docstore.append(doc)
                except json.JSONDecodeError:
                    continue

        if not docstore:
            skipped_count += 1
            continue

        # Chunk documents
        chunks = chunk_documents(docstore, args.chunk_size)

        if not chunks:
            skipped_count += 1
            continue

        # Get chunk IDs and texts
        orig_ids = list(chunks.keys())
        texts = [chunks[oid]["page_content"] for oid in orig_ids]

        # Embed all chunks in batches
        embeddings = embed_texts(
            texts, model, args.batch_size,
            desc=f"  Claim {claim_id} ({len(texts)} chunks)"
        )
        embeddings = embeddings.astype(np.float32)
        pos_to_id = np.array(orig_ids)

        # Save vecstore for this claim
        claim_output_dir = os.path.join(output_dir, str(claim_id))
        save_vecstore(embeddings, chunks, pos_to_id, claim_output_dir)
        processed_count += 1

        # Clear cache after each claim
        torch.cuda.empty_cache()

    return processed_count, skipped_count


def main():
    args = parse_args()

    # Load model based on type
    model = load_model(args.model_name, args.model_type, args.device, args.max_length)

    print(f"Processing claims {args.start_idx} to {args.end_idx - 1}")
    print(f"Model type: {args.model_type}")
    print(f"Chunk size: {args.chunk_size} characters")
    print(f"Input directory: {args.input_dir}")
    print(f"Model suffix: {args.model_suffix}")

    claim_range = range(args.start_idx, args.end_idx)

    # Check if input_dir points to a specific store (single store mode)
    input_basename = os.path.basename(args.input_dir.rstrip('/'))
    is_single_store = (
        input_basename.startswith("text_related_store_text_") or
        input_basename.startswith("image_related_store_text_")
    )

    if is_single_store:
        # Single store mode: process just the specified directory
        store_name = "text_related" if "text_related_store" in input_basename else "image_related"
        output_dir = os.path.join(args.output_dir, f"{input_basename}_{args.model_suffix}")
        os.makedirs(output_dir, exist_ok=True)

        print(f"\n=== Processing {store_name} store (single mode) ===")
        print(f"  Input: {args.input_dir}")
        print(f"  Output: {output_dir}")

        processed, skipped = process_store(
            store_name, args.input_dir, output_dir,
            model, args, claim_range
        )

        print(f"\n=== Pre-processing complete! ===")
        print(f"{store_name}: Processed={processed}, Skipped={skipped}")
        print(f"Output saved to: {output_dir}")

    else:
        # Multi-store mode: process both text_related and image_related
        # Define sub-directories based on split (train/val/test)
        if "train" in args.input_dir:
            sub_dirs = ["text_related_store_text_train", "image_related_store_text_train"]
        elif "val" in args.input_dir:
            sub_dirs = ["text_related_store_text_val", "image_related_store_text_val"]
        elif "test" in args.input_dir:
            sub_dirs = ["text_related_store_text_test", "image_related_store_text_test"]
        else:
            raise ValueError("Input directory must contain 'train', 'val', or 'test'")

        # Process text_related store
        text_input_path = os.path.join(args.input_dir, sub_dirs[0])
        text_output_dir = os.path.join(args.output_dir, f"{sub_dirs[0]}_{args.model_suffix}")
        os.makedirs(text_output_dir, exist_ok=True)

        print(f"\n=== Processing text_related store ===")
        print(f"  Input: {text_input_path}")
        print(f"  Output: {text_output_dir}")

        text_processed, text_skipped = process_store(
            "text_related", text_input_path, text_output_dir,
            model, args, claim_range
        )

        # Process image_related store
        image_input_path = os.path.join(args.input_dir, sub_dirs[1])
        image_output_dir = os.path.join(args.output_dir, f"{sub_dirs[1]}_{args.model_suffix}")
        os.makedirs(image_output_dir, exist_ok=True)

        print(f"\n=== Processing image_related store ===")
        print(f"  Input: {image_input_path}")
        print(f"  Output: {image_output_dir}")

        image_processed, image_skipped = process_store(
            "image_related", image_input_path, image_output_dir,
            model, args, claim_range
        )

        print(f"\n=== Pre-processing complete! ===")
        print(f"text_related:  Processed={text_processed}, Skipped={text_skipped}")
        print(f"image_related: Processed={image_processed}, Skipped={image_skipped}")
        print(f"Output saved to: {args.output_dir}")


if __name__ == '__main__':
    main()

