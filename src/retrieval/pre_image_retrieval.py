"""
Pre-calculate image embeddings for the knowledge store using Ops-MM-embedding-v1-2B model.
This script pre-builds image embeddings to avoid re-calculating them during retrieval.
"""

import os
import argparse
import pickle
import torch
import numpy as np
from tqdm import tqdm
from PIL import Image
from glob import glob


def parse_args():
    parser = argparse.ArgumentParser(description="Pre-calculate image embeddings for knowledge store")
    parser.add_argument('--input_dir', type=str,
                        default='dataset/AVerImaTeC_Shared_Task/Knowledge_Store/val/image_related/image_related_store_image_val',
                        help='Path to input image knowledge store directory')
    parser.add_argument('--output_dir', type=str,
                        default='dataset/AVerImaTeC_Shared_Task/Vector_Store/val/image_related',
                        help='Path to output embedding directory')
    parser.add_argument('--model_name', type=str, default='OpenSearch-AI/Ops-MM-embedding-v1-2B',
                        help='Image embedding model name')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size for embedding')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='Device to run model on')
    parser.add_argument('--use_flash_attention', action='store_true',
                        help='Use flash attention for better performance')
    parser.add_argument('--start_idx', type=int, default=0,
                        help='Start claim index (inclusive)')
    parser.add_argument('--end_idx', type=int, default=152,
                        help='End claim index (exclusive)')
    return parser.parse_args()


def load_model(model_name, device, use_flash_attention=False):
    """Load Ops-MM-embedding model."""
    from src.models.ops_mm_embedding_v1 import OpsMMEmbeddingV1

    print(f"Loading model: {model_name}")

    if use_flash_attention:
        model = OpsMMEmbeddingV1(
            model_name,
            device=device,
            attn_implementation="flash_attention_2"
        )
    else:
        model = OpsMMEmbeddingV1(
            model_name,
            device=device
        )

    return model


def load_images_from_dir(image_dir):
    """
    Load all images from a directory.

    Returns:
        images: List of PIL Image objects
        image_paths: List of image file paths
        image_ids: List of image IDs (filenames without extension)
    """
    images = []
    image_paths = []
    image_ids = []

    # Get all image files (jpg, jpeg, png)
    image_files = sorted(
        glob(os.path.join(image_dir, "*.jpg")) +
        glob(os.path.join(image_dir, "*.jpeg")) +
        glob(os.path.join(image_dir, "*.png"))
    )

    for img_path in image_files:
        try:
            img = Image.open(img_path).convert('RGB')
            images.append(img)
            image_paths.append(img_path)
            # Extract image ID from filename (e.g., "0.jpg" -> "0")
            img_id = os.path.splitext(os.path.basename(img_path))[0]
            image_ids.append(img_id)
        except Exception as e:
            print(f"Warning: Failed to load image {img_path}: {e}")
            continue

    return images, image_paths, image_ids


def embed_images_batch(images, model, batch_size=8, desc=None):
    """
    Embed a list of images using the Ops-MM model in batches.

    Args:
        images: List of PIL Image objects
        model: OpsMMEmbeddingV1 model
        batch_size: Batch size for embedding
        desc: Description for progress bar

    Returns:
        embeddings: numpy array of shape (N, embedding_dim)
    """
    if not images:
        return np.array([])

    all_embeddings = []

    batch_iter = range(0, len(images), batch_size)
    if desc:
        batch_iter = tqdm(batch_iter, desc=desc, leave=False)

    for i in batch_iter:
        batch_images = images[i:i + batch_size]

        # Get image embeddings
        embeddings = model.get_image_embeddings(batch_images)

        # Convert to numpy if tensor (convert bfloat16 to float32 first)
        if isinstance(embeddings, torch.Tensor):
            embeddings = embeddings.float().cpu().numpy()

        all_embeddings.append(embeddings)

        # Clear GPU cache periodically
        torch.cuda.empty_cache()

    return np.vstack(all_embeddings) if all_embeddings else np.array([])


def save_image_vecstore(embeddings, image_paths, image_ids, output_dir):
    """
    Save image embeddings and metadata.

    Saves:
        - image_embeddings.npy: Image embeddings array
        - image_paths.pkl: List of image file paths
        - image_ids.pkl: List of image IDs (for mapping)
    """
    os.makedirs(output_dir, exist_ok=True)

    # Save embeddings as numpy array
    embeddings_path = os.path.join(output_dir, "image_embeddings.npy")
    np.save(embeddings_path, embeddings)

    # Save image paths
    paths_path = os.path.join(output_dir, "image_paths.pkl")
    with open(paths_path, "wb") as f:
        pickle.dump(image_paths, f)

    # Save image IDs
    ids_path = os.path.join(output_dir, "image_ids.pkl")
    with open(ids_path, "wb") as f:
        pickle.dump(image_ids, f)


def main():
    args = parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load model
    model = load_model(args.model_name, args.device, args.use_flash_attention)

    print(f"Processing claims {args.start_idx} to {args.end_idx - 1}")
    print(f"Input directory: {args.input_dir}")
    print(f"Output directory: {args.output_dir}")

    processed_count = 0
    skipped_count = 0
    total_images = 0

    for claim_id in tqdm(range(args.start_idx, args.end_idx), desc="Processing claims"):
        # Path to claim's image directory
        claim_image_dir = os.path.join(args.input_dir, str(claim_id))

        if not os.path.exists(claim_image_dir):
            skipped_count += 1
            continue

        # Load images from this claim's directory
        images, image_paths, image_ids = load_images_from_dir(claim_image_dir)

        if not images:
            skipped_count += 1
            continue

        # Embed images in batches
        embeddings = embed_images_batch(
            images, model, args.batch_size,
            desc=f"  Claim {claim_id} ({len(images)} images)"
        )
        embeddings = embeddings.astype(np.float32)

        # Save vecstore for this claim
        claim_output_dir = os.path.join(args.output_dir, str(claim_id))
        save_image_vecstore(embeddings, image_paths, image_ids, claim_output_dir)

        processed_count += 1
        total_images += len(images)

        # Clear cache after each claim
        torch.cuda.empty_cache()

    print(f"\nPre-processing complete!")
    print(f"Processed: {processed_count} claims, {total_images} total images")
    print(f"Skipped (not found): {skipped_count}")
    print(f"Output saved to: {args.output_dir}")


if __name__ == '__main__':
    main()

