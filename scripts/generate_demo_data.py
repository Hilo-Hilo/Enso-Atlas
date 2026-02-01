#!/usr/bin/env python3
"""
Generate demo data for Enso Atlas.

Creates synthetic embeddings and coordinates that mimic real
Path Foundation output for testing the full pipeline.

This script works on DGX Spark (ARM64 + CUDA).

Usage:
    python scripts/generate_demo_data.py --output data/demo
    python scripts/generate_demo_data.py --num-slides 20 --patches-per-slide 500
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
from pathlib import Path

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def generate_slide_embeddings(
    num_patches: int,
    embedding_dim: int = 384,
    label: int = 0,
    seed: int = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic embeddings for a slide.
    
    Creates embeddings that have some structure based on the label
    to make the MIL classifier work meaningfully.
    
    Args:
        num_patches: Number of patches to generate
        embedding_dim: Dimension of embeddings (384 for Path Foundation)
        label: 0 for non-responder, 1 for responder
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (embeddings, coordinates)
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Base embeddings with some structure
    embeddings = np.random.randn(num_patches, embedding_dim).astype(np.float32)
    
    # Add label-specific patterns to make classification meaningful
    # Responders have higher values in certain dimensions
    if label == 1:
        # Add positive bias to first 50 dimensions
        embeddings[:, :50] += np.random.uniform(0.2, 0.5, 50)
        # Add some highly activating patches
        high_attention_patches = np.random.choice(num_patches, size=num_patches // 10)
        embeddings[high_attention_patches, :100] += 0.8
    else:
        # Non-responders have different pattern
        embeddings[:, 50:100] += np.random.uniform(0.1, 0.3, 50)
        # Fewer highly activating patches
        high_attention_patches = np.random.choice(num_patches, size=num_patches // 20)
        embeddings[high_attention_patches, 100:150] += 0.5
    
    # Normalize embeddings (Path Foundation outputs are L2-normalized)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / (norms + 1e-8)
    
    # Convert to FP16 for storage efficiency
    embeddings = embeddings.astype(np.float16)
    
    # Generate coordinates (simulate 20x magnification on 50000x50000 slide)
    slide_size = 50000
    patch_size = 224
    step = 256  # Overlap between patches
    
    # Grid-based sampling with some randomness
    grid_size = int(np.sqrt(num_patches))
    coords = []
    
    for i in range(num_patches):
        x = np.random.randint(0, slide_size - patch_size)
        y = np.random.randint(0, slide_size - patch_size)
        coords.append([x, y])
    
    coords = np.array(coords, dtype=np.int32)
    
    return embeddings, coords


def train_demo_classifier(
    embeddings_dir: Path,
    output_path: Path,
    labels: dict,
) -> None:
    """
    Train a simple CLAM classifier on the demo data.
    
    Args:
        embeddings_dir: Directory containing embedding files
        output_path: Path to save the trained model
        labels: Dict mapping slide_id to label (0 or 1)
    """
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
    
    from enso_atlas.config import MILConfig
    from enso_atlas.mil.clam import CLAMClassifier
    
    logger.info("Training demo CLAM classifier...")
    
    # Load all embeddings
    embeddings_list = []
    label_list = []
    
    for slide_id, label in labels.items():
        emb_path = embeddings_dir / f"{slide_id}.npy"
        if emb_path.exists():
            emb = np.load(emb_path).astype(np.float32)
            embeddings_list.append(emb)
            label_list.append(label)
    
    if len(embeddings_list) < 4:
        logger.warning("Not enough slides for training, using random initialization")
        # Create classifier with random weights
        config = MILConfig(input_dim=384, hidden_dim=128)
        classifier = CLAMClassifier(config)
        classifier._setup_device()
        classifier._build_model()
        classifier._model.to(classifier._device)
        classifier.save(output_path)
        return
    
    # Split into train/val
    n_train = int(len(embeddings_list) * 0.8)
    indices = np.random.permutation(len(embeddings_list))
    
    train_embs = [embeddings_list[i] for i in indices[:n_train]]
    train_labels = [label_list[i] for i in indices[:n_train]]
    val_embs = [embeddings_list[i] for i in indices[n_train:]]
    val_labels = [label_list[i] for i in indices[n_train:]]
    
    # Train
    config = MILConfig(
        input_dim=384,
        hidden_dim=128,
        epochs=50,
        patience=10,
    )
    classifier = CLAMClassifier(config)
    
    try:
        classifier.fit(
            train_embs,
            train_labels,
            val_embeddings=val_embs if val_embs else None,
            val_labels=val_labels if val_labels else None,
        )
        classifier.save(output_path)
        logger.info(f"Saved trained model to {output_path}")
    except Exception as e:
        logger.error(f"Training failed: {e}")
        # Save untrained model
        classifier._setup_device()
        classifier._build_model()
        classifier._model.to(classifier._device)
        classifier.save(output_path)


def main():
    parser = argparse.ArgumentParser(
        description="Generate demo data for Enso Atlas"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/demo"),
        help="Output directory for demo data",
    )
    parser.add_argument(
        "--num-slides",
        type=int,
        default=10,
        help="Number of slides to generate",
    )
    parser.add_argument(
        "--patches-per-slide",
        type=int,
        default=500,
        help="Number of patches per slide",
    )
    parser.add_argument(
        "--embedding-dim",
        type=int,
        default=384,
        help="Embedding dimension (384 for Path Foundation)",
    )
    parser.add_argument(
        "--responder-ratio",
        type=float,
        default=0.4,
        help="Ratio of responder slides (0.0 to 1.0)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--train-model",
        action="store_true",
        help="Train a demo CLAM classifier on generated data",
    )
    parser.add_argument(
        "--model-output",
        type=Path,
        default=Path("models/demo_clam.pt"),
        help="Output path for trained model",
    )
    
    args = parser.parse_args()
    
    # Set random seed
    np.random.seed(args.seed)
    
    # Create output directories
    embeddings_dir = args.output / "embeddings"
    embeddings_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Generating {args.num_slides} demo slides...")
    logger.info(f"Output directory: {args.output}")
    
    # Determine labels
    num_responders = int(args.num_slides * args.responder_ratio)
    labels = {}
    
    # Generate slides
    for i in range(args.num_slides):
        slide_id = f"slide_{i:03d}"
        label = 1 if i < num_responders else 0
        labels[slide_id] = label
        
        # Vary patch count slightly
        num_patches = args.patches_per_slide + np.random.randint(-100, 100)
        num_patches = max(100, num_patches)
        
        embeddings, coords = generate_slide_embeddings(
            num_patches=num_patches,
            embedding_dim=args.embedding_dim,
            label=label,
            seed=args.seed + i,
        )
        
        # Save embeddings
        emb_path = embeddings_dir / f"{slide_id}.npy"
        np.save(emb_path, embeddings)
        
        # Save coordinates
        coord_path = embeddings_dir / f"{slide_id}_coords.npy"
        np.save(coord_path, coords)
        
        label_str = "responder" if label == 1 else "non-responder"
        logger.info(f"  {slide_id}: {num_patches} patches, {label_str}")
    
    # Save labels CSV
    labels_path = args.output / "labels.csv"
    with open(labels_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["slide_id", "label", "response"])
        writer.writeheader()
        for slide_id, label in labels.items():
            writer.writerow({
                "slide_id": slide_id,
                "label": label,
                "response": "responder" if label == 1 else "non-responder",
            })
    
    logger.info(f"Saved labels to {labels_path}")
    
    # Train model if requested
    if args.train_model:
        args.model_output.parent.mkdir(parents=True, exist_ok=True)
        train_demo_classifier(embeddings_dir, args.model_output, labels)
    
    logger.info("Demo data generation complete!")
    logger.info(f"  Embeddings: {embeddings_dir}")
    logger.info(f"  Labels: {labels_path}")
    if args.train_model:
        logger.info(f"  Model: {args.model_output}")


if __name__ == "__main__":
    main()
