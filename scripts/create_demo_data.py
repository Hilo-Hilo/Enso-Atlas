#!/usr/bin/env python3
"""
Create demo data for testing Enso Atlas without real WSIs or models.

Generates:
- Fake embeddings (random 384-dim vectors)
- Fake attention weights
- Sample labels.csv
- A trained dummy MIL model

This allows testing the full UI and pipeline without:
- Downloading 253GB of WSIs
- Downloading Path Foundation / MedGemma models
"""

import numpy as np
from pathlib import Path
import json


def create_demo_embeddings(output_dir: Path, n_slides: int = 10, n_patches_range: tuple = (50, 200)):
    """Create fake embeddings for demo slides."""
    output_dir.mkdir(parents=True, exist_ok=True)
    embeddings_dir = output_dir / "embeddings"
    embeddings_dir.mkdir(exist_ok=True)

    labels = []

    for i in range(n_slides):
        slide_id = f"demo_slide_{i:03d}"
        n_patches = np.random.randint(*n_patches_range)

        # Generate random embeddings (384-dim like Path Foundation)
        embeddings = np.random.randn(n_patches, 384).astype(np.float32)

        # Normalize (Path Foundation outputs are typically normalized)
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

        # Save embeddings
        np.save(embeddings_dir / f"{slide_id}.npy", embeddings)

        # Generate random coordinates
        coords = np.random.randint(0, 50000, size=(n_patches, 2))
        np.save(embeddings_dir / f"{slide_id}_coords.npy", coords)

        # Random label (0 or 1)
        label = np.random.randint(0, 2)
        labels.append({
            "slide_id": slide_id,
            "patient_id": f"patient_{i // 3:02d}",  # 3 slides per patient
            "label": label,
            "label_name": "effective" if label == 1 else "invalid"
        })

        print(f"Created {slide_id}: {n_patches} patches, label={label}")

    # Write labels.csv
    import csv
    labels_path = output_dir / "labels.csv"
    with open(labels_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["slide_id", "patient_id", "label", "label_name"])
        writer.writeheader()
        writer.writerows(labels)

    print(f"\nCreated {n_slides} demo slides in {output_dir}")
    print(f"Labels saved to {labels_path}")
    return labels


def train_demo_model(data_dir: Path, output_path: Path):
    """Train a quick MIL model on demo data."""
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

    from enso_atlas.config import MILConfig
    from enso_atlas.mil.clam import CLAMClassifier

    embeddings_dir = data_dir / "embeddings"
    labels_path = data_dir / "labels.csv"

    # Load data
    import csv
    with open(labels_path) as f:
        reader = csv.DictReader(f)
        labels_data = list(reader)

    embeddings_list = []
    labels = []

    for row in labels_data:
        emb_path = embeddings_dir / f"{row['slide_id']}.npy"
        if emb_path.exists():
            embeddings_list.append(np.load(emb_path))
            labels.append(int(row['label']))

    print(f"Loaded {len(embeddings_list)} slides for training")

    # Train model (quick, few epochs)
    config = MILConfig(
        input_dim=384,
        hidden_dim=128,
        epochs=20,
        patience=5,
        learning_rate=0.001
    )

    classifier = CLAMClassifier(config)

    # Simple train/val split
    split = int(len(embeddings_list) * 0.8)
    history = classifier.fit(
        embeddings_list[:split],
        labels[:split],
        embeddings_list[split:],
        labels[split:]
    )

    # Save model
    output_path.parent.mkdir(parents=True, exist_ok=True)
    classifier.save(output_path)

    print(f"\nModel saved to {output_path}")
    if history.get('val_auc'):
        print(f"Final val AUC: {history['val_auc'][-1]:.3f}")

    return classifier


def create_demo_thumbnail():
    """Create a fake slide thumbnail for demo."""
    from PIL import Image
    import numpy as np

    # Create a simple gradient + noise image
    size = (1024, 1024)
    arr = np.random.randint(180, 255, (*size, 3), dtype=np.uint8)

    # Add some "tissue-like" regions
    for _ in range(5):
        cx, cy = np.random.randint(100, 900, 2)
        r = np.random.randint(100, 300)
        y, x = np.ogrid[:size[0], :size[1]]
        mask = (x - cx)**2 + (y - cy)**2 < r**2
        arr[mask] = np.random.randint(150, 200, 3, dtype=np.uint8)

    return Image.fromarray(arr)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Create demo data for Enso Atlas")
    parser.add_argument("--output-dir", type=Path, default=Path("data/demo"))
    parser.add_argument("--n-slides", type=int, default=10)
    parser.add_argument("--train-model", action="store_true", help="Also train a demo MIL model")

    args = parser.parse_args()

    print("=" * 50)
    print("Creating Enso Atlas Demo Data")
    print("=" * 50)

    # Create demo embeddings
    create_demo_embeddings(args.output_dir, n_slides=args.n_slides)

    # Train model if requested
    if args.train_model:
        print("\n" + "=" * 50)
        print("Training Demo MIL Model")
        print("=" * 50)
        train_demo_model(args.output_dir, Path("models/demo_clam.pt"))

    print("\nDemo data created successfully!")
    print(f"\nTo test the UI with demo data:")
    print(f"  python -m enso_atlas.ui.app")


if __name__ == "__main__":
    main()
