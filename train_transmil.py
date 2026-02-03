#!/usr/bin/env python3
"""
TransMIL Training Script - Optimized for DGX Spark

Features:
- Mixed precision training (AMP)
- Gradient accumulation
- Early stopping
- Checkpoint saving
- Progress logging
- Stratified train/val split
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score

# Add parent dir to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "models"))
from transmil import TransMIL


def load_embeddings(embeddings_dir, slide_ids):
    """Load embeddings for given slide IDs."""
    embeddings_dir = Path(embeddings_dir)
    embeddings = {}
    
    for sid in slide_ids:
        emb_path = embeddings_dir / f"{sid}.npy"
        if emb_path.exists():
            embeddings[sid] = np.load(emb_path)
    
    return embeddings


def train_epoch(model, train_data, labels, device, optimizer, scaler, accumulation_steps=4):
    """Train one epoch with gradient accumulation and AMP."""
    model.train()
    total_loss = 0
    optimizer.zero_grad()
    
    slide_ids = list(train_data.keys())
    np.random.shuffle(slide_ids)
    
    for i, sid in enumerate(slide_ids):
        emb = torch.tensor(train_data[sid], dtype=torch.float32).to(device)
        label = torch.tensor([labels[sid]], dtype=torch.float32).to(device)
        
        with autocast():
            pred = model(emb)
        # BCE outside autocast for safety
        loss = F.binary_cross_entropy(pred.float().view(-1), label.float()) / accumulation_steps
        
        scaler.scale(loss).backward()
        total_loss += loss.item() * accumulation_steps
        
        if (i + 1) % accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
    
    # Handle remaining gradients
    if len(slide_ids) % accumulation_steps != 0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
    
    return total_loss / len(slide_ids)


@torch.no_grad()
def evaluate(model, val_data, labels, device):
    """Evaluate model on validation set."""
    model.eval()
    preds = []
    true_labels = []
    
    for sid, emb in val_data.items():
        emb = torch.tensor(emb, dtype=torch.float32).to(device)
        
        with autocast():
            pred = model(emb)
        
        preds.append(pred.float().item())
        true_labels.append(labels[sid])
    
    preds = np.array(preds)
    true_labels = np.array(true_labels)
    
    try:
        auc = roc_auc_score(true_labels, preds)
    except:
        auc = 0.5
    
    acc = accuracy_score(true_labels, (preds > 0.5).astype(int))
    
    return auc, acc, preds, true_labels


def main():
    parser = argparse.ArgumentParser(description="Train TransMIL model")
    parser.add_argument("--embeddings_dir", type=str, required=True,
                        help="Directory containing .npy embeddings")
    parser.add_argument("--labels_file", type=str, required=True,
                        help="JSON file with slide_id -> label mapping")
    parser.add_argument("--output_dir", type=str, default="./outputs/transmil",
                        help="Output directory for checkpoints")
    parser.add_argument("--epochs", type=int, default=100,
                        help="Maximum training epochs")
    parser.add_argument("--lr", type=float, default=2e-4,
                        help="Learning rate")
    parser.add_argument("--hidden_dim", type=int, default=512,
                        help="Transformer hidden dimension")
    parser.add_argument("--num_heads", type=int, default=8,
                        help="Number of attention heads")
    parser.add_argument("--num_layers", type=int, default=2,
                        help="Number of transformer layers")
    parser.add_argument("--dropout", type=float, default=0.1,
                        help="Dropout rate")
    parser.add_argument("--patience", type=int, default=15,
                        help="Early stopping patience")
    parser.add_argument("--accumulation_steps", type=int, default=4,
                        help="Gradient accumulation steps")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    args = parser.parse_args()
    
    # Set seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load labels
    print(f"\nLoading labels from {args.labels_file}")
    with open(args.labels_file) as f:
        all_labels = json.load(f)
    
    # Find available embeddings
    embeddings_dir = Path(args.embeddings_dir)
    available_slides = []
    available_labels = []
    
    for sid, label in all_labels.items():
        if (embeddings_dir / f"{sid}.npy").exists():
            available_slides.append(sid)
            available_labels.append(label)
    
    print(f"Found {len(available_slides)} slides with embeddings")
    pos_count = sum(available_labels)
    neg_count = len(available_labels) - pos_count
    print(f"Label distribution: {pos_count} positive ({100*pos_count/len(available_labels):.1f}%), "
          f"{neg_count} negative ({100*neg_count/len(available_labels):.1f}%)")
    
    # Train/val split
    train_slides, val_slides, train_labels_list, val_labels_list = train_test_split(
        available_slides, available_labels,
        test_size=0.2, random_state=args.seed, stratify=available_labels
    )
    
    print(f"\nTrain: {len(train_slides)} slides")
    print(f"Val: {len(val_slides)} slides")
    
    # Load embeddings
    print("\nLoading embeddings...")
    start = time.time()
    train_data = load_embeddings(embeddings_dir, train_slides)
    val_data = load_embeddings(embeddings_dir, val_slides)
    print(f"Loaded in {time.time() - start:.1f}s")
    
    # Create label dicts
    labels_dict = dict(zip(available_slides, available_labels))
    
    # Check embedding dimensions
    sample_emb = next(iter(train_data.values()))
    input_dim = sample_emb.shape[1]
    print(f"Input dimension: {input_dim}")
    
    # Initialize model
    model = TransMIL(
        input_dim=input_dim,
        hidden_dim=args.hidden_dim,
        num_classes=1,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        dropout=args.dropout
    ).to(device)
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")
    
    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=1e-5
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=1e-6
    )
    scaler = GradScaler()
    
    # Training loop
    best_auc = 0
    patience_counter = 0
    history = {"train_loss": [], "val_auc": [], "val_acc": []}
    
    print(f"\n{'='*60}")
    print("Starting training...")
    print(f"{'='*60}\n")
    
    for epoch in range(args.epochs):
        epoch_start = time.time()
        
        # Train
        train_loss = train_epoch(
            model, train_data, labels_dict, device, 
            optimizer, scaler, args.accumulation_steps
        )
        
        # Evaluate
        val_auc, val_acc, val_preds, val_true = evaluate(
            model, val_data, labels_dict, device
        )
        
        # Update scheduler
        scheduler.step()
        
        # Record history
        history["train_loss"].append(train_loss)
        history["val_auc"].append(val_auc)
        history["val_acc"].append(val_acc)
        
        epoch_time = time.time() - epoch_start
        
        print(f"Epoch {epoch+1:3d}/{args.epochs} | "
              f"Loss: {train_loss:.4f} | "
              f"Val AUC: {val_auc:.4f} | "
              f"Val Acc: {val_acc:.4f} | "
              f"LR: {scheduler.get_last_lr()[0]:.2e} | "
              f"Time: {epoch_time:.1f}s")
        
        # Save best model
        if val_auc > best_auc:
            best_auc = val_auc
            patience_counter = 0
            
            # Save checkpoint
            torch.save(model.state_dict(), output_dir / "best_model.pt")
            
            # Save config
            config = {
                "input_dim": input_dim,
                "hidden_dim": args.hidden_dim,
                "num_classes": 1,
                "num_heads": args.num_heads,
                "num_layers": args.num_layers,
                "dropout": args.dropout,
                "best_auc": best_auc,
                "best_epoch": epoch + 1
            }
            with open(output_dir / "config.json", "w") as f:
                json.dump(config, f, indent=2)
            
            print(f"  -> New best! Saved to {output_dir / 'best_model.pt'}")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"\nEarly stopping at epoch {epoch+1}")
                break
    
    # Final evaluation
    print(f"\n{'='*60}")
    print("Training complete!")
    print(f"{'='*60}")
    print(f"Best validation AUC: {best_auc:.4f}")
    
    # Save final results
    results = {
        "best_auc": best_auc,
        "final_epoch": epoch + 1,
        "train_slides": len(train_slides),
        "val_slides": len(val_slides),
        "history": history,
        "timestamp": datetime.now().isoformat()
    }
    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    main()
