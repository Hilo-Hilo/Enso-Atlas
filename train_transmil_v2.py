#!/usr/bin/env python3
"""
TransMIL Training Script v2 - Fixed for class imbalance

Fixes:
- Weighted BCE loss for class imbalance
- Proper learning rate warmup
- Better gradient handling
- Debug output for predictions
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
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score

sys.path.insert(0, str(Path(__file__).parent.parent))
from transmil import TransMIL


class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance."""
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, pred, target):
        bce = F.binary_cross_entropy(pred, target, reduction='none')
        pt = torch.exp(-bce)
        focal_weight = self.alpha * (1 - pt) ** self.gamma
        return (focal_weight * bce).mean()


def train_epoch(model, train_data, labels, device, optimizer, pos_weight):
    """Train one epoch with weighted loss."""
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    slide_ids = list(train_data.keys())
    np.random.shuffle(slide_ids)
    
    for sid in slide_ids:
        optimizer.zero_grad()
        
        emb = torch.tensor(train_data[sid], dtype=torch.float32).to(device)
        label = torch.tensor([labels[sid]], dtype=torch.float32).to(device)
        
        pred = model(emb)
        
        # Weighted BCE: weight positive class less since it's majority
        weight = torch.where(label == 1, 1.0 / pos_weight, pos_weight)
        loss = F.binary_cross_entropy(pred.view(-1), label, weight=weight)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item()
        all_preds.append(pred.item())
        all_labels.append(label.item())
    
    # Debug: check prediction distribution
    preds_arr = np.array(all_preds)
    print(f"    Train preds: min={preds_arr.min():.3f}, max={preds_arr.max():.3f}, "
          f"mean={preds_arr.mean():.3f}, std={preds_arr.std():.3f}")
    
    return total_loss / len(slide_ids)


@torch.no_grad()
def evaluate(model, val_data, labels, device):
    """Evaluate model on validation set."""
    model.eval()
    preds = []
    true_labels = []
    
    for sid, emb in val_data.items():
        emb = torch.tensor(emb, dtype=torch.float32).to(device)
        pred = model(emb)
        preds.append(pred.item())
        true_labels.append(labels[sid])
    
    preds = np.array(preds)
    true_labels = np.array(true_labels)
    
    # Debug output
    print(f"    Val preds: min={preds.min():.3f}, max={preds.max():.3f}, "
          f"mean={preds.mean():.3f}, std={preds.std():.3f}")
    
    try:
        auc = roc_auc_score(true_labels, preds)
    except:
        auc = 0.5
    
    acc = accuracy_score(true_labels, (preds > 0.5).astype(int))
    
    return auc, acc, preds, true_labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--embeddings_dir", type=str, required=True)
    parser.add_argument("--labels_file", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./outputs/transmil_v2")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--hidden_dim", type=int, default=512)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load labels
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
    
    print(f"Slides: {len(available_slides)}")
    pos_count = sum(available_labels)
    neg_count = len(available_labels) - pos_count
    print(f"Positive: {pos_count} ({100*pos_count/len(available_labels):.1f}%)")
    print(f"Negative: {neg_count} ({100*neg_count/len(available_labels):.1f}%)")
    
    # Calculate pos_weight for loss (inverse frequency)
    pos_weight = pos_count / neg_count  # ~5.2 for 84%/16% split
    print(f"Pos weight: {pos_weight:.2f}")
    
    # Train/val split (stratified)
    train_slides, val_slides, train_labels_list, val_labels_list = train_test_split(
        available_slides, available_labels,
        test_size=0.2, random_state=args.seed, stratify=available_labels
    )
    
    print(f"Train: {len(train_slides)}, Val: {len(val_slides)}")
    
    # Load embeddings
    print("Loading embeddings...")
    train_data = {}
    val_data = {}
    for sid in train_slides:
        train_data[sid] = np.load(embeddings_dir / f"{sid}.npy")
    for sid in val_slides:
        val_data[sid] = np.load(embeddings_dir / f"{sid}.npy")
    
    labels_dict = dict(zip(available_slides, available_labels))
    
    # Model
    model = TransMIL(
        input_dim=384,
        hidden_dim=args.hidden_dim,
        num_classes=1,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        dropout=args.dropout
    ).to(device)
    
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Optimizer with warmup
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.lr, epochs=args.epochs, 
        steps_per_epoch=len(train_slides), pct_start=0.1
    )
    
    # Training
    best_auc = 0
    patience_counter = 0
    history = {"train_loss": [], "val_auc": [], "val_acc": []}
    
    print(f"\n{'='*60}")
    print("Training TransMIL v2")
    print(f"{'='*60}\n")
    
    for epoch in range(args.epochs):
        t0 = time.time()
        
        # Train
        train_loss = train_epoch(model, train_data, labels_dict, device, optimizer, pos_weight)
        
        # Step scheduler per epoch (simplified)
        for _ in range(len(train_slides)):
            scheduler.step()
        
        # Evaluate
        val_auc, val_acc, _, _ = evaluate(model, val_data, labels_dict, device)
        
        history["train_loss"].append(train_loss)
        history["val_auc"].append(val_auc)
        history["val_acc"].append(val_acc)
        
        print(f"Epoch {epoch+1:3d} | Loss: {train_loss:.4f} | "
              f"AUC: {val_auc:.4f} | Acc: {val_acc:.4f} | "
              f"LR: {optimizer.param_groups[0]['lr']:.2e} | "
              f"Time: {time.time()-t0:.1f}s")
        
        if val_auc > best_auc:
            best_auc = val_auc
            patience_counter = 0
            torch.save(model.state_dict(), output_dir / "best_model.pt")
            
            config = {
                "input_dim": 384,
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
            
            print(f"  -> New best AUC: {best_auc:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"\nEarly stopping at epoch {epoch+1}")
                break
    
    print(f"\n{'='*60}")
    print(f"Best AUC: {best_auc:.4f}")
    print(f"{'='*60}")
    
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


if __name__ == "__main__":
    main()
