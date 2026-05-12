#!/usr/bin/env python3
"""
TransMIL Training Script v3 - Proper class balancing for platinum sensitivity

Key improvements over v2:
- Focal loss (gamma=2) for class imbalance
- Oversampling of minority class (resistant) during training
- Stratified 5-fold cross-validation
- Optimal threshold selection via Youden's J statistic
- Reports specificity, sensitivity, and balanced accuracy
- Uses slide_labels.json (all 202 slides)
"""

import os
import sys
import json
import time
import copy
import argparse
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score, roc_curve

sys.path.insert(0, str(Path(__file__).parent.parent / "models"))
from transmil import TransMIL


class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance.
    
    With gamma > 0, down-weights easy (well-classified) examples,
    focusing training on hard misclassified examples.
    Alpha balances positive/negative classes.
    """
    def __init__(self, alpha=None, gamma=2.0):
        super().__init__()
        self.alpha = alpha  # per-sample alpha weight
        self.gamma = gamma
    
    def forward(self, pred, target):
        bce = F.binary_cross_entropy(pred, target, reduction='none')
        pt = torch.exp(-bce)  # p_t
        focal_weight = (1 - pt) ** self.gamma
        
        if self.alpha is not None:
            # alpha for positive class, (1-alpha) for negative
            alpha_t = self.alpha * target + (1 - self.alpha) * (1 - target)
            focal_weight = alpha_t * focal_weight
        
        return (focal_weight * bce).mean()


def train_epoch_oversampled(model, train_data, labels, device, optimizer, 
                             loss_fn, oversample_factor=3):
    """Train one epoch with minority class oversampling.
    
    Oversamples resistant (label=0) slides to balance training.
    """
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    # Separate positive and negative slides
    pos_slides = [s for s in train_data if labels[s] == 1]
    neg_slides = [s for s in train_data if labels[s] == 0]
    
    # Oversample minority class
    oversampled_neg = neg_slides * oversample_factor
    
    # Combine and shuffle
    slide_ids = pos_slides + oversampled_neg
    np.random.shuffle(slide_ids)
    
    for sid in slide_ids:
        optimizer.zero_grad()
        
        emb = torch.tensor(train_data[sid], dtype=torch.float32).to(device)
        label = torch.tensor([labels[sid]], dtype=torch.float32).to(device)
        
        pred = model(emb)
        loss = loss_fn(pred.view(-1), label)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item()
        all_preds.append(pred.item())
        all_labels.append(labels[sid])
    
    preds_arr = np.array(all_preds)
    labels_arr = np.array(all_labels)
    
    # Per-class prediction stats
    pos_preds = preds_arr[labels_arr == 1]
    neg_preds = preds_arr[labels_arr == 0]
    print(f"    Train preds - Sensitive: mean={pos_preds.mean():.3f}±{pos_preds.std():.3f} | "
          f"Resistant: mean={neg_preds.mean():.3f}±{neg_preds.std():.3f}")
    
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
    
    try:
        auc = roc_auc_score(true_labels, preds)
    except:
        auc = 0.5
    
    # Find optimal threshold using Youden's J statistic
    try:
        fpr, tpr, thresholds = roc_curve(true_labels, preds)
        j_scores = tpr - fpr
        best_idx = np.argmax(j_scores)
        optimal_threshold = thresholds[best_idx]
        optimal_sensitivity = tpr[best_idx]
        optimal_specificity = 1 - fpr[best_idx]
    except:
        optimal_threshold = 0.5
        optimal_sensitivity = 0.0
        optimal_specificity = 0.0
    
    # Metrics at 0.5 threshold
    pred_labels_05 = (preds > 0.5).astype(int)
    acc_05 = accuracy_score(true_labels, pred_labels_05)
    sens_05 = np.mean(preds[true_labels == 1] > 0.5) if np.sum(true_labels == 1) > 0 else 0
    spec_05 = np.mean(preds[true_labels == 0] <= 0.5) if np.sum(true_labels == 0) > 0 else 0
    
    # Metrics at optimal threshold
    pred_labels_opt = (preds > optimal_threshold).astype(int)
    acc_opt = accuracy_score(true_labels, pred_labels_opt)
    
    # Per-class prediction stats
    pos_preds = preds[true_labels == 1]
    neg_preds = preds[true_labels == 0]
    if len(neg_preds) > 0:
        print(f"    Val preds - Sensitive: mean={pos_preds.mean():.3f}±{pos_preds.std():.3f} | "
              f"Resistant: mean={neg_preds.mean():.3f}±{neg_preds.std():.3f}")
    
    return {
        'auc': auc,
        'accuracy_05': acc_05,
        'sensitivity_05': sens_05,
        'specificity_05': spec_05,
        'optimal_threshold': optimal_threshold,
        'sensitivity_opt': optimal_sensitivity,
        'specificity_opt': optimal_specificity,
        'accuracy_opt': acc_opt,
        'preds': preds,
        'true_labels': true_labels,
    }


def main():
    parser = argparse.ArgumentParser(description="TransMIL v3 Training")
    parser.add_argument("--embeddings_dir", type=str, 
                        default="data/tcga_full/embeddings")
    parser.add_argument("--labels_file", type=str, 
                        default="models/slide_labels.json")
    parser.add_argument("--output_dir", type=str, 
                        default="results/transmil_v3")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--hidden_dim", type=int, default=512)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--n_folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--focal_gamma", type=float, default=2.0)
    parser.add_argument("--oversample_factor", type=int, default=4)
    args = parser.parse_args()
    
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
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
    slide_ids = []
    slide_labels = []
    
    for sid, label in all_labels.items():
        if (embeddings_dir / f"{sid}.npy").exists():
            slide_ids.append(sid)
            slide_labels.append(label)
    
    slide_ids = np.array(slide_ids)
    slide_labels = np.array(slide_labels)
    
    n_pos = int(slide_labels.sum())
    n_neg = len(slide_labels) - n_pos
    print(f"\nTotal slides: {len(slide_ids)}")
    print(f"Sensitive (1): {n_pos} ({100*n_pos/len(slide_ids):.1f}%)")
    print(f"Resistant (0): {n_neg} ({100*n_neg/len(slide_ids):.1f}%)")
    
    # Calculate alpha for focal loss (proportion of negative class)
    alpha = n_neg / len(slide_ids)  # ~0.16 — weight for positive class
    # Actually we want higher weight on the minority class
    # alpha should weight the rare class more
    focal_alpha = 1 - (n_neg / len(slide_ids))  # weight for positive class
    # But we want to focus on minority (negative/resistant), so:
    # Use alpha = proportion of majority class for minority class weighting
    print(f"Focal loss: gamma={args.focal_gamma}, alpha={n_pos/len(slide_ids):.3f}")
    print(f"Oversample factor: {args.oversample_factor}x for minority class")
    
    # Load all embeddings into memory
    print("\nLoading embeddings...")
    all_embeddings = {}
    for sid in slide_ids:
        all_embeddings[sid] = np.load(embeddings_dir / f"{sid}.npy")
    print(f"Loaded {len(all_embeddings)} slide embeddings")
    
    labels_dict = dict(zip(slide_ids, [int(l) for l in slide_labels]))
    
    # Stratified K-Fold CV
    skf = StratifiedKFold(n_splits=args.n_folds, shuffle=True, random_state=args.seed)
    
    fold_results = []
    all_val_preds = np.zeros(len(slide_ids))
    all_val_labels = np.zeros(len(slide_ids))
    best_global_auc = 0
    best_global_model_state = None
    
    print(f"\n{'='*70}")
    print(f"TransMIL v3 — {args.n_folds}-Fold Stratified Cross-Validation")
    print(f"{'='*70}")
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(slide_ids, slide_labels)):
        print(f"\n{'—'*70}")
        print(f"FOLD {fold+1}/{args.n_folds}")
        print(f"{'—'*70}")
        
        train_sids = slide_ids[train_idx]
        val_sids = slide_ids[val_idx]
        
        train_pos = sum(1 for s in train_sids if labels_dict[s] == 1)
        train_neg = len(train_sids) - train_pos
        val_pos = sum(1 for s in val_sids if labels_dict[s] == 1)
        val_neg = len(val_sids) - val_pos
        
        print(f"Train: {len(train_sids)} ({train_pos} sens / {train_neg} res)")
        print(f"Val:   {len(val_sids)} ({val_pos} sens / {val_neg} res)")
        
        train_data = {s: all_embeddings[s] for s in train_sids}
        val_data = {s: all_embeddings[s] for s in val_sids}
        
        # Fresh model per fold
        model = TransMIL(
            input_dim=384,
            hidden_dim=args.hidden_dim,
            num_classes=1,
            num_heads=args.num_heads,
            num_layers=args.num_layers,
            dropout=args.dropout
        ).to(device)
        
        # Focal loss with alpha favoring minority class
        # alpha = weight for class 1 (sensitive/majority). 
        # We want LOWER alpha for majority, HIGHER for minority.
        # In focal loss: alpha_t = alpha * y + (1-alpha) * (1-y)
        # So alpha=0.25 means: class 1 gets weight 0.25, class 0 gets weight 0.75
        minority_alpha = 0.25  # low weight for majority (sens), high for minority (res)
        loss_fn = FocalLoss(alpha=minority_alpha, gamma=args.focal_gamma)
        
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=args.lr, weight_decay=0.01
        )
        
        # Cosine annealing with warm restarts
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=10, T_mult=2, eta_min=1e-6
        )
        
        best_auc = 0
        best_spec = 0
        patience_counter = 0
        best_model_state = None
        best_metrics = None
        
        fold_history = {
            'train_loss': [], 'val_auc': [], 'val_spec_opt': [], 
            'val_sens_opt': [], 'val_acc_opt': [], 'lr': []
        }
        
        for epoch in range(args.epochs):
            t0 = time.time()
            
            train_loss = train_epoch_oversampled(
                model, train_data, labels_dict, device, optimizer,
                loss_fn, oversample_factor=args.oversample_factor
            )
            
            scheduler.step()
            
            metrics = evaluate(model, val_data, labels_dict, device)
            
            fold_history['train_loss'].append(train_loss)
            fold_history['val_auc'].append(metrics['auc'])
            fold_history['val_spec_opt'].append(metrics['specificity_opt'])
            fold_history['val_sens_opt'].append(metrics['sensitivity_opt'])
            fold_history['val_acc_opt'].append(metrics['accuracy_opt'])
            fold_history['lr'].append(optimizer.param_groups[0]['lr'])
            
            # Use combined score: AUC + specificity bonus (we need specificity > 0!)
            combined_score = metrics['auc'] + 0.1 * metrics['specificity_opt']
            
            print(f"  Epoch {epoch+1:3d} | Loss: {train_loss:.4f} | "
                  f"AUC: {metrics['auc']:.4f} | "
                  f"Sens: {metrics['sensitivity_opt']:.3f} | "
                  f"Spec: {metrics['specificity_opt']:.3f} | "
                  f"Thr: {metrics['optimal_threshold']:.3f} | "
                  f"LR: {optimizer.param_groups[0]['lr']:.2e} | "
                  f"{time.time()-t0:.1f}s")
            
            if combined_score > best_auc + 0.1 * best_spec:
                best_auc = metrics['auc']
                best_spec = metrics['specificity_opt']
                patience_counter = 0
                best_model_state = copy.deepcopy(model.state_dict())
                best_metrics = metrics.copy()
                del best_metrics['preds']
                del best_metrics['true_labels']
                best_metrics['best_epoch'] = epoch + 1
                print(f"  -> New best: AUC={best_auc:.4f}, Spec={best_spec:.3f}")
            else:
                patience_counter += 1
                if patience_counter >= args.patience:
                    print(f"  Early stopping at epoch {epoch+1}")
                    break
        
        # Save best model for this fold
        torch.save(best_model_state, output_dir / f"model_fold{fold+1}.pt")
        
        # Evaluate best model on val set to get final predictions
        model.load_state_dict(best_model_state)
        final_metrics = evaluate(model, val_data, labels_dict, device)
        
        # Store predictions for global analysis
        for i, sid in enumerate(val_sids):
            global_idx = np.where(slide_ids == sid)[0][0]
            all_val_preds[global_idx] = final_metrics['preds'][i]
            all_val_labels[global_idx] = final_metrics['true_labels'][i]
        
        fold_result = {
            'fold': fold + 1,
            'train_slides': len(train_sids),
            'val_slides': len(val_sids),
            'auc': best_auc,
            'specificity_opt': best_metrics['specificity_opt'],
            'sensitivity_opt': best_metrics['sensitivity_opt'],
            'optimal_threshold': best_metrics['optimal_threshold'],
            'accuracy_opt': best_metrics['accuracy_opt'],
            'specificity_05': best_metrics['specificity_05'],
            'sensitivity_05': best_metrics['sensitivity_05'],
            'accuracy_05': best_metrics['accuracy_05'],
            'best_epoch': best_metrics['best_epoch'],
        }
        fold_results.append(fold_result)
        
        print(f"\n  Fold {fold+1} Best: AUC={best_auc:.4f} | "
              f"Sens={best_metrics['sensitivity_opt']:.3f} | "
              f"Spec={best_metrics['specificity_opt']:.3f} | "
              f"Thr={best_metrics['optimal_threshold']:.3f}")
        
        # Track best global model
        if best_auc > best_global_auc:
            best_global_auc = best_auc
            best_global_model_state = copy.deepcopy(best_model_state)
    
    # Aggregate results
    print(f"\n{'='*70}")
    print("AGGREGATE RESULTS")
    print(f"{'='*70}")
    
    aucs = [r['auc'] for r in fold_results]
    specs = [r['specificity_opt'] for r in fold_results]
    senss = [r['sensitivity_opt'] for r in fold_results]
    accs = [r['accuracy_opt'] for r in fold_results]
    
    print(f"\nPer-fold AUCs: {[f'{a:.4f}' for a in aucs]}")
    print(f"Mean AUC:      {np.mean(aucs):.4f} ± {np.std(aucs):.4f}")
    print(f"\nPer-fold Specificity: {[f'{s:.3f}' for s in specs]}")
    print(f"Mean Specificity:     {np.mean(specs):.3f} ± {np.std(specs):.3f}")
    print(f"\nPer-fold Sensitivity: {[f'{s:.3f}' for s in senss]}")
    print(f"Mean Sensitivity:     {np.mean(senss):.3f} ± {np.std(senss):.3f}")
    print(f"\nPer-fold Accuracy:    {[f'{a:.3f}' for a in accs]}")
    print(f"Mean Accuracy:        {np.mean(accs):.3f} ± {np.std(accs):.3f}")
    
    # Global AUC from pooled predictions
    try:
        global_auc = roc_auc_score(all_val_labels, all_val_preds)
        fpr, tpr, thresholds = roc_curve(all_val_labels, all_val_preds)
        j_scores = tpr - fpr
        best_idx = np.argmax(j_scores)
        global_threshold = thresholds[best_idx]
        global_sensitivity = tpr[best_idx]
        global_specificity = 1 - fpr[best_idx]
        
        print(f"\nPooled (global) AUC:  {global_auc:.4f}")
        print(f"Global optimal threshold: {global_threshold:.4f}")
        print(f"Global sensitivity:       {global_sensitivity:.3f}")
        print(f"Global specificity:       {global_specificity:.3f}")
    except:
        global_auc = np.mean(aucs)
        global_threshold = 0.5
        global_sensitivity = np.mean(senss)
        global_specificity = np.mean(specs)
    
    # Save best model
    torch.save(best_global_model_state, output_dir / "best_model.pt")
    
    # Save config
    config = {
        "input_dim": 384,
        "hidden_dim": args.hidden_dim,
        "num_classes": 1,
        "num_heads": args.num_heads,
        "num_layers": args.num_layers,
        "dropout": args.dropout,
        "optimal_threshold": float(global_threshold),
    }
    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    # Save results
    results = {
        "timestamp": datetime.now().isoformat(),
        "script": "train_transmil_v3.py",
        "config": {
            "input_dim": 384,
            "hidden_dim": args.hidden_dim,
            "num_heads": args.num_heads,
            "num_layers": args.num_layers,
            "dropout": args.dropout,
            "learning_rate": args.lr,
            "weight_decay": 0.01,
            "epochs": args.epochs,
            "patience": args.patience,
            "n_folds": args.n_folds,
            "seed": args.seed,
            "focal_gamma": args.focal_gamma,
            "focal_alpha": minority_alpha,
            "oversample_factor": args.oversample_factor,
        },
        "device": str(device),
        "n_slides": len(slide_ids),
        "class_distribution": {
            "sensitive": n_pos,
            "resistant": n_neg,
        },
        "aggregate_metrics": {
            "auc": {"mean": float(np.mean(aucs)), "std": float(np.std(aucs)),
                    "per_fold": [float(a) for a in aucs]},
            "specificity_opt": {"mean": float(np.mean(specs)), "std": float(np.std(specs)),
                               "per_fold": [float(s) for s in specs]},
            "sensitivity_opt": {"mean": float(np.mean(senss)), "std": float(np.std(senss)),
                               "per_fold": [float(s) for s in senss]},
            "accuracy_opt": {"mean": float(np.mean(accs)), "std": float(np.std(accs)),
                            "per_fold": [float(a) for a in accs]},
        },
        "global_metrics": {
            "auc": float(global_auc),
            "optimal_threshold": float(global_threshold),
            "sensitivity": float(global_sensitivity),
            "specificity": float(global_specificity),
        },
        "per_fold_metrics": fold_results,
        "comparison_with_v2": {
            "v2_mean_auc": 0.752,
            "v2_specificity": 0.0,
            "v2_n_slides": 152,
            "v3_mean_auc": float(np.mean(aucs)),
            "v3_mean_specificity": float(np.mean(specs)),
            "v3_n_slides": len(slide_ids),
        }
    }
    
    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {output_dir}")
    print(f"Best model saved to {output_dir / 'best_model.pt'}")
    
    # Summary comparison
    print(f"\n{'='*70}")
    print("COMPARISON: v2 → v3")
    print(f"{'='*70}")
    print(f"Slides:      152 → {len(slide_ids)}")
    print(f"Mean AUC:    0.752 → {np.mean(aucs):.4f}")
    print(f"Specificity: 0.000 → {np.mean(specs):.3f}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
