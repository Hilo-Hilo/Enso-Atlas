#!/usr/bin/env python3
"""
CLAM Single-Branch Attention MIL Training Script
For TCGA ovarian cancer platinum sensitivity prediction
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
import argparse
from pathlib import Path
import json

# ============== CLAM Model Architecture ==============

class GatedAttention(nn.Module):
    """Gated Attention mechanism for MIL"""
    def __init__(self, input_dim=384, hidden_dim=256, dropout=0.25):
        super().__init__()
        self.attention_a = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Dropout(dropout)
        )
        self.attention_b = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Sigmoid(),
            nn.Dropout(dropout)
        )
        self.attention_c = nn.Linear(hidden_dim, 1)
    
    def forward(self, x):
        # x: [N, input_dim] where N is number of patches
        a = self.attention_a(x)  # [N, hidden_dim]
        b = self.attention_b(x)  # [N, hidden_dim]
        A = a * b  # Gated attention [N, hidden_dim]
        A = self.attention_c(A)  # [N, 1]
        return A


class CLAM_SB(nn.Module):
    """CLAM Single-Branch Model with Gated Attention"""
    def __init__(self, input_dim=384, hidden_dim=256, n_classes=2, dropout=0.25):
        super().__init__()
        
        # Feature encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Gated attention
        self.attention = GatedAttention(hidden_dim, hidden_dim // 2, dropout)
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, n_classes)
        )
        
        self.n_classes = n_classes
    
    def forward(self, x, return_attention=False):
        """
        Args:
            x: [N, input_dim] - bag of patch features
            return_attention: whether to return attention weights
        Returns:
            logits: [1, n_classes]
            attention_weights: [N] (if return_attention=True)
        """
        # Encode features
        h = self.encoder(x)  # [N, hidden_dim]
        
        # Compute attention scores
        A = self.attention(h)  # [N, 1]
        A = torch.transpose(A, 1, 0)  # [1, N]
        A = F.softmax(A, dim=1)  # Normalize attention weights
        
        # Weighted aggregation
        M = torch.mm(A, h)  # [1, hidden_dim]
        
        # Classification
        logits = self.classifier(M)  # [1, n_classes]
        
        if return_attention:
            return logits, A.squeeze(0)  # Return [N] attention weights
        return logits


# ============== Dataset ==============

class TCGADataset(Dataset):
    def __init__(self, slide_ids, labels, embeddings_dir):
        self.slide_ids = slide_ids
        self.labels = labels
        self.embeddings_dir = Path(embeddings_dir)
        
    def __len__(self):
        return len(self.slide_ids)
    
    def __getitem__(self, idx):
        slide_id = self.slide_ids[idx]
        label = self.labels[idx]
        
        # Load embeddings
        emb_path = self.embeddings_dir / f"{slide_id}.npy"
        features = np.load(emb_path)
        features = torch.from_numpy(features).float()
        
        return features, label, slide_id


def collate_fn(batch):
    """Custom collate for variable-sized bags"""
    features, labels, slide_ids = zip(*batch)
    return features[0], labels[0], slide_ids[0]  # Process one bag at a time


# ============== Training ==============

def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    for features, label, _ in dataloader:
        features = features.to(device)
        label = torch.tensor([label], dtype=torch.long).to(device)
        
        optimizer.zero_grad()
        logits = model(features)
        loss = criterion(logits, label)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        probs = F.softmax(logits, dim=1)
        all_preds.append(probs[0, 1].item())
        all_labels.append(label.item())
    
    avg_loss = total_loss / len(dataloader)
    auc = roc_auc_score(all_labels, all_preds) if len(set(all_labels)) > 1 else 0.5
    return avg_loss, auc


def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for features, label, _ in dataloader:
            features = features.to(device)
            label = torch.tensor([label], dtype=torch.long).to(device)
            
            logits = model(features)
            loss = criterion(logits, label)
            
            total_loss += loss.item()
            probs = F.softmax(logits, dim=1)
            all_preds.append(probs[0, 1].item())
            all_labels.append(label.item())
    
    avg_loss = total_loss / len(dataloader)
    auc = roc_auc_score(all_labels, all_preds) if len(set(all_labels)) > 1 else 0.5
    acc = accuracy_score(all_labels, [1 if p > 0.5 else 0 for p in all_preds])
    
    return avg_loss, auc, acc, all_preds, all_labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--embeddings_dir', type=str, 
                        default='/home/hansonwen/med-gemma-hackathon/data/tcga_full/embeddings')
    parser.add_argument('--labels_csv', type=str,
                        default='/home/hansonwen/med-gemma-hackathon/data/tcga_full/labels.csv')
    parser.add_argument('--output_dir', type=str,
                        default='/home/hansonwen/med-gemma-hackathon/models')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--patience', type=int, default=7)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    
    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load labels
    labels_df = pd.read_csv(args.labels_csv)
    print(f"Loaded {len(labels_df)} labeled slides")
    
    # Filter to slides that have embeddings
    embeddings_dir = Path(args.embeddings_dir)
    available_slides = set()
    for f in embeddings_dir.glob("*.npy"):
        if not f.name.endswith("_coords.npy"):
            available_slides.add(f.stem)
    
    labels_df = labels_df[labels_df['slide_id'].isin(available_slides)]
    print(f"Found {len(labels_df)} slides with embeddings")
    print(f"Label distribution: {labels_df['label'].value_counts().to_dict()}")
    
    # Split data
    train_df, val_df = train_test_split(
        labels_df, test_size=0.2, random_state=args.seed, stratify=labels_df['label']
    )
    print(f"Train: {len(train_df)}, Val: {len(val_df)}")
    
    # Create datasets
    train_dataset = TCGADataset(
        train_df['slide_id'].tolist(),
        train_df['label'].tolist(),
        args.embeddings_dir
    )
    val_dataset = TCGADataset(
        val_df['slide_id'].tolist(),
        val_df['label'].tolist(),
        args.embeddings_dir
    )
    
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)
    
    # Initialize model
    model = CLAM_SB(input_dim=384, hidden_dim=256, n_classes=2, dropout=0.25)
    model = model.to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Loss and optimizer - weighted for class imbalance
    class_counts = labels_df['label'].value_counts().sort_index()
    weights = torch.tensor([1.0 / class_counts[0], 1.0 / class_counts[1]], dtype=torch.float32).to(device)
    weights = weights / weights.sum()
    criterion = nn.CrossEntropyLoss(weight=weights)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=3, factor=0.5)
    
    # Training loop
    os.makedirs(args.output_dir, exist_ok=True)
    best_auc = 0
    patience_counter = 0
    history = {'train_loss': [], 'train_auc': [], 'val_loss': [], 'val_auc': [], 'val_acc': []}
    
    print("\nStarting training...")
    for epoch in range(args.epochs):
        train_loss, train_auc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_auc, val_acc, _, _ = evaluate(model, val_loader, criterion, device)
        
        scheduler.step(val_auc)
        
        history['train_loss'].append(train_loss)
        history['train_auc'].append(train_auc)
        history['val_loss'].append(val_loss)
        history['val_auc'].append(val_auc)
        history['val_acc'].append(val_acc)
        
        print(f"Epoch {epoch+1:02d}/{args.epochs}: "
              f"Train Loss={train_loss:.4f}, Train AUC={train_auc:.4f} | "
              f"Val Loss={val_loss:.4f}, Val AUC={val_auc:.4f}, Val Acc={val_acc:.4f}")
        
        # Save best model
        if val_auc > best_auc:
            best_auc = val_auc
            patience_counter = 0
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_auc': best_auc,
                'model_config': {
                    'input_dim': 384,
                    'hidden_dim': 256,
                    'n_classes': 2,
                    'dropout': 0.25
                }
            }
            torch.save(checkpoint, os.path.join(args.output_dir, 'clam_attention.pt'))
            print(f"  -> Saved best model (AUC: {best_auc:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"\nEarly stopping at epoch {epoch+1}")
                break
    
    # Save training history
    with open(os.path.join(args.output_dir, 'training_history.json'), 'w') as f:
        json.dump(history, f, indent=2)
    
    print(f"\nTraining complete! Best Val AUC: {best_auc:.4f}")
    print(f"Model saved to: {os.path.join(args.output_dir, 'clam_attention.pt')}")


if __name__ == '__main__':
    main()
