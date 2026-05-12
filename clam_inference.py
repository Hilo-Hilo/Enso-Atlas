#!/usr/bin/env python3
"""
CLAM Attention Extraction Script
Returns attention weights per patch for heatmap visualization
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
import argparse
import json


# ============== CLAM Model (same as training) ==============

class GatedAttention(nn.Module):
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
        a = self.attention_a(x)
        b = self.attention_b(x)
        A = a * b
        A = self.attention_c(A)
        return A


class CLAM_SB(nn.Module):
    def __init__(self, input_dim=384, hidden_dim=256, n_classes=2, dropout=0.25):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.attention = GatedAttention(hidden_dim, hidden_dim // 2, dropout)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, n_classes)
        )
        self.n_classes = n_classes
    
    def forward(self, x, return_attention=False):
        h = self.encoder(x)
        A = self.attention(h)
        A = torch.transpose(A, 1, 0)
        A = F.softmax(A, dim=1)
        M = torch.mm(A, h)
        logits = self.classifier(M)
        
        if return_attention:
            return logits, A.squeeze(0)
        return logits


def load_model(checkpoint_path, device='cpu'):
    """Load trained CLAM model"""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint.get('model_config', {
        'input_dim': 384,
        'hidden_dim': 256,
        'n_classes': 2,
        'dropout': 0.25
    })
    
    model = CLAM_SB(**config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    return model, config


def extract_attention(model, features, device='cpu'):
    """
    Extract attention weights for a slide
    
    Args:
        model: Trained CLAM model
        features: [N, 384] numpy array of patch features
        device: torch device
    
    Returns:
        dict with prediction and attention weights
    """
    features_tensor = torch.from_numpy(features).float().to(device)
    
    with torch.no_grad():
        logits, attention = model(features_tensor, return_attention=True)
        probs = F.softmax(logits, dim=1)
        pred_class = torch.argmax(probs, dim=1).item()
        pred_prob = probs[0, 1].item()  # Probability of class 1
    
    return {
        'prediction': pred_class,
        'probability': pred_prob,
        'attention_weights': attention.cpu().numpy()  # [N] array
    }


def get_slide_attention(model_path, embeddings_path, coords_path=None, device='cpu'):
    """
    Get attention weights with coordinates for a slide
    
    Args:
        model_path: Path to clam_attention.pt
        embeddings_path: Path to slide embeddings .npy file
        coords_path: Optional path to coordinates .npy file
        device: torch device
    
    Returns:
        dict with prediction, attention, and optionally coordinates
    """
    model, _ = load_model(model_path, device)
    features = np.load(embeddings_path)
    
    result = extract_attention(model, features, device)
    
    # Load coordinates if available
    if coords_path and Path(coords_path).exists():
        coords = np.load(coords_path)
        result['coordinates'] = coords  # [N, 2] array of (x, y) patch positions
    
    return result


def batch_extract(model_path, embeddings_dir, output_path, device='cpu'):
    """
    Extract attention for all slides and save to JSON
    """
    model, _ = load_model(model_path, device)
    embeddings_dir = Path(embeddings_dir)
    
    results = {}
    for emb_file in sorted(embeddings_dir.glob("*.npy")):
        if emb_file.name.endswith("_coords.npy"):
            continue
        
        slide_id = emb_file.stem
        features = np.load(emb_file)
        
        result = extract_attention(model, features, device)
        
        # Check for coordinates
        coords_file = emb_file.parent / f"{slide_id}_coords.npy"
        if coords_file.exists():
            coords = np.load(coords_file)
            result['coordinates'] = coords.tolist()
        
        # Convert numpy arrays to lists for JSON
        result['attention_weights'] = result['attention_weights'].tolist()
        results[slide_id] = result
        
        print(f"Processed {slide_id}: pred={result['prediction']}, prob={result['probability']:.3f}")
    
    with open(output_path, 'w') as f:
        json.dump(results, f)
    
    print(f"\nSaved attention data for {len(results)} slides to {output_path}")
    return results


# ============== Example Usage ==============

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, 
                        default='/home/hansonwen/med-gemma-hackathon/models/clam_attention.pt')
    parser.add_argument('--embeddings_dir', type=str,
                        default='/home/hansonwen/med-gemma-hackathon/data/tcga_full/embeddings')
    parser.add_argument('--slide_id', type=str, default=None,
                        help='Single slide to process (optional)')
    parser.add_argument('--output', type=str, 
                        default='/home/hansonwen/med-gemma-hackathon/models/attention_weights.json')
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()
    
    device = args.device if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    if args.slide_id:
        # Single slide
        emb_path = f"{args.embeddings_dir}/{args.slide_id}.npy"
        coords_path = f"{args.embeddings_dir}/{args.slide_id}_coords.npy"
        result = get_slide_attention(args.model, emb_path, coords_path, device)
        print(f"\nSlide: {args.slide_id}")
        print(f"Prediction: {result['prediction']} (prob: {result['probability']:.3f})")
        print(f"Attention weights shape: {result['attention_weights'].shape}")
        print(f"Top 5 attention values: {sorted(result['attention_weights'], reverse=True)[:5]}")
    else:
        # Batch all slides
        batch_extract(args.model, args.embeddings_dir, args.output, device)


# ============== API for Enso Atlas Integration ==============

def get_attention_for_visualization(model_path, slide_id, embeddings_dir):
    """
    Simple API for Enso Atlas to get attention data
    
    Returns:
        {
            'slide_id': str,
            'prediction': 0 or 1,
            'probability': float,
            'attention': list of floats (normalized 0-1),
            'coordinates': list of [x, y] if available
        }
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, _ = load_model(model_path, device)
    
    emb_path = Path(embeddings_dir) / f"{slide_id}.npy"
    coords_path = Path(embeddings_dir) / f"{slide_id}_coords.npy"
    
    features = np.load(emb_path)
    result = extract_attention(model, features, device)
    
    output = {
        'slide_id': slide_id,
        'prediction': result['prediction'],
        'probability': result['probability'],
        'attention': result['attention_weights'].tolist()
    }
    
    if coords_path.exists():
        output['coordinates'] = np.load(coords_path).tolist()
    
    return output
