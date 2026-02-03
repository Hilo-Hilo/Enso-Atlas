"""
TransMIL: Transformer-based Multiple Instance Learning
Based on: https://arxiv.org/abs/2106.00908

Supports attention heatmap visualization via multi-head self-attention weights.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class PPEG(nn.Module):
    """Pyramid Position Encoding Generator."""
    
    def __init__(self, dim=512):
        super().__init__()
        self.proj = nn.Conv2d(dim, dim, 7, 1, 7 // 2, groups=dim)
        self.proj1 = nn.Conv2d(dim, dim, 5, 1, 5 // 2, groups=dim)
        self.proj2 = nn.Conv2d(dim, dim, 3, 1, 3 // 2, groups=dim)

    def forward(self, x, H, W):
        B, _, C = x.shape
        cls_token, feat_token = x[:, 0], x[:, 1:]
        cnn_feat = feat_token.transpose(1, 2).view(B, C, H, W)
        x = self.proj(cnn_feat) + cnn_feat + self.proj1(cnn_feat) + self.proj2(cnn_feat)
        x = x.flatten(2).transpose(1, 2)
        x = torch.cat((cls_token.unsqueeze(1), x), dim=1)
        return x


class TransLayer(nn.Module):
    """Transformer layer with multi-head self-attention."""
    
    def __init__(self, dim=512, num_heads=8, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, return_attn=False):
        residual = x
        x = self.norm(x)
        if return_attn:
            x, attn_weights = self.attn(x, x, x, need_weights=True, average_attn_weights=True)
            x = residual + self.dropout(x)
            return x, attn_weights
        else:
            x, _ = self.attn(x, x, x, need_weights=False)
            x = residual + self.dropout(x)
            return x


class TransMIL(nn.Module):
    """
    TransMIL: Transformer-based Correlated Multiple Instance Learning.
    
    Args:
        input_dim: Input feature dimension (384 for Path Foundation)
        hidden_dim: Transformer hidden dimension
        num_classes: Number of output classes (1 for binary with sigmoid)
        num_heads: Number of attention heads
        num_layers: Number of transformer layers
        dropout: Dropout rate
    """
    
    def __init__(
        self,
        input_dim=384,
        hidden_dim=512,
        num_classes=1,
        num_heads=8,
        num_layers=2,
        dropout=0.1
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        
        # Project input to hidden dim
        self.fc_in = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Class token
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim))
        
        # Transformer layers
        self.layers = nn.ModuleList([
            TransLayer(hidden_dim, num_heads, dropout) for _ in range(num_layers)
        ])
        
        # Position encoding
        self.ppeg = PPEG(hidden_dim)
        
        # Final norm and classifier
        self.norm = nn.LayerNorm(hidden_dim)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
            nn.Sigmoid() if num_classes == 1 else nn.Identity()
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x, return_attention=False):
        """
        Forward pass.
        
        Args:
            x: (N, input_dim) patch embeddings for single slide
            return_attention: Whether to return attention weights for heatmap
        
        Returns:
            logit: (1,) prediction probability
            attention: (N,) attention weights if return_attention=True
        """
        # x: (N, input_dim) -> need to add batch dim
        if x.dim() == 2:
            x = x.unsqueeze(0)  # (1, N, input_dim)
        
        B, N, _ = x.shape
        
        # Project to hidden dim
        x = self.fc_in(x)  # (B, N, hidden_dim)
        
        # Add class token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)  # (B, N+1, hidden_dim)
        
        # Compute spatial dims for PPEG (approximate square)
        H = W = int(np.ceil(np.sqrt(N)))
        
        # Pad if needed
        if H * W > N:
            padding = torch.zeros(B, H * W - N, self.hidden_dim, device=x.device)
            x_padded = torch.cat([x[:, :1], x[:, 1:], padding], dim=1)
        else:
            x_padded = x
        
        # Transformer layers with PPEG
        attn_weights_list = []
        for i, layer in enumerate(self.layers):
            if i == 0:
                x_padded = self.ppeg(x_padded, H, W)
            
            if return_attention and i == len(self.layers) - 1:
                x_padded, attn = layer(x_padded, return_attn=True)
                attn_weights_list.append(attn)
            else:
                x_padded = layer(x_padded)
        
        # Remove padding and get class token
        x = x_padded[:, :N+1]
        x = self.norm(x)
        cls_output = x[:, 0]  # (B, hidden_dim)
        
        # Classify
        logit = self.classifier(cls_output)  # (B, num_classes)
        
        if return_attention:
            # Extract attention from class token to all patches
            # attn shape: (B, N+1, N+1), we want cls_token -> patches
            attn = attn_weights_list[-1][:, 0, 1:N+1]  # (B, N)
            attn = attn.squeeze(0)  # (N,) for single slide
            return logit.squeeze(), attn
        
        return logit.squeeze()


class TransMILTrainer:
    """Training utilities for TransMIL."""
    
    def __init__(self, model, device='cuda', lr=1e-4, weight_decay=1e-5):
        self.model = model.to(device)
        self.device = device
        self.optimizer = torch.optim.AdamW(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=50, eta_min=1e-6
        )
    
    def train_step(self, embeddings, label):
        """Single training step."""
        self.model.train()
        self.optimizer.zero_grad()
        
        embeddings = embeddings.to(self.device)
        label = label.to(self.device)
        
        pred = self.model(embeddings)
        loss = F.binary_cross_entropy(pred.view(-1), label.view(-1))
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        
        return loss.item()
    
    @torch.no_grad()
    def eval_step(self, embeddings):
        """Single evaluation step."""
        self.model.eval()
        embeddings = embeddings.to(self.device)
        pred, attn = self.model(embeddings, return_attention=True)
        return pred.cpu().item(), attn.cpu().numpy()


if __name__ == "__main__":
    # Test
    model = TransMIL(input_dim=384, hidden_dim=512, num_classes=1)
    x = torch.randn(1000, 384)  # 1000 patches, 384 dim
    
    # Forward pass
    pred = model(x)
    print(f"Prediction: {pred.item():.4f}")
    
    # With attention
    pred, attn = model(x, return_attention=True)
    print(f"Prediction: {pred.item():.4f}, Attention shape: {attn.shape}")
    print(f"Attention sum: {attn.sum():.4f}")  # Should be ~1.0
