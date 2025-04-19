#!/usr/bin/env python
"""
Enhanced Speech Emotion Recognition Model optimized for RAVDESS dataset
This model incorporates advanced architecture elements including:
- Multi-head self-attention
- Residual connections
- Layer normalization
- Dropout for regularization
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import math

logger = logging.getLogger(__name__)

class ConvBlock(nn.Module):
    """Convolutional block with batch normalization and residual connection"""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dropout=0.1):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        
        # Residual connection
        self.residual = nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride) if in_channels != out_channels or stride != 1 else nn.Identity()
        
    def forward(self, x):
        residual = self.residual(x)
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = x + residual
        return x

class SelfAttention(nn.Module):
    """Multi-head self-attention module"""
    def __init__(self, embed_dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
        
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        batch_size, seq_len, embed_dim = x.size()
        
        # Project inputs to queries, keys, and values
        q = self.q_proj(x).reshape(batch_size, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = self.k_proj(x).reshape(batch_size, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.v_proj(x).reshape(batch_size, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        
        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention weights
        context = torch.matmul(attention_weights, v)
        context = context.permute(0, 2, 1, 3).reshape(batch_size, seq_len, embed_dim)
        output = self.out_proj(context)
        
        return output

class EnhancedRAVDESSModel(nn.Module):
    """Enhanced Speech Emotion Recognition Model for RAVDESS dataset"""
    def __init__(self, num_emotions=8, input_dim=1, channels=[32, 64, 128, 256], 
                 hidden_dim=512, dropout=0.3, attention_heads=4):
        super().__init__()
        
        self.num_emotions = num_emotions
        
        # Initial convolutional layers
        self.conv_layers = nn.ModuleList()
        in_channels = input_dim
        
        for i, out_channels in enumerate(channels):
            stride = 2 if i < len(channels) - 1 else 1
            self.conv_layers.append(
                ConvBlock(in_channels, out_channels, kernel_size=3, stride=stride, dropout=dropout)
            )
            in_channels = out_channels
        
        # Bidirectional LSTM for temporal modeling
        self.lstm = nn.LSTM(
            input_size=channels[-1],
            hidden_size=hidden_dim // 2,
            num_layers=2,
            bidirectional=True,
            dropout=dropout if dropout > 0 else 0,
            batch_first=True
        )
        
        # Self-attention mechanism
        self.attention = SelfAttention(hidden_dim, num_heads=attention_heads, dropout=dropout)
        self.layer_norm1 = nn.LayerNorm(hidden_dim)
        self.layer_norm2 = nn.LayerNorm(hidden_dim)
        
        # Feed-forward network
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )
        
        # Global average pooling for time dimension
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_emotions)
        )
    
    def forward(self, x):
        # Ensure input has the right shape: [batch_size, channels, time]
        if x.dim() == 2:
            x = x.unsqueeze(1)  # Add channel dimension
        
        # Apply convolutional layers
        for conv in self.conv_layers:
            x = conv(x)
        
        # Reshape for sequence processing: [batch_size, time, channels]
        x = x.transpose(1, 2)
        
        # Apply LSTM
        x, _ = self.lstm(x)
        
        # Apply self-attention with residual connection and layer normalization
        residual = x
        x = self.layer_norm1(x)
        x = self.attention(x)
        x = x + residual
        
        # Apply feed-forward with residual connection and layer normalization
        residual = x
        x = self.layer_norm2(x)
        x = self.feed_forward(x)
        x = x + residual
        
        # Global pooling over time dimension
        x = x.transpose(1, 2)  # [batch_size, channels, time]
        x = self.global_pool(x)
        x = x.squeeze(-1)  # [batch_size, channels]
        
        # Classification
        logits = self.classifier(x)
        probabilities = F.softmax(logits, dim=1)
        
        return probabilities

# For backward compatibility, also expose as RAVDESSEmotionModel
RAVDESSEmotionModel = EnhancedRAVDESSModel

if __name__ == "__main__":
    # Quick test of the model
    model = EnhancedRAVDESSModel(num_emotions=8)
    x = torch.randn(2, 48000)  # 2 samples, 3 seconds at 16kHz
    out = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")
    print("Model summary:")
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_params:,}") 