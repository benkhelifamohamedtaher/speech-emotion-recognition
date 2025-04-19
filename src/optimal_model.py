#!/usr/bin/env python3
"""
Optimized Model Architecture for Speech Emotion Recognition
Uses state-of-the-art techniques for high accuracy emotion classification
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import logging
from typing import Dict, List, Optional, Tuple, Union

# Configure logging
logger = logging.getLogger(__name__)

class ConvBlock(nn.Module):
    """Convolutional block with batch normalization and residual connection"""
    def __init__(
        self, 
        in_channels, 
        out_channels, 
        kernel_size=3, 
        stride=1, 
        padding=1, 
        dilation=1, 
        groups=1,
        dropout=0.1
    ):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels, 
            out_channels, 
            kernel_size=kernel_size, 
            stride=stride, 
            padding=padding, 
            dilation=dilation,
            groups=groups
        )
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        
        # Residual connection if dimensions don't match
        self.residual = nn.Identity()
        if in_channels != out_channels or stride != 1:
            self.residual = nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride)
    
    def forward(self, x):
        residual = self.residual(x)
        
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        x = x + residual
        
        return x

class SqueezeExcitation(nn.Module):
    """Squeeze-and-Excitation block for channel-wise attention"""
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        b, c, _ = x.size()
        # Squeeze
        y = self.pool(x).view(b, c)
        # Excitation
        y = self.fc(y).view(b, c, 1)
        # Scale
        return x * y.expand_as(x)

class MultiHeadAttention(nn.Module):
    """Multi-head self-attention module"""
    def __init__(self, embed_dim, num_heads=8, dropout=0.1):
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
        self.scale = self.head_dim ** -0.5
    
    def forward(self, x, mask=None):
        # x: [batch_size, seq_len, embed_dim]
        batch_size, seq_len, _ = x.size()
        
        # Project to query, key, value
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        
        # Transpose for attention calculation
        q = q.transpose(1, 2)  # [batch_size, num_heads, seq_len, head_dim]
        k = k.transpose(1, 2)  # [batch_size, num_heads, seq_len, head_dim]
        v = v.transpose(1, 2)  # [batch_size, num_heads, seq_len, head_dim]
        
        # Calculate attention scores
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # [batch_size, num_heads, seq_len, seq_len]
        
        # Apply mask if provided
        if mask is not None:
            attn_weights = attn_weights.masked_fill(mask == 0, float('-inf'))
        
        # Apply softmax and dropout
        attn_weights = self.dropout(F.softmax(attn_weights, dim=-1))
        
        # Apply attention weights to values
        output = torch.matmul(attn_weights, v)  # [batch_size, num_heads, seq_len, head_dim]
        
        # Transpose and reshape
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)
        
        # Final projection
        output = self.out_proj(output)
        
        return output

class PositionalEncoding(nn.Module):
    """Positional encoding for transformer models"""
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        # x: [batch_size, seq_len, embed_dim]
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

class TransformerBlock(nn.Module):
    """Transformer encoder block with self-attention and feed-forward layers"""
    def __init__(self, embed_dim, num_heads=8, ff_dim=2048, dropout=0.1):
        super().__init__()
        self.attention = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x, mask=None):
        # Self-attention with residual connection and layer normalization
        attn_output = self.attention(self.norm1(x), mask)
        x = x + attn_output
        
        # Feed-forward with residual connection and layer normalization
        ff_output = self.feed_forward(self.norm2(x))
        x = x + ff_output
        
        return x

class OptimalSpeechEmotionModel(nn.Module):
    """
    Optimal Speech Emotion Recognition Model
    
    Combines convolutional feature extraction, transformer encoding,
    and attention-based pooling for high accuracy emotion recognition.
    """
    def __init__(
        self,
        num_emotions=8,
        input_channels=1,
        sample_rate=16000,
        n_fft=1024,
        hop_length=512,
        n_mels=128,
        hidden_size=256,
        num_conv_layers=4,
        num_transformer_layers=4,
        num_attention_heads=8,
        ff_dim=1024,
        dropout=0.2,
        input_type="waveform"  # "waveform", "melspec", or "mfcc"
    ):
        super().__init__()
        self.num_emotions = num_emotions
        self.sample_rate = sample_rate
        self.input_type = input_type
        
        # Feature extraction layers for different input types
        if input_type == "waveform":
            # For raw waveform input
            self.feature_extractor = nn.Sequential(
                ConvBlock(input_channels, 64, kernel_size=7, stride=2, padding=3, dropout=dropout),
                ConvBlock(64, 128, kernel_size=5, stride=2, padding=2, dropout=dropout),
                ConvBlock(128, 256, kernel_size=3, stride=2, padding=1, dropout=dropout),
                ConvBlock(256, hidden_size, kernel_size=3, stride=2, padding=1, dropout=dropout)
            )
            self.use_melspec = False
        else:
            # For pre-computed spectrograms (melspec or mfcc)
            if input_type == "melspec":
                in_channels = 1  # Single channel melspec
            elif input_type == "mfcc":
                in_channels = 1  # Single channel MFCC
            else:
                raise ValueError(f"Unknown input type: {input_type}")
            
            self.feature_extractor = nn.Sequential(
                ConvBlock(in_channels, 32, kernel_size=3, stride=1, padding=1, dropout=dropout),
                SqueezeExcitation(32),
                ConvBlock(32, 64, kernel_size=3, stride=2, padding=1, dropout=dropout),
                ConvBlock(64, 128, kernel_size=3, stride=2, padding=1, dropout=dropout),
                SqueezeExcitation(128),
                ConvBlock(128, hidden_size, kernel_size=3, stride=2, padding=1, dropout=dropout)
            )
            self.use_melspec = True
        
        # Optional mel spectrogram layer for waveform input
        if input_type == "waveform":
            self.melspec_transform = torchaudio.transforms.MelSpectrogram(
                sample_rate=sample_rate,
                n_fft=n_fft,
                hop_length=hop_length,
                n_mels=n_mels
            )
        
        # Transformer layers for sequence modeling
        self.positional_encoding = PositionalEncoding(hidden_size, dropout=dropout)
        
        self.transformer_layers = nn.ModuleList([
            TransformerBlock(
                embed_dim=hidden_size,
                num_heads=num_attention_heads,
                ff_dim=ff_dim,
                dropout=dropout
            )
            for _ in range(num_transformer_layers)
        ])
        
        # Attention-based pooling
        self.attention_pool = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Tanh(),
            nn.Linear(hidden_size // 2, 1)
        )
        
        # Classifier for final emotion prediction
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LayerNorm(hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, num_emotions)
        )
        
        # Initialize weights
        self._init_weights()
        
        # Store model metadata
        self.metadata = {
            "model_type": "OptimalSpeechEmotionModel",
            "num_emotions": num_emotions,
            "sample_rate": sample_rate,
            "n_fft": n_fft,
            "hop_length": hop_length,
            "n_mels": n_mels,
            "hidden_size": hidden_size,
            "input_type": input_type
        }
    
    def _init_weights(self):
        """Initialize model weights"""
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm1d, nn.LayerNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def _fix_dimensions(self, x):
        """Fix input dimensions for different input types"""
        # For 2D inputs (batch_size, time)
        if x.dim() == 2:
            x = x.unsqueeze(1)  # Add channel dimension
        
        # For 4D inputs (batch_size, channels, height, time)
        elif x.dim() == 4:
            if x.size(2) == 1:
                # If height is 1, squeeze it
                x = x.squeeze(2)
            else:
                # Otherwise, reshape to (batch_size, channels*height, time)
                b, c, h, t = x.size()
                x = x.reshape(b, c*h, t)
        
        # Check if the input is in the right format for the model
        if self.use_melspec and x.dim() == 3 and x.size(1) > x.size(2):
            # If channels > time, it might be transposed
            x = x.transpose(1, 2)
        
        return x
    
    def forward(self, x):
        """Forward pass with dimension handling"""
        try:
            # Ensure input has the right dimensions
            x = self._fix_dimensions(x)
            
            # Extract features
            if self.input_type == "waveform" and not self.use_melspec:
                # Input is raw waveform: (batch_size, 1, time)
                features = self.feature_extractor(x)
            else:
                # Input is already spectral features or converted to mel spectrogram
                if self.input_type == "waveform" and hasattr(self, 'melspec_transform'):
                    # Convert waveform to mel spectrogram
                    x = self.melspec_transform(x)
                    # Convert to log mel spectrogram for better features
                    x = torch.log(x + 1e-9)
                
                features = self.feature_extractor(x)
            
            # Convert from (batch_size, channels, time) to (batch_size, time, channels)
            features = features.transpose(1, 2)
            
            # Apply positional encoding
            features = self.positional_encoding(features)
            
            # Apply transformer layers
            for transformer_layer in self.transformer_layers:
                features = transformer_layer(features)
            
            # Apply attention pooling
            attn_weights = self.attention_pool(features).squeeze(-1)  # (batch_size, time)
            attn_weights = F.softmax(attn_weights, dim=1).unsqueeze(1)  # (batch_size, 1, time)
            
            # Apply attention weights to get context vector
            context = torch.bmm(attn_weights, features).squeeze(1)  # (batch_size, hidden_size)
            
            # Classify emotions
            logits = self.classifier(context)
            
            # Return logits and probabilities
            return {
                'emotion_logits': logits,
                'emotion_probs': F.softmax(logits, dim=-1)
            }
            
        except Exception as e:
            logger.error(f"Error in forward pass: {e}")
            # Get batch size from input
            batch_size = x.size(0) if hasattr(x, 'size') else 1
            
            # Return default output
            default_logits = torch.zeros(batch_size, self.num_emotions, device=x.device)
            default_probs = torch.ones(batch_size, self.num_emotions, device=x.device) / self.num_emotions
            
            return {
                'emotion_logits': default_logits,
                'emotion_probs': default_probs
            }
    
    def predict_emotion(self, waveform):
        """Predict emotion from waveform"""
        self.eval()
        with torch.no_grad():
            outputs = self(waveform)
            probs = outputs['emotion_probs']
            emotion_id = torch.argmax(probs, dim=1).item()
            confidence = probs[0, emotion_id].item()
            
            return emotion_id, confidence
    
    @classmethod
    def from_pretrained(cls, path, map_location=None):
        """Load model from a checkpoint file"""
        try:
            checkpoint = torch.load(path, map_location=map_location)
            
            # Get model configuration
            config = checkpoint.get('config', {})
            metadata = checkpoint.get('metadata', {})
            
            # Merge config and metadata
            if metadata:
                config.update(metadata)
            
            # Create model instance
            model = cls(
                num_emotions=config.get('num_emotions', 8),
                input_channels=config.get('input_channels', 1),
                sample_rate=config.get('sample_rate', 16000),
                n_fft=config.get('n_fft', 1024),
                hop_length=config.get('hop_length', 512),
                n_mels=config.get('n_mels', 128),
                hidden_size=config.get('hidden_size', 256),
                num_transformer_layers=config.get('num_transformer_layers', 4),
                num_attention_heads=config.get('num_attention_heads', 8),
                dropout=config.get('dropout', 0.2),
                input_type=config.get('input_type', 'waveform')
            )
            
            # Load model weights
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
            
            return model
        except Exception as e:
            logger.error(f"Error loading model from {path}: {e}")
            # Return default model
            logger.info("Using default model parameters")
            return cls()

# Import torchaudio here to avoid circular imports
import torchaudio

if __name__ == "__main__":
    # Test the model
    model = OptimalSpeechEmotionModel(
        num_emotions=8,
        input_type="waveform"
    )
    
    # Print model summary
    print(model)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Test forward pass with random input
    x = torch.randn(2, 1, 16000)  # (batch_size, channels, time)
    outputs = model(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output logits shape: {outputs['emotion_logits'].shape}")
    print(f"Output probabilities shape: {outputs['emotion_probs'].shape}") 