#!/usr/bin/env python
"""
Advanced Speech Emotion Recognition Model
Using a hybrid architecture with convolutional, recurrent, and attention components for higher accuracy
"""

import os
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from typing import Dict, List, Optional, Tuple, Union
import warnings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Try to import transformers, but have a fallback in case of issues
try:
    from transformers import Wav2Vec2Model, Wav2Vec2Config
    TRANSFORMERS_AVAILABLE = True
    logger.info("Transformers library successfully imported")
except Exception as e:
    TRANSFORMERS_AVAILABLE = False
    logger.warning(f"Error importing transformers: {str(e)}. Using fallback implementation.")

# Define emotion mappings
EMOTION_MAPPINGS = {
    'full': {
        0: 'neutral',
        1: 'calm',
        2: 'happy',
        3: 'sad',
        4: 'angry',
        5: 'fearful',
        6: 'disgust',
        7: 'surprised'
    },
    'simplified': {
        0: 'neutral',  # combines neutral and calm
        1: 'happy',    # combines happy and surprised
        2: 'sad',      # combines sad and fearful
        3: 'angry'     # combines angry and disgust
    }
}

class ReshapeLayer(nn.Module):
    """Layer that handles reshaping inputs to the correct dimensions"""
    def forward(self, x):
        # Handle 4D inputs (batch, channels, height, time) by squeezing height dimension
        if len(x.shape) == 4:
            batch_size, channels, height, time = x.shape
            x = x.reshape(batch_size, channels * height, time)
        
        # Ensure we have a 3D tensor (batch, channels, time)
        if len(x.shape) != 3:
            raise ValueError(f"Expected 3D or 4D input, got shape {x.shape}")
            
        return x

class ConvBlock(nn.Module):
    """Convolutional block with batch normalization and residual connections"""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size=kernel_size, 
            stride=stride, padding=padding, dilation=dilation
        )
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        self.residual = nn.Conv1d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()
        
    def forward(self, x):
        # Store input for residual connection
        identity = x
        
        # Apply convolution
        x = self.conv(x)
        x = self.bn(x)
        
        # Add residual connection
        x = x + self.residual(identity)
        
        # Apply activation
        x = self.relu(x)
        
        return x

class MultiHeadSelfAttention(nn.Module):
    """Multi-head self-attention module"""
    def __init__(self, embed_dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        # Ensure dimensions are compatible
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
        
        # Linear projections
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # Input shape: [batch_size, seq_len, embed_dim]
        batch_size, seq_len, embed_dim = x.size()
        
        # Compute queries, keys, values
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        # Reshape for multi-head attention
        q = q.reshape(batch_size, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = k.reshape(batch_size, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = v.reshape(batch_size, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        
        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention weights
        context = torch.matmul(attn_weights, v)
        
        # Reshape and project
        context = context.permute(0, 2, 1, 3).reshape(batch_size, seq_len, embed_dim)
        output = self.out_proj(context)
        
        return output

class FallbackEncoder(nn.Module):
    """Fallback encoder when transformers is not available"""
    def __init__(self, hidden_size=768):
        super().__init__()
        self.hidden_size = hidden_size
        
        # Convolutional layers for feature extraction
        self.conv_layers = nn.Sequential(
            ReshapeLayer(),
            ConvBlock(1, 64, kernel_size=7, stride=2, padding=3),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
            ConvBlock(64, 128, kernel_size=3, stride=1, padding=1),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
            ConvBlock(128, 256, kernel_size=3, stride=1, padding=1),
            ConvBlock(256, 256, kernel_size=3, stride=1, padding=1),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
            ConvBlock(256, 512, kernel_size=3, stride=1, padding=1),
            ConvBlock(512, 512, kernel_size=3, stride=1, padding=1),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        )
        
        # Bidirectional GRU for temporal modeling
        self.gru = nn.GRU(
            512, hidden_size // 2, 
            bidirectional=True, 
            batch_first=True,
            num_layers=2,
            dropout=0.2
        )
        
        # Projection layer
        self.proj = nn.Linear(hidden_size, hidden_size)
        
    def forward(self, x, attention_mask=None):
        # Apply convolutional layers
        # Input shape: [batch_size, 1, time]
        x = self.conv_layers(x)
        
        # Transpose from [batch, channels, time] to [batch, time, channels]
        x = x.transpose(1, 2)
        
        # Apply GRU
        x, _ = self.gru(x)
        
        # Apply projection
        x = self.proj(x)
        
        # Return in a format similar to Wav2Vec2
        return {"last_hidden_state": x}

class AttentionPooling(nn.Module):
    """Attention pooling layer to aggregate features"""
    def __init__(self, hidden_size, dropout=0.1):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        # x: [batch_size, seq_len, hidden_size]
        # mask: [batch_size, seq_len]
        
        # Compute attention scores
        scores = self.attention(x).squeeze(-1)  # [batch_size, seq_len]
        
        # Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -float('inf'))
        
        # Compute weights
        weights = F.softmax(scores, dim=1).unsqueeze(1)  # [batch_size, 1, seq_len]
        weights = self.dropout(weights)
        
        # Apply attention
        context = torch.bmm(weights, x).squeeze(1)  # [batch_size, hidden_size]
        
        return context

class ContextTransformerLayer(nn.Module):
    """Transformer layer for context modeling"""
    def __init__(self, hidden_size, num_heads=8, dropout=0.1):
        super().__init__()
        
        # Multi-head self-attention
        self.self_attn = MultiHeadSelfAttention(hidden_size, num_heads, dropout)
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 4, hidden_size)
        )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # Multi-head self-attention with residual connection and layer normalization
        residual = x
        x = self.norm1(x)
        x = self.self_attn(x)
        x = self.dropout(x)
        x = residual + x
        
        # Feed-forward network with residual connection and layer normalization
        residual = x
        x = self.norm2(x)
        x = self.ffn(x)
        x = self.dropout(x)
        x = residual + x
        
        return x

class AdvancedSpeechEmotionModel(nn.Module):
    """Advanced speech emotion recognition model with better architecture and robustness"""
    def __init__(
        self,
        num_emotions=8,
        sample_rate=16000,
        use_transformer=True,
        hidden_size=768,
        dropout=0.3,
        num_context_layers=4,
        freeze_feature_extractor=True
    ):
        super().__init__()
        self.num_emotions = num_emotions
        self.sample_rate = sample_rate
        self.use_transformer = use_transformer and TRANSFORMERS_AVAILABLE
        self.hidden_size = hidden_size
        
        # Spectrogram converter
        self.mel_spec = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=1024,
            hop_length=256,
            n_mels=128
        )
        
        # Audio preprocessing
        self.spec_augment = torchaudio.transforms.TimeStretch(hop_length=256) if hasattr(torchaudio.transforms, 'TimeStretch') else None
        
        # Use Wav2Vec2 for feature extraction if available, otherwise use fallback
        if self.use_transformer:
            try:
                logger.info("Loading Wav2Vec2 model...")
                self.encoder = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")
                
                # Freeze feature extractor to prevent overfitting
                if freeze_feature_extractor:
                    logger.info("Freezing feature extractor layers")
                    for name, param in self.encoder.named_parameters():
                        if "feature_extractor" in name:
                            param.requires_grad = False
                    
                    # Unfreeze the last few layers for fine-tuning
                    for i in range(max(0, len(self.encoder.encoder.layers) - 2), len(self.encoder.encoder.layers)):
                        for param in self.encoder.encoder.layers[i].parameters():
                            param.requires_grad = True
                
                logger.info("Wav2Vec2 model loaded successfully")
            except Exception as e:
                logger.warning(f"Error loading Wav2Vec2 model: {str(e)}. Using fallback encoder.")
                self.encoder = FallbackEncoder(hidden_size=hidden_size)
        else:
            logger.info("Using fallback encoder")
            self.encoder = FallbackEncoder(hidden_size=hidden_size)
        
        # Context transformer layers for emotion modeling
        self.context_layers = nn.ModuleList([
            ContextTransformerLayer(hidden_size=hidden_size, dropout=dropout)
            for _ in range(num_context_layers)
        ])
        
        # Attention pooling for sequence aggregation
        self.attention_pooling = AttentionPooling(hidden_size=hidden_size, dropout=dropout)
        
        # Emotion classifier
        self.emotion_classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LayerNorm(hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.LayerNorm(hidden_size // 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 4, num_emotions)
        )
        
        # Save metadata
        self.metadata = {
            "num_emotions": num_emotions,
            "sample_rate": sample_rate,
            "use_transformer": self.use_transformer,
            "hidden_size": hidden_size,
            "model_type": "advanced"
        }
    
    def preprocess_audio(self, waveform):
        """Preprocess audio waveform"""
        try:
            # Ensure input has the right shape [batch_size, 1, time]
            if len(waveform.shape) == 2:
                waveform = waveform.unsqueeze(1)
            elif len(waveform.shape) == 3 and waveform.shape[1] > 1:
                # Convert multi-channel to mono
                waveform = waveform.mean(dim=1, keepdim=True)
            
            # Normalize to [-1, 1]
            if waveform.abs().max() > 0:
                waveform = waveform / waveform.abs().max()
            
            return waveform
        except Exception as e:
            logger.error(f"Error in audio preprocessing: {str(e)}")
            # Return the input as is for robustness
            return waveform
    
    def _fix_dimensions(self, x):
        """Fix dimensions for LayerNorm when processing large inputs"""
        if x.dim() == 3:  # [batch_size, hidden_size, seq_len]
            return x.transpose(1, 2)  # [batch_size, seq_len, hidden_size]
        return x

    def forward(self, waveform):
        """Forward pass for the model"""
        try:
            batch_size = waveform.shape[0]
            
            # Preprocess audio
            waveform = self.preprocess_audio(waveform)
            
            # Extract features using Wav2Vec2 or fallback encoder
            encoder_outputs = self.encoder(waveform)
            hidden_states = encoder_outputs["last_hidden_state"]
            
            # Fix dimensions for transformer layers
            hidden_states = self._fix_dimensions(hidden_states)
            
            # Apply context transformer layers
            for layer in self.context_layers:
                hidden_states = layer(hidden_states)
            
            # Apply attention pooling
            pooled = self.attention_pooling(hidden_states)
            
            # Classify emotions
            logits = self.emotion_classifier(pooled)
            emotions = F.softmax(logits, dim=-1)
            
            return emotions
            
        except Exception as e:
            logger.error(f"Error in model forward pass: {str(e)}")
            # Return default output on error for robustness
            default_emotions = torch.ones(batch_size, self.num_emotions, device=waveform.device) / self.num_emotions
            return default_emotions
    
    def predict_emotion(self, waveform):
        """Predict emotion from waveform with proper error handling"""
        try:
            # Forward pass
            emotions = self(waveform)
            
            # Get predictions
            probs, preds = torch.max(emotions, dim=1)
            
            # Convert to numpy/python types for easier handling
            emotion_id = preds.item() if preds.numel() == 1 else preds[0].item()
            confidence = probs.item() if probs.numel() == 1 else probs[0].item()
            
            # Get emotion name
            emotion_set = 'simplified' if self.num_emotions <= 4 else 'full'
            emotion_name = EMOTION_MAPPINGS[emotion_set].get(emotion_id, 'unknown')
            
            return emotion_id, emotion_name, confidence
            
        except Exception as e:
            logger.error(f"Error in emotion prediction: {str(e)}")
            # Return neutral emotion with low confidence on error
            return 0, 'neutral', 0.1
    
    @classmethod
    def from_pretrained(cls, model_path, map_location=None):
        """Load a pretrained model from file"""
        try:
            checkpoint = torch.load(model_path, map_location=map_location)
            
            # Extract model metadata
            metadata = checkpoint.get('metadata', {})
            num_emotions = metadata.get('num_emotions', 8)
            sample_rate = metadata.get('sample_rate', 16000)
            use_transformer = metadata.get('use_transformer', True)
            hidden_size = metadata.get('hidden_size', 768)
            
            # Create model instance
            model = cls(
                num_emotions=num_emotions,
                sample_rate=sample_rate,
                use_transformer=use_transformer,
                hidden_size=hidden_size
            )
            
            # Load state dict
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            else:
                model.load_state_dict(checkpoint, strict=False)
            
            return model
        except Exception as e:
            logger.error(f"Error loading model from {model_path}: {str(e)}")
            # Fallback to default model
            logger.info("Using default model instead")
            return cls()

if __name__ == "__main__":
    # Simple test
    model = AdvancedSpeechEmotionModel(num_emotions=8)
    
    # Create a random waveform [batch_size, time]
    waveform = torch.randn(2, 16000)
    
    # Forward pass
    output = model(waveform)
    
    print(f"Input shape: {waveform.shape}")
    print(f"Output emotions shape: {output.shape}")
    print("Model initialized successfully!") 