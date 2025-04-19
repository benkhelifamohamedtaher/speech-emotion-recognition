#!/usr/bin/env python3
"""
Advanced Emotion Recognition Model with Improved Architecture
Combines CNN feature extraction with transformer blocks for improved performance
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.transforms as T
import math
import logging
import numpy as np
from typing import Dict, Optional, Tuple, List, Union

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class WeightedFocalLoss(nn.Module):
    """Weighted Focal Loss to address class imbalance"""
    
    def __init__(self, samples_per_class, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        
        # Calculate class weights based on samples per class
        if samples_per_class is not None:
            total_samples = torch.sum(samples_per_class)
            class_weights = total_samples / (samples_per_class * len(samples_per_class))
            self.class_weights = class_weights / torch.sum(class_weights)
        else:
            self.class_weights = None
    
    def forward(self, inputs, targets):
        """
        Args:
            inputs: Logits from the model (B, C)
            targets: Ground truth class indices (B,)
        """
        # Get device
        device = inputs.device
        
        # Convert class weights to device if needed
        if self.class_weights is not None:
            class_weights = self.class_weights.to(device)
        
        # Apply softmax to convert logits to probabilities
        probs = F.softmax(inputs, dim=1)
        
        # Get probability of the target class
        batch_size = inputs.size(0)
        p_t = probs[torch.arange(batch_size), targets]
        
        # Apply the focal term
        focal_weight = (1 - p_t) ** self.gamma
        
        # Get alpha weight
        if self.class_weights is not None:
            alpha_t = class_weights[targets]
        else:
            alpha_t = self.alpha
        
        # Calculate focal loss
        focal_loss = -alpha_t * focal_weight * torch.log(p_t + 1e-7)
        
        return focal_loss.mean()


class SpecAugment(nn.Module):
    """SpecAugment augmentation for spectrograms
    
    Implements time masking and frequency masking as described in:
    "SpecAugment: A Simple Data Augmentation Method for Automatic Speech Recognition"
    """
    
    def __init__(self, 
                 freq_mask_param=10, 
                 time_mask_param=10,
                 freq_mask_num=2,
                 time_mask_num=2):
        super().__init__()
        self.freq_mask_param = freq_mask_param
        self.time_mask_param = time_mask_param
        self.freq_mask_num = freq_mask_num
        self.time_mask_num = time_mask_num
    
    def forward(self, spec):
        """
        Args:
            spec: Spectrogram (B, C, F, T) or (B, F, T)
        
        Returns:
            Augmented spectrogram with same shape
        """
        if spec.dim() == 4:
            # Handle (B, C, F, T) case - apply independently to each channel
            B, C, F, T = spec.size()
            for b in range(B):
                for c in range(C):
                    spec[b, c] = self._apply_augmentation(spec[b, c], F, T)
        else:
            # Handle (B, F, T) case
            B, F, T = spec.size()
            for b in range(B):
                spec[b] = self._apply_augmentation(spec[b], F, T)
                
        return spec
    
    def _apply_augmentation(self, spec, F, T):
        """Apply augmentation to a single spectrogram"""
        # Apply frequency masking
        for _ in range(self.freq_mask_num):
            f = int(torch.randint(0, self.freq_mask_param, (1,)).item())
            f0 = int(torch.randint(0, F - f, (1,)).item())
            spec[f0:f0 + f, :] = 0
        
        # Apply time masking
        for _ in range(self.time_mask_num):
            t = int(torch.randint(0, self.time_mask_param, (1,)).item())
            t0 = int(torch.randint(0, T - t, (1,)).item())
            spec[:, t0:t0 + t] = 0
            
        return spec


class MelSpectrogram(nn.Module):
    """Mel Spectrogram extractor with normalization"""
    
    def __init__(self, sample_rate=16000, n_fft=1024, hop_length=512, n_mels=128, normalize=True):
        super().__init__()
        self.transform = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            power=2.0
        )
        self.normalize = normalize
        
        # Add SpecAugment for training
        self.spec_augment = SpecAugment(
            freq_mask_param=20,  # Maximum frequency mask width
            time_mask_param=40,  # Maximum time mask width
            freq_mask_num=2,     # Number of frequency masks
            time_mask_num=2      # Number of time masks
        )
        self.training = True  # Training mode flag
    
    def forward(self, waveform, apply_augmentation=True):
        """
        Args:
            waveform: Audio signal (B, C, T) or (B, T)
            apply_augmentation: Whether to apply SpecAugment (for training)
        
        Returns:
            mel_spec: Mel spectrogram (B, n_mels, T')
        """
        # Handle shape issues - ensure we have the expected shape for torchaudio
        if waveform.dim() == 3 and waveform.size(1) == 1:
            # If shape is [B, 1, T], reshape to [B, T]
            waveform = waveform.squeeze(1)
        elif waveform.dim() == 3 and waveform.size(1) > 1:
            # If multi-channel, convert to mono by averaging channels
            waveform = torch.mean(waveform, dim=1)
        
        # Convert to mel spectrogram
        mel_spec = self.transform(waveform)
        
        # Add a small constant to avoid log(0)
        mel_spec = torch.log(mel_spec + 1e-9)
        
        # Instance normalization (normalize each spectrogram separately)
        if self.normalize:
            mean = torch.mean(mel_spec, dim=(1, 2), keepdim=True)
            std = torch.std(mel_spec, dim=(1, 2), keepdim=True) + 1e-9
            mel_spec = (mel_spec - mean) / std
        
        # Apply SpecAugment in training mode if requested and possible
        if self.training and apply_augmentation and mel_spec.dim() == 3:
            # Make sure mel_spec is the right shape for spec_augment (B, F, T)
            mel_spec = self.spec_augment(mel_spec)
        
        return mel_spec


class ConvBlock(nn.Module):
    """Convolutional block with batch normalization and residual connection"""
    
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, use_residual=True):
        super().__init__()
        self.use_residual = use_residual and (in_channels == out_channels)
        
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=kernel_size,
            stride=stride, padding=padding, dilation=dilation
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = nn.GELU()
    
    def forward(self, x):
        """
        Args:
            x: Input tensor (B, C, H, W)
        
        Returns:
            output: Output tensor (B, C_out, H', W')
        """
        identity = x
        out = self.conv(x)
        out = self.bn(out)
        out = self.activation(out)
        
        if self.use_residual:
            out = out + identity
        
        return out


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer models"""
    
    def __init__(self, d_model, max_len=1000):
        super().__init__()
        
        # Create positional encoding
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Register buffer (not a parameter but should be part of model state)
        self.register_buffer('pe', pe.unsqueeze(0))
    
    def forward(self, x):
        """
        Args:
            x: Input tensor (B, L, D)
        
        Returns:
            x_with_pos: Input with positional encoding added (B, L, D)
        """
        x = x + self.pe[:, :x.size(1), :]
        return x


class SelfAttention(nn.Module):
    """Improved Self-Attention module with relative positional encoding"""
    
    def __init__(self, d_model, num_heads, dropout=0.1, max_len=1000):
        super().__init__()
        
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # Projections
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.output = nn.Linear(d_model, d_model)
        
        # Relative positional embedding
        self.rel_pos_embed = nn.Parameter(torch.zeros(2 * max_len - 1, self.d_k))
        nn.init.xavier_normal_(self.rel_pos_embed)
        
        # Layer normalization
        self.norm = nn.LayerNorm(d_model)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # For caching attention weights
        self.attn_weights = None
    
    def _relative_position_to_index(self, seq_len):
        """Convert relative positions to indices in the relative positional embedding"""
        range_vec = torch.arange(seq_len)
        range_mat = range_vec.unsqueeze(0).repeat(seq_len, 1)
        distance_mat = range_mat - range_mat.transpose(0, 1)
        distance_mat_clipped = distance_mat + seq_len - 1  # shift to [0, 2*seq_len-1]
        return distance_mat_clipped
    
    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (B, L, D)
        
        Returns:
            output: Output tensor of shape (B, L, D)
        """
        batch_size, seq_len, _ = x.shape
        residual = x
        
        # Apply layer normalization (pre-norm)
        x = self.norm(x)
        
        # Linear projections
        q = self.query(x).view(batch_size, seq_len, self.num_heads, self.d_k)
        k = self.key(x).view(batch_size, seq_len, self.num_heads, self.d_k)
        v = self.value(x).view(batch_size, seq_len, self.num_heads, self.d_k)
        
        # Transpose for attention dot product: (B, H, L, D_k)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # Content-based attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.d_k ** 0.5)
        
        # Add relative positional embedding
        rel_pos_index = self._relative_position_to_index(seq_len).to(x.device)
        rel_pos = self.rel_pos_embed[rel_pos_index]  # [L, L, D_k]
        
        # Reshape for batch and heads
        rel_pos = rel_pos.unsqueeze(0).unsqueeze(0)  # [1, 1, L, L, D_k]
        rel_pos = rel_pos.expand(batch_size, self.num_heads, -1, -1, -1)  # [B, H, L, L, D_k]
        
        # Calculate positional attention term
        q_expanded = q.unsqueeze(-2)  # [B, H, L, 1, D_k]
        pos_scores = torch.matmul(q_expanded, rel_pos.transpose(-2, -1)).squeeze(-2)  # [B, H, L, L]
        
        # Add content and positional scores
        scores = scores + pos_scores
        
        # Apply attention
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Save attention weights for visualization
        self.attn_weights = attn_weights.detach()
        
        # Apply attention to values
        context = torch.matmul(attn_weights, v)
        
        # Transpose to move head dimension back: (B, L, H, D_k)
        context = context.transpose(1, 2).contiguous()
        
        # Combine heads
        context = context.view(batch_size, seq_len, self.d_model)
        
        # Final linear projection
        output = self.output(context)
        output = self.dropout(output)
        
        # Add residual connection
        return output + residual


class TransformerBlock(nn.Module):
    """Enhanced Transformer block with SwiGLU and improved attention"""
    
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1, max_len=1000):
        super().__init__()
        
        # Self-attention with relative positional embedding
        self.self_attn = SelfAttention(d_model, num_heads, dropout, max_len)
        
        # SwiGLU feedforward network (improved version of GELU)
        self.norm = nn.LayerNorm(d_model)
        self.w1 = nn.Linear(d_model, d_ff)
        self.w2 = nn.Linear(d_model, d_ff)
        self.w3 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
    
    def _swiglu(self, x):
        """SwiGLU activation function from Transformer with SwiGLU paper"""
        return F.silu(self.w1(x)) * self.w2(x)
    
    def forward(self, x):
        """
        Args:
            x: Input tensor (B, L, D)
        
        Returns:
            output: Output tensor (B, L, D)
        """
        # Multi-head attention
        x = self.self_attn(x)
        
        # SwiGLU FFN
        residual = x
        x = self.norm(x)
        x = self.w3(self._swiglu(x))
        x = self.dropout(x)
        x = x + residual
        
        return x


class AdvancedEmotionRecognitionModel(nn.Module):
    """Advanced Emotion Recognition Model with CNN and Transformer"""
    
    def __init__(self, 
                 num_emotions=8,
                 sample_rate=16000,
                 feature_dim=256,
                 hidden_dim=512,
                 transformer_layers=4,
                 transformer_heads=8,
                 dropout=0.1,
                 samples_per_class=None):
        super().__init__()
        
        # Model parameters
        self.num_emotions = num_emotions
        self.sample_rate = sample_rate
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        
        # Define feature extraction layers
        self.mel_extractor = MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=1024,
            hop_length=512,
            n_mels=128,
            normalize=True
        )
        
        # CNN feature extraction
        self.feature_extractor = nn.Sequential(
            ConvBlock(1, 32, kernel_size=3, stride=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            ConvBlock(32, 64, kernel_size=3, stride=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            ConvBlock(64, 128, kernel_size=3, stride=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            ConvBlock(128, feature_dim, kernel_size=3, stride=1),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Calculate feature map size after CNN
        # Assuming input mel spec is (batch, 1, 128, time)
        # After 4 max pool layers with stride 2, spatial dimensions are reduced by 2^4 = 16
        self.feat_h = 128 // 16  # mel bins dimension after CNN
        
        # Positional encoding for transformer
        self.pos_encoding = PositionalEncoding(feature_dim)
        
        # Transformer layers with enhanced attention
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(
                d_model=feature_dim,
                num_heads=transformer_heads,
                d_ff=hidden_dim,
                dropout=dropout,
                max_len=1000  # Maximum sequence length for relative position
            ) for _ in range(transformer_layers)
        ])
        
        # Output layer
        self.norm = nn.LayerNorm(feature_dim)
        self.pool = nn.AdaptiveAvgPool1d(1)
        
        # Multi-stage classifier with deep supervision
        self.classifier = nn.Linear(feature_dim, num_emotions)
        self.intermediate_classifiers = nn.ModuleList([
            nn.Linear(feature_dim, num_emotions) for _ in range(transformer_layers - 1)
        ])
        
        # Loss function
        self.loss_fn = WeightedFocalLoss(samples_per_class)
        
        # Mixup alpha parameter
        self.mixup_alpha = 0.2
    
    def _mixup_data(self, x, targets):
        """Apply mixup augmentation to the data
        
        Args:
            x: Input tensor (B, ...)
            targets: Target classes (B,)
            
        Returns:
            mixed_x: Mixed input
            target_a, target_b: Original targets
            lam: Mixing coefficient
        """
        if self.mixup_alpha > 0:
            lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)
        else:
            lam = 1
            
        batch_size = x.size(0)
        index = torch.randperm(batch_size).to(x.device)
        
        mixed_x = lam * x + (1 - lam) * x[index]
        target_a, target_b = targets, targets[index]
        
        return mixed_x, target_a, target_b, lam

    def _mixup_loss(self, pred, target_a, target_b, lam):
        """Compute mixup loss
        
        Args:
            pred: Model predictions
            target_a, target_b: Targets from mixup
            lam: Mixing coefficient
            
        Returns:
            loss: Mixed loss value
        """
        loss_a = self.loss_fn(pred, target_a)
        loss_b = self.loss_fn(pred, target_b)
        return lam * loss_a + (1 - lam) * loss_b
    
    def forward(self, waveform, emotion_targets=None, apply_mixup=False, apply_augmentation=True):
        """
        Args:
            waveform: Audio signal (B, T)
            emotion_targets: Ground truth emotion labels (B,) - Optional
            apply_mixup: Whether to apply mixup augmentation (training only)
            apply_augmentation: Whether to apply SpecAugment (training only)
            
        Returns:
            outputs: Dictionary with model outputs
        """
        # Setup mixup variables
        target_a = target_b = None
        lam = 1.0
        
        # Extract mel spectrogram
        mel_spec = self.mel_extractor(waveform, apply_augmentation=apply_augmentation)  # (B, mel_bins, time)
        
        # Add channel dimension
        mel_spec = mel_spec.unsqueeze(1)  # (B, 1, mel_bins, time)
        
        # Apply mixup in training if requested
        if self.training and apply_mixup and emotion_targets is not None:
            mel_spec, target_a, target_b, lam = self._mixup_data(mel_spec, emotion_targets)
        
        # Extract features using CNN
        features = self.feature_extractor(mel_spec)  # (B, feature_dim, H, W)
        
        # Reshape for transformer
        batch_size, channels, height, width = features.size()
        features = features.permute(0, 2, 3, 1)  # (B, H, W, C)
        features = features.reshape(batch_size, height * width, channels)  # (B, H*W, C)
        
        # Apply positional encoding
        features = self.pos_encoding(features)
        
        # Store intermediate outputs for deep supervision
        intermediate_features = []
        
        # Apply transformer blocks
        x = features
        for block in self.transformer_blocks:
            x = block(x)
            intermediate_features.append(x)
        
        # Apply layer normalization
        x = self.norm(x)
        
        # Global pooling
        x = x.transpose(1, 2)  # (B, C, L)
        pooled = self.pool(x).squeeze(-1)  # (B, C)
        
        # Apply final classifier
        logits = self.classifier(pooled)  # (B, num_emotions)
        
        # Prepare outputs
        outputs = {
            'emotion_logits': logits
        }
        
        # Add intermediate logits
        if self.training:
            intermediate_logits = []
            for i, feat in enumerate(intermediate_features[:-1]):  # Skip the last one which is used for final
                # Pool and classify
                feat = feat.transpose(1, 2)  # (B, C, L)
                feat_pooled = self.pool(feat).squeeze(-1)  # (B, C)
                im_logits = self.intermediate_classifiers[i](feat_pooled)
                intermediate_logits.append(im_logits)
            
            outputs['intermediate_logits'] = intermediate_logits
        
        # Calculate loss if targets are provided
        if emotion_targets is not None:
            if self.training and apply_mixup:
                # Mixup loss for final prediction
                loss = self._mixup_loss(logits, target_a, target_b, lam)
                
                # Add intermediate losses if in training
                if 'intermediate_logits' in outputs:
                    intermediate_loss = 0
                    for im_logits in outputs['intermediate_logits']:
                        intermediate_loss += self._mixup_loss(im_logits, target_a, target_b, lam)
                    
                    # Weight intermediate loss (0.3) vs final loss (0.7)
                    intermediate_loss = intermediate_loss / len(outputs['intermediate_logits']) * 0.3
                    loss = loss * 0.7 + intermediate_loss
            else:
                # Standard loss
                loss = self.loss_fn(logits, emotion_targets)
                
                # Add intermediate losses if in training
                if 'intermediate_logits' in outputs:
                    intermediate_loss = 0
                    for im_logits in outputs['intermediate_logits']:
                        intermediate_loss += self.loss_fn(im_logits, emotion_targets)
                    
                    # Weight intermediate loss (0.3) vs final loss (0.7)
                    intermediate_loss = intermediate_loss / len(outputs['intermediate_logits']) * 0.3
                    loss = loss * 0.7 + intermediate_loss
            
            outputs['loss'] = loss
        
        return outputs

    def predict_emotion(self, waveform):
        """
        Predict emotion from waveform
        
        Args:
            waveform: Tensor of shape (batch_size, 1, time)
            
        Returns:
            Tuple of (predicted_emotion_id, confidence)
        """
        # Set to evaluation mode
        self.eval()
        
        # Make prediction
        with torch.no_grad():
            outputs = self(waveform, apply_mixup=False, apply_augmentation=False)
            probs = F.softmax(outputs['emotion_logits'], dim=1)
            
            # Get predicted emotion
            predicted_id = torch.argmax(probs, dim=1).item()
            confidence = probs[0, predicted_id].item()
        
        return predicted_id, confidence
    
    @classmethod
    def from_pretrained(cls, path, map_location=None):
        """
        Load a pretrained model from checkpoint
        
        Args:
            path: Path to checkpoint
            map_location: Optional map_location for torch.load
            
        Returns:
            Loaded model
        """
        # Load checkpoint
        checkpoint = torch.load(path, map_location=map_location)
        
        # Get parameters
        params = checkpoint.get('params', {})
        
        # Create model
        model = cls(**params)
        
        # Load state dict
        if 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        return model


def create_advanced_emotion_recognizer(pretrained_path=None, device='cpu'):
    """
    Create an advanced emotion recognizer
    
    Args:
        pretrained_path: Optional path to pretrained model
        device: Device to use
        
    Returns:
        Emotion recognizer model
    """
    if pretrained_path:
        # Load pretrained model
        model = AdvancedEmotionRecognitionModel.from_pretrained(pretrained_path, map_location=device)
    else:
        # Create new model
        model = AdvancedEmotionRecognitionModel(
            num_emotions=8,
            feature_dim=256,
            hidden_dim=512,
            transformer_layers=4,
            transformer_heads=8,
            dropout=0.1
        )
    
    # Move to device
    model = model.to(device)
    
    return model


if __name__ == "__main__":
    # Create model
    model = create_advanced_emotion_recognizer()
    
    # Print model summary
    print(model)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Test with random input
    x = torch.randn(2, 1, 16000)
    outputs = model(x)
    
    print(f"\nInput shape: {x.shape}")
    print(f"Output logits shape: {outputs['emotion_logits'].shape}")
    print(f"Output probabilities shape: {F.softmax(outputs['emotion_logits'], dim=1).shape}") 