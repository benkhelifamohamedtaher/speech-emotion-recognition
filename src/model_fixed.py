#!/usr/bin/env python3
"""
Fixed Speech Emotion Recognition model
This model is designed to handle tensor dimension issues and properly
process audio inputs for emotion classification.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings
import numpy as np

# Try importing transformers, with fallback for handling errors
try:
    from transformers import Wav2Vec2Model, Wav2Vec2Config
except Exception as e:
    warnings.warn(f"Error importing transformers: {e}. Using fallback implementation.")
    
    # Fallback implementation
    class Wav2Vec2Config:
        def __init__(self, hidden_size=768):
            self.hidden_size = hidden_size

    class Wav2Vec2Model(nn.Module):
        def __init__(self, config=None):
            super().__init__()
            if config is None:
                self.config = Wav2Vec2Config()
            else:
                self.config = config
            
            # Simple convolutional encoder with proper dimension handling
            self.encoder = nn.Sequential(
                nn.Conv1d(1, 64, kernel_size=10, stride=5, padding=5),
                nn.BatchNorm1d(64),
                nn.GELU(),
                nn.Conv1d(64, 128, kernel_size=8, stride=4, padding=4),
                nn.BatchNorm1d(128),
                nn.GELU(),
                nn.Conv1d(128, 256, kernel_size=4, stride=2, padding=2),
                nn.BatchNorm1d(256),
                nn.GELU(),
                nn.Conv1d(256, 512, kernel_size=4, stride=2, padding=2),
                nn.BatchNorm1d(512),
                nn.GELU(),
                nn.Conv1d(512, 768, kernel_size=4, stride=2, padding=2),
                nn.BatchNorm1d(768),
                nn.GELU(),
            )
        
        def forward(self, input_values, attention_mask=None):
            # Ensure input has the right shape [batch_size, 1, sequence_length]
            if input_values.dim() == 2:
                # [batch_size, sequence_length] -> [batch_size, 1, sequence_length]
                x = input_values.unsqueeze(1)
            elif input_values.dim() == 3 and input_values.shape[1] == 1:
                # [batch_size, 1, sequence_length] - already correct
                x = input_values
            elif input_values.dim() == 3 and input_values.shape[1] != 1:
                # Handle case where channels dimension is in wrong position
                x = input_values.transpose(1, 2)
            elif input_values.dim() == 4:
                # [batch_size, 1, 1, sequence_length] - squeeze extra dimension
                x = input_values.squeeze(2)
                if x.dim() == 4:  # If squeezing didn't work (maybe multiple dims are 1)
                    x = input_values.view(input_values.shape[0], 1, -1)  # Force correct shape
            else:
                # Unexpected shape - reshape to ensure correct dimensions
                x = input_values.reshape(input_values.shape[0], 1, -1)
                warnings.warn(f"Unexpected input shape {input_values.shape}, reshaped to {x.shape}")
            
            # Apply encoder
            try:
                features = self.encoder(x)
            except RuntimeError as e:
                # If there's a dimension error, try to fix it
                if "Expected 3D" in str(e) or "Expected 2D" in str(e) or "Expected 4D" in str(e):
                    warnings.warn(f"Dimension error in encoder: {e}. Attempting to fix dimensions.")
                    # Try reshaping to ensure [batch_size, channels, length]
                    if x.dim() != 3:
                        x = x.view(x.shape[0], 1, -1)
                    features = self.encoder(x)
                else:
                    raise e
            
            # Convert to transformer-style output [batch_size, seq_len, hidden_size]
            features = features.transpose(1, 2)  # [batch_size, seq_len, channels]
            
            # Create a class to match transformer output format
            class DummyOutput:
                def __init__(self, last_hidden_state):
                    self.last_hidden_state = last_hidden_state
            
            return DummyOutput(features)
        
        @classmethod
        def from_pretrained(cls, model_name, *args, **kwargs):
            print(f"Using fallback model instead of {model_name}")
            return cls()


class FixedSpeechEmotionRecognitionModel(nn.Module):
    """
    Fixed Speech Emotion Recognition model that handles tensor dimension issues
    Uses Wav2Vec 2.0 as encoder with proper tensor handling
    """
    def __init__(self, num_emotions=7, model_name="facebook/wav2vec2-base-960h"):
        super().__init__()
        # Load encoder (with fallback if needed)
        try:
            self.encoder = Wav2Vec2Model.from_pretrained(model_name)
        except Exception as e:
            warnings.warn(f"Error loading Wav2Vec2 model: {e}. Using fallback encoder.")
            self.encoder = Wav2Vec2Model()
        
        hidden_size = self.encoder.config.hidden_size  # Typically 768
        
        # Temporal pooling with attention to handle variable length
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.Tanh(),
            nn.Linear(128, 1),
            nn.Softmax(dim=1)
        )
        
        # Emotion classification branch
        self.emotion_classifier = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_emotions)
        )
        
        # Voice Activity Detection (VAD) branch
        self.vad = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        """
        Forward pass through the model
        Args:
            x: Input audio waveform [batch_size, sequence_length] or [batch_size, 1, sequence_length]
        Returns:
            tuple: (emotion_probs, vad_probs)
        """
        # Ensure input has a consistent shape
        if x.dim() == 2:  # [batch_size, sequence_length]
            x = x
        elif x.dim() == 3 and x.shape[1] == 1:  # [batch_size, 1, sequence_length]
            x = x.squeeze(1)
        elif x.dim() == 4 and x.shape[1] == 1 and x.shape[2] == 1:  # [batch_size, 1, 1, sequence_length]
            x = x.squeeze(1).squeeze(1)
        elif x.dim() == 4:  # Other 4D shape
            x = x.view(x.shape[0], -1)  # Flatten to [batch_size, sequence_length]
        else:
            # Try to flatten to 2D
            x = x.view(x.shape[0], -1)
        
        # Get encoder outputs
        try:
            encoder_outputs = self.encoder(x).last_hidden_state  # [batch_size, seq_len, hidden_size]
        except Exception as e:
            # If there's an error, try more aggressive dimension fixing
            warnings.warn(f"Error in encoder forward pass: {e}. Attempting to fix dimensions.")
            try:
                # Try reshaping specifically for the encoder
                if x.dim() > 2:
                    x = x.reshape(x.shape[0], -1)  # Flatten to [batch_size, sequence]
                encoder_outputs = self.encoder(x).last_hidden_state
            except Exception as e2:
                warnings.warn(f"Second attempt failed: {e2}. Trying one more approach.")
                try:
                    # Last resort - add a dummy channel dimension
                    x_fixed = x.unsqueeze(1) if x.dim() == 2 else x
                    encoder_outputs = self.encoder(x_fixed).last_hidden_state
                except Exception as e3:
                    raise RuntimeError(f"All dimension fixing attempts failed: {e3}")
        
        # Apply attention for weighted temporal pooling
        try:
            attention_weights = self.attention(encoder_outputs)  # [batch_size, seq_len, 1]
            context = torch.sum(attention_weights * encoder_outputs, dim=1)  # [batch_size, hidden_size]
        except RuntimeError as e:
            # Handle possible dimension mismatch in attention mechanism
            warnings.warn(f"Error in attention mechanism: {e}. Using mean pooling instead.")
            context = torch.mean(encoder_outputs, dim=1)  # Simple mean pooling as fallback
        
        # Emotion classification
        emotion_logits = self.emotion_classifier(context)
        emotion_probs = F.softmax(emotion_logits, dim=-1)
        
        # Voice activity detection
        vad_probs = self.vad(context)
        
        return emotion_probs, vad_probs


class EnhancedFixedSpeechEmotionRecognitionModel(nn.Module):
    """
    Enhanced Speech Emotion Recognition model with robust handling of tensor dimensions
    Includes multi-level features and advanced pooling
    """
    def __init__(self, num_emotions=7, model_name="facebook/wav2vec2-base-960h"):
        super().__init__()
        # Load encoder (with fallback if needed)
        try:
            self.encoder = Wav2Vec2Model.from_pretrained(model_name)
        except Exception as e:
            warnings.warn(f"Error loading Wav2Vec2 model: {e}. Using fallback encoder.")
            self.encoder = Wav2Vec2Model()
        
        hidden_size = self.encoder.config.hidden_size  # Typically 768
        
        # Multi-head attention for temporal pooling
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )
        
        # Multi-level feature extraction
        self.conv_features = nn.Sequential(
            nn.Conv1d(hidden_size, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Conv1d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)  # Global pooling to get fixed-length representation
        )
        
        # Feature fusion
        self.fusion = nn.Sequential(
            nn.Linear(hidden_size + 128, 256),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # Emotion classification branch with deeper network
        self.emotion_classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_emotions)
        )
        
        # Voice Activity Detection (VAD) branch
        self.vad = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        # Arousal-Valence branch for dimensional emotion model
        self.arousal_valence = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 2)  # [arousal, valence]
        )
    
    def forward(self, x):
        """
        Forward pass through the enhanced model with tensor dimension handling
        Args:
            x: Input audio waveform [batch_size, sequence_length] or [batch_size, 1, sequence_length]
        Returns:
            dict: Dictionary containing emotion probabilities, VAD, and arousal-valence
        """
        # Ensure input is 2D (batch_size, sequence_length) or 3D (batch_size, 1, sequence_length)
        if x.dim() == 3 and x.shape[1] == 1:
            # If [batch_size, 1, seq_len], squeeze the middle dimension
            x = x.squeeze(1)
        elif x.dim() == 4:
            # If [batch_size, 1, 1, seq_len], remove the extra dimensions
            x = x.squeeze(1).squeeze(1)
        
        # Get encoder outputs with dimension error handling
        try:
            encoder_outputs = self.encoder(x).last_hidden_state  # [batch_size, seq_len, hidden_size]
        except Exception as e:
            # If there's an error, try with a manual dimension fix
            warnings.warn(f"Error in encoder forward pass: {e}. Attempting to fix dimensions.")
            if x.dim() == 2:
                # Add a channel dimension for the encoder
                x_fixed = x.unsqueeze(1)
                encoder_outputs = self.encoder(x_fixed).last_hidden_state
            else:
                raise e
        
        # Apply multi-head attention for global context
        # Create a query from averaged sequence
        query = encoder_outputs.mean(dim=1, keepdim=True)
        attn_output, _ = self.multihead_attn(query, encoder_outputs, encoder_outputs)
        attn_output = attn_output.squeeze(1)  # [batch_size, hidden_size]
        
        # Apply convolutional feature extraction
        # Transpose for conv layers [batch_size, hidden_size, seq_len]
        conv_input = encoder_outputs.transpose(1, 2)
        conv_output = self.conv_features(conv_input).squeeze(-1)  # [batch_size, 128]
        
        # Feature fusion
        fused_features = torch.cat([attn_output, conv_output], dim=1)
        fused_features = self.fusion(fused_features)
        
        # Emotion classification
        emotion_logits = self.emotion_classifier(fused_features)
        emotion_probs = F.softmax(emotion_logits, dim=-1)
        
        # Voice activity detection
        vad_probs = self.vad(fused_features)
        
        # Arousal-Valence prediction
        av_output = self.arousal_valence(fused_features)
        arousal = av_output[:, 0]
        valence = av_output[:, 1]
        
        return {
            "emotion_probs": emotion_probs,
            "vad_probs": vad_probs,
            "arousal": arousal,
            "valence": valence
        }


# Quick test to verify model works with different input dimensions
def test_model():
    # Create inputs with different dimensions
    batch_size = 2
    seq_len = 16000
    
    # Create models
    fixed_model = FixedSpeechEmotionRecognitionModel(num_emotions=7)
    enhanced_model = EnhancedFixedSpeechEmotionRecognitionModel(num_emotions=7)
    
    # Test with different input shapes
    shapes = [
        (batch_size, seq_len),                # 2D: [batch_size, seq_len]
        (batch_size, 1, seq_len),             # 3D: [batch_size, 1, seq_len]
        (batch_size, seq_len, 1),             # 3D: [batch_size, seq_len, 1] (wrong dim order)
        (batch_size, 1, 1, seq_len)           # 4D: [batch_size, 1, 1, seq_len]
    ]
    
    print("Testing model with different input shapes:")
    for shape in shapes:
        try:
            x = torch.randn(*shape)
            print(f"\nInput shape: {shape}")
            
            # Test fixed model
            outputs = fixed_model(x)
            if isinstance(outputs, tuple):
                emotion_probs, vad_probs = outputs
                print(f"Fixed model output: emotion_probs shape {emotion_probs.shape}, vad_probs shape {vad_probs.shape}")
            
            # Test enhanced model
            outputs = enhanced_model(x)
            if isinstance(outputs, dict):
                print(f"Enhanced model output shapes:")
                for key, value in outputs.items():
                    print(f"  {key}: {value.shape}")
            
        except Exception as e:
            print(f"Error with shape {shape}: {e}")


if __name__ == "__main__":
    # Run test to verify model works
    test_model() 