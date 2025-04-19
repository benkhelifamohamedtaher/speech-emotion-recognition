#!/usr/bin/env python3
"""
Advanced Speech Emotion Recognition Model
This module defines a state-of-the-art model architecture for speech emotion recognition,
specifically optimized for the RAVDESS dataset.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import logging
import numpy as np
from collections import deque
import warnings
import math
import torchaudio
from transformers import Wav2Vec2Model, Wav2Vec2Config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# RAVDESS emotion categories (matching the original dataset)
RAVDESS_EMOTIONS = [
    'neutral',  # 01
    'calm',     # 02
    'happy',    # 03
    'sad',      # 04
    'angry',    # 05
    'fearful',  # 06
    'disgust',  # 07
    'surprised' # 08
]

# Mapping between RAVDESS numerical IDs and emotion labels
RAVDESS_ID_TO_EMOTION = {
    "01": "neutral",
    "02": "calm",
    "03": "happy", 
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised"
}

# Mapping to simplified emotion set if needed
SIMPLIFIED_EMOTION_MAP = {
    "neutral": "neutral",
    "calm": "neutral",
    "happy": "happy",
    "sad": "sad",
    "angry": "angry",
    "fearful": "fear",
    "disgust": "disgust",
    "surprised": "surprise"
}

class AttentionPooling(nn.Module):
    """
    Attention pooling module for aggregating frame-level features
    """
    def __init__(self, in_dim, hidden_dim=128):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
            nn.Softmax(dim=1)
        )
        
    def forward(self, x):
        """
        Args:
            x: Tensor of shape (batch_size, seq_len, in_dim)
            
        Returns:
            Tensor of shape (batch_size, in_dim)
        """
        # Calculate attention weights
        weights = self.attention(x)  # (batch_size, seq_len, 1)
        
        # Apply attention weights
        weighted_x = weights * x  # (batch_size, seq_len, in_dim)
        
        # Sum along the sequence dimension
        pooled_x = torch.sum(weighted_x, dim=1)  # (batch_size, in_dim)
        
        return pooled_x, weights


class MultiHeadAttentionPooling(nn.Module):
    """
    Multi-head attention pooling for capturing different aspects of emotion
    """
    def __init__(self, in_dim, num_heads=4, hidden_dim=128):
        super().__init__()
        self.num_heads = num_heads
        self.attention_heads = nn.ModuleList([
            AttentionPooling(in_dim, hidden_dim) for _ in range(num_heads)
        ])
        
        # Projection to combine heads
        self.projection = nn.Linear(in_dim * num_heads, in_dim)
        
    def forward(self, x):
        """
        Args:
            x: Tensor of shape (batch_size, seq_len, in_dim)
            
        Returns:
            Tensor of shape (batch_size, in_dim)
        """
        pooled_outputs = []
        attention_weights = []
        
        for head in self.attention_heads:
            pooled, weights = head(x)
            pooled_outputs.append(pooled)
            attention_weights.append(weights)
        
        # Concatenate the outputs from each head
        concat_output = torch.cat(pooled_outputs, dim=1)  # (batch_size, in_dim * num_heads)
        
        # Project back to the original dimension
        output = self.projection(concat_output)  # (batch_size, in_dim)
        
        return output, attention_weights


class ContextTransformer(nn.Module):
    """
    Context transformer for modeling temporal dependencies
    """
    def __init__(self, in_dim, num_heads=4, dim_feedforward=1024, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(in_dim, num_heads, dropout=dropout)
        self.norm1 = nn.LayerNorm(in_dim)
        self.ff = nn.Sequential(
            nn.Linear(in_dim, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, in_dim)
        )
        self.norm2 = nn.LayerNorm(in_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        """
        Args:
            x: Tensor of shape (batch_size, seq_len, in_dim)
            
        Returns:
            Tensor of shape (batch_size, seq_len, in_dim)
        """
        # Permute for multi-head attention (seq_len, batch_size, in_dim)
        x_perm = x.permute(1, 0, 2)
        
        # Self-attention block
        attn_output, _ = self.self_attn(x_perm, x_perm, x_perm)
        x_perm = x_perm + self.dropout(attn_output)
        x_perm = self.norm1(x_perm)
        
        # Feed-forward block
        ff_output = self.ff(x_perm)
        x_perm = x_perm + self.dropout(ff_output)
        x_perm = self.norm2(x_perm)
        
        # Permute back to (batch_size, seq_len, in_dim)
        return x_perm.permute(1, 0, 2)


class SpeechFeatureExtractor(nn.Module):
    """
    Advanced speech feature extractor using Wav2Vec2
    """
    def __init__(self, wav2vec_model_name="facebook/wav2vec2-base", freeze_feature_extractor=True):
        super().__init__()
        
        # Load the pre-trained model
        try:
            self.wav2vec = Wav2Vec2Model.from_pretrained(wav2vec_model_name)
        except Exception as e:
            print(f"Warning: Could not load pretrained model: {e}")
            print("Using base configuration instead")
            config = Wav2Vec2Config()
            self.wav2vec = Wav2Vec2Model(config)
        
        # Freeze the feature extractor (convolutional layers)
        if freeze_feature_extractor:
            for param in self.wav2vec.feature_extractor.parameters():
                param.requires_grad = False
                
        self.out_dim = self.wav2vec.config.hidden_size
    
    def forward(self, x):
        """
        Args:
            x: Tensor of shape (batch_size, 1, seq_len)
            
        Returns:
            Tensor of shape (batch_size, new_seq_len, hidden_size)
        """
        # Remove channel dim for wav2vec
        x = x.squeeze(1)
        
        # Forward pass through wav2vec model
        with torch.no_grad() if all(p.requires_grad == False for p in self.wav2vec.parameters()) else torch.enable_grad():
            outputs = self.wav2vec(x, output_hidden_states=True)
        
        # Get the hidden states from the transformer
        last_hidden = outputs.last_hidden_state  # (batch_size, new_seq_len, hidden_size)
        
        return last_hidden


class EmotionClassifier(nn.Module):
    """
    Emotion classifier using frame-level features
    """
    def __init__(self, input_dim, hidden_dim=256, dropout_rate=0.3, num_classes=8):
        super().__init__()
        
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.LeakyReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 2, num_classes)
        )
    
    def forward(self, x):
        """
        Args:
            x: Tensor of shape (batch_size, input_dim)
            
        Returns:
            Tensor of shape (batch_size, num_classes)
        """
        return self.classifier(x)


class GenderClassifier(nn.Module):
    """
    Gender classifier as an auxiliary task
    """
    def __init__(self, input_dim, hidden_dim=128, dropout_rate=0.3):
        super().__init__()
        
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, 2)  # Binary classification: male/female
        )
    
    def forward(self, x):
        """
        Args:
            x: Tensor of shape (batch_size, input_dim)
            
        Returns:
            Tensor of shape (batch_size, 2)
        """
        return self.classifier(x)


class AdvancedSpeechEmotionRecognizer(nn.Module):
    """
    Advanced Speech Emotion Recognition model with multi-level attention and context modeling
    """
    def __init__(self, 
                 num_emotions=8,
                 wav2vec_model_name="facebook/wav2vec2-base",
                 freeze_feature_extractor=True,
                 context_layers=2,
                 attention_heads=4,
                 dropout_rate=0.3,
                 use_gender_branch=True,
                 use_spectrogram_branch=True):
        super().__init__()
        
        # Feature extractor
        self.feature_extractor = SpeechFeatureExtractor(
            wav2vec_model_name=wav2vec_model_name,
            freeze_feature_extractor=freeze_feature_extractor
        )
        feature_dim = self.feature_extractor.out_dim
        
        # Additional spectrogram branch
        self.use_spectrogram_branch = use_spectrogram_branch
        if use_spectrogram_branch:
            self.mel_spec = torchaudio.transforms.MelSpectrogram(
                sample_rate=16000,
                n_fft=1024,
                hop_length=512,
                n_mels=128
            )
            self.spec_conv = nn.Sequential(
                nn.Conv2d(1, 16, kernel_size=3, padding=1),
                nn.BatchNorm2d(16),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(16, 32, kernel_size=3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.MaxPool2d(2)
            )
            self.spec_pool = nn.AdaptiveAvgPool2d((1, 1))
            spec_dim = 64
            combined_dim = feature_dim + spec_dim
        else:
            combined_dim = feature_dim
        
        # Context transformer layers for temporal modeling
        self.context_layers = nn.ModuleList([
            ContextTransformer(feature_dim, attention_heads, feature_dim * 4, dropout_rate)
            for _ in range(context_layers)
        ])
        
        # Multi-head attention pooling
        self.attention_pooling = MultiHeadAttentionPooling(feature_dim, attention_heads)
        
        # Emotion classification branch
        self.emotion_classifier = EmotionClassifier(
            combined_dim, combined_dim * 2, dropout_rate, num_emotions
        )
        
        # Optional gender classification branch (for multi-task learning)
        self.use_gender_branch = use_gender_branch
        if use_gender_branch:
            self.gender_classifier = GenderClassifier(combined_dim, combined_dim // 2, dropout_rate)
    
    def forward(self, waveform, extract_features=False):
        """
        Args:
            waveform: Tensor of shape (batch_size, 1, seq_len)
            extract_features: If True, return intermediate features
            
        Returns:
            dict containing emotion_logits and optionally gender_logits
        """
        # Get frame-level features
        features = self.feature_extractor(waveform)  # (batch_size, seq_len, feature_dim)
        
        # Apply context transformer layers
        context_features = features
        for layer in self.context_layers:
            context_features = layer(context_features)
        
        # Apply attention pooling
        pooled_features, attention_weights = self.attention_pooling(context_features)
        
        # Spectrogram branch (optional)
        if self.use_spectrogram_branch:
            # Compute mel spectrogram
            with torch.no_grad():
                # Remove channel dim for mel_spec
                wave_for_spec = waveform.squeeze(1)
                # Compute spectrogram
                spec = self.mel_spec(wave_for_spec)  # (batch_size, n_mels, time)
                # Add channel dimension
                spec = spec.unsqueeze(1)  # (batch_size, 1, n_mels, time)
                # Apply log scaling
                spec = torch.log(spec + 1e-9)
            
            # Apply convolutional layers
            spec_features = self.spec_conv(spec)
            # Global pooling
            spec_features = self.spec_pool(spec_features)
            # Flatten
            spec_features = spec_features.view(spec_features.size(0), -1)
            
            # Concatenate with pooled wav2vec features
            combined_features = torch.cat([pooled_features, spec_features], dim=1)
        else:
            combined_features = pooled_features
        
        # Emotion classification
        emotion_logits = self.emotion_classifier(combined_features)
        
        # Gender classification (optional)
        if self.use_gender_branch:
            gender_logits = self.gender_classifier(combined_features)
            output = {
                'emotion_logits': emotion_logits,
                'gender_logits': gender_logits
            }
        else:
            output = {
                'emotion_logits': emotion_logits
            }
        
        # Return intermediate features if requested
        if extract_features:
            output['features'] = pooled_features
            output['attention_weights'] = attention_weights
        
        return output
    
    def predict_emotion(self, waveform):
        """
        Make emotion prediction from waveform
        
        Args:
            waveform: Tensor of shape (batch_size, 1, seq_len)
            
        Returns:
            Predicted emotion indices and probabilities
        """
        outputs = self.forward(waveform)
        emotion_logits = outputs['emotion_logits']
        emotion_probs = F.softmax(emotion_logits, dim=1)
        emotion_preds = torch.argmax(emotion_probs, dim=1)
        
        return emotion_preds, emotion_probs


class RAVDESSEmotionModel(nn.Module):
    """Model optimized for RAVDESS dataset emotion recognition"""
    
    def __init__(self, num_emotions=8, use_simplified=False):
        super(RAVDESSEmotionModel, self).__init__()
        self.use_simplified = use_simplified
        
        # Actual emotions in RAVDESS: neutral, calm, happy, sad, angry, fearful, disgust, surprised
        # Simplified emotions: neutral, happy, sad, angry
        self.num_emotions = 4 if use_simplified else num_emotions
        
        # Feature extraction
        self.conv1 = nn.Conv1d(1, 64, kernel_size=8, stride=2, padding=2)
        self.bn1 = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=6, stride=2, padding=2)
        self.bn2 = nn.BatchNorm1d(128)
        self.conv3 = nn.Conv1d(128, 256, kernel_size=4, stride=2, padding=2)
        self.bn3 = nn.BatchNorm1d(256)
        
        # Recurrent layers for temporal modeling
        self.lstm = nn.LSTM(256, 128, num_layers=2, batch_first=True, bidirectional=True)
        
        # Attention mechanism
        self.attention = RAVDESSAttention(256)  # bidirectional LSTM output size
        
        # Emotion classification
        self.dropout = nn.Dropout(0.3)
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, self.num_emotions)
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize model weights for better convergence"""
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input tensor of shape (batch_size, sequence_length)
                or (batch_size, 1, sequence_length)
        
        Returns:
            emotion_probs: Emotion probabilities
        """
        # Handle different input dimensions
        if x.dim() == 2:
            # Add channel dimension if it's missing
            x = x.unsqueeze(1)
        elif x.dim() == 3 and x.size(2) == 1:
            # Transpose if the channel dimension is last
            x = x.transpose(1, 2)
        
        # Log shape for debugging
        batch_size = x.size(0)
        logger.debug(f"Input shape: {x.shape}")
        
        # Apply feature extraction
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        
        # Reshape for LSTM: (batch, channels, seq_len) -> (batch, seq_len, channels)
        x = x.transpose(1, 2)
        
        # Apply LSTM
        try:
            lstm_out, _ = self.lstm(x)
        except Exception as e:
            logger.error(f"LSTM error with input shape {x.shape}: {e}")
            # If the sequence is too short, pad it
            if x.size(1) < 2:
                x = F.pad(x, (0, 0, 0, 2 - x.size(1), 0, 0))
                logger.info(f"Padded input to shape {x.shape}")
                lstm_out, _ = self.lstm(x)
            else:
                raise e
        
        # Apply attention
        context, _ = self.attention(lstm_out)
        
        # Emotion classification
        x = F.relu(self.fc1(self.dropout(context)))
        emotion_logits = self.fc2(x)
        emotion_probs = F.softmax(emotion_logits, dim=1)
        
        return emotion_probs
    
    def map_emotions(self, emotion_idx):
        """Map emotion index to emotion name based on RAVDESS convention"""
        if self.use_simplified:
            # Simplified mapping (4 emotions)
            emotions = {
                0: "neutral",
                1: "happy",
                2: "sad",
                3: "angry"
            }
        else:
            # Full RAVDESS mapping (8 emotions)
            emotions = {
                0: "neutral",
                1: "calm",
                2: "happy",
                3: "sad",
                4: "angry",
                5: "fearful",
                6: "disgust",
                7: "surprised"
            }
        
        return emotions.get(emotion_idx, "unknown")
    
    def ravdess_idx_to_simplified(self, ravdess_idx):
        """Map RAVDESS emotion index to simplified emotion index"""
        # RAVDESS emotion indices: 01=neutral, 02=calm, 03=happy, 04=sad, 05=angry, 06=fearful, 07=disgust, 08=surprised
        # Simplified indices: 0=neutral, 1=happy, 2=sad, 3=angry
        mapping = {
            1: 0,  # neutral -> neutral
            2: 0,  # calm -> neutral (closest match)
            3: 1,  # happy -> happy
            4: 2,  # sad -> sad
            5: 3,  # angry -> angry
            6: 2,  # fearful -> sad (closest match)
            7: 3,  # disgust -> angry (closest match)
            8: 1   # surprised -> happy (closest match)
        }
        return mapping.get(ravdess_idx, 0)
    
    def load_state_dict_with_dimension_fix(self, state_dict):
        """Load state dict with dimension handling for compatibility with different versions"""
        model_state_dict = self.state_dict()
        
        # Check for dimension mismatches in fc2 layer (most common issue when switching between full/simplified)
        if 'fc2.weight' in state_dict and state_dict['fc2.weight'].size(0) != model_state_dict['fc2.weight'].size(0):
            old_emotions = state_dict['fc2.weight'].size(0)
            new_emotions = model_state_dict['fc2.weight'].size(0)
            
            logger.warning(f"Model was trained with {old_emotions} emotions but trying to use with {new_emotions} emotions")
            
            if old_emotions > new_emotions:
                # Keep only the relevant emotions (e.g., simplified subset)
                logger.info("Adapting model from full emotions to simplified emotions")
                
                # For simplified, keep neutral (0), happy (2), sad (3), and angry (4)
                indices_to_keep = [0, 2, 3, 4]
                
                # Adjust fc2 weights and bias
                state_dict['fc2.weight'] = state_dict['fc2.weight'][indices_to_keep]
                state_dict['fc2.bias'] = state_dict['fc2.bias'][indices_to_keep]
            else:
                # This would require expanding the model - not supported
                logger.error("Cannot expand model dimensions from simplified to full emotions")
                raise ValueError("Cannot load a simplified model for full emotion recognition")
        
        # Load the possibly modified state dict
        super().load_state_dict(state_dict, strict=False)
        
        # Log which keys couldn't be loaded
        model_keys = set(model_state_dict.keys())
        loaded_keys = set(state_dict.keys())
        missing_keys = model_keys - loaded_keys
        
        if missing_keys:
            logger.warning(f"Missing keys when loading model: {missing_keys}")
    
    @classmethod
    def load_from_checkpoint(cls, checkpoint_path, map_location=None, use_simplified=False):
        """Load model from checkpoint with proper dimension handling"""
        try:
            checkpoint = torch.load(checkpoint_path, map_location=map_location)
            
            # Create model instance
            model = cls(use_simplified=use_simplified)
            
            # Load state dict with dimension handling
            if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                model.load_state_dict_with_dimension_fix(checkpoint['state_dict'])
            else:
                model.load_state_dict_with_dimension_fix(checkpoint)
            
            logger.info(f"Successfully loaded model from {checkpoint_path}")
            return model
        
        except Exception as e:
            logger.error(f"Error loading model from {checkpoint_path}: {e}")
            raise


# Pre-trained feature extractor for RAVDESS audio
class RAVDESSFeatureExtractor:
    """Extract features from audio for RAVDESS model"""
    
    def __init__(self, sample_rate=16000):
        self.sample_rate = sample_rate
    
    def extract_features(self, waveform, sample_rate=None):
        """
        Extract features from waveform
        
        Args:
            waveform: Audio waveform as torch tensor
            sample_rate: Sample rate of the audio
            
        Returns:
            features: Processed features ready for model
        """
        # Get sample rate
        sr = sample_rate if sample_rate is not None else self.sample_rate
        
        # Ensure waveform is a torch tensor
        if not isinstance(waveform, torch.Tensor):
            waveform = torch.tensor(waveform, dtype=torch.float32)
        
        # Ensure tensor is 2D (batch_size, audio_length)
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        
        # Normalize waveform
        waveform = waveform / (torch.max(torch.abs(waveform)) + 1e-8)
        
        return waveform


class RAVDESSRecognizer:
    """
    Real-time speech emotion recognition for RAVDESS dataset
    """
    def __init__(self, model_path, device='cpu', use_simplified_emotions=False):
        self.device = device
        
        # Create model
        self.model = RAVDESSEmotionModel(
            num_emotions=8,
            use_simplified=use_simplified_emotions
        )
        
        # Load weights
        if os.path.exists(model_path):
            try:
                logger.info(f"Loading model from {model_path}")
                checkpoint = torch.load(model_path, map_location=device)
                
                # Handle different checkpoint formats
                if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                    self.model.load_state_dict(checkpoint['model_state_dict'])
                else:
                    self.model.load_state_dict(checkpoint)
                    
                logger.info("Model loaded successfully")
            except Exception as e:
                logger.error(f"Error loading model: {e}")
                logger.warning("Using untrained model")
        else:
            logger.warning(f"Model file not found: {model_path}")
            logger.warning("Using untrained model")
        
        # Move model to device and set to evaluation mode
        self.model.to(device)
        self.model.eval()
        
        # Prediction smoothing with history
        self.emotion_history = deque(maxlen=5)
    
    def predict(self, waveform):
        """Predict emotion from audio waveform"""
        # Convert to PyTorch tensor if needed
        if not isinstance(waveform, torch.Tensor):
            waveform = torch.tensor(waveform, dtype=torch.float32)
        
        # Add batch dimension if needed
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        
        # Move to device
        waveform = waveform.to(self.device)
        
        # Get prediction
        with torch.no_grad():
            try:
                emotion_probs = self.model(waveform)
                
                # Extract results
                emotion_probs = emotion_probs.squeeze().cpu().numpy()
                
                # Update history for smoothing
                self.emotion_history.append(emotion_probs)
                
                # Get smoothed prediction
                smoothed_probs = np.mean(self.emotion_history, axis=0)
                
                # Get emotion label and confidence
                emotion, confidence = RAVDESSEmotionModel.get_emotion_label(
                    torch.from_numpy(smoothed_probs)
                )
                
                return {
                    'emotion': emotion,
                    'confidence': confidence,
                    'probs': smoothed_probs
                }
                
            except Exception as e:
                logger.error(f"Error during prediction: {e}")
                return {
                    'emotion': 'error',
                    'confidence': 0.0,
                    'probs': np.zeros(self.model.num_emotions)
                }


def train_ravdess_model(model, train_loader, val_loader, epochs=30, 
                       learning_rate=1e-4, device='cpu', output_dir='./models/ravdess'):
    """Train the RAVDESS emotion recognition model"""
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Move model to device
    model.to(device)
    
    # Set up optimizer and loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    criterion = nn.CrossEntropyLoss()
    
    # Best model tracking
    best_val_loss = float('inf')
    
    # Training loop
    for epoch in range(1, epochs + 1):
        # Training phase
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, batch in enumerate(train_loader):
            # Get data
            waveforms, emotions = batch['waveform'].to(device), batch['emotion'].to(device)
            
            # Forward pass
            optimizer.zero_grad()
            emotion_probs = model(waveforms)
            
            # Compute loss
            loss = criterion(emotion_probs, emotions)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Update statistics
            train_loss += loss.item()
            _, predicted = emotion_probs.max(1)
            total += emotions.size(0)
            correct += predicted.eq(emotions).sum().item()
            
        train_loss /= len(train_loader)
        train_acc = correct / total
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader):
                # Get data
                waveforms, emotions = batch['waveform'].to(device), batch['emotion'].to(device)
                
                # Forward pass
                emotion_probs = model(waveforms)
                
                # Compute loss
                loss = criterion(emotion_probs, emotions)
                
                # Update statistics
                val_loss += loss.item()
                _, predicted = emotion_probs.max(1)
                total += emotions.size(0)
                correct += predicted.eq(emotions).sum().item()
        
        val_loss /= len(val_loader)
        val_acc = correct / total
        
        # Print progress
        logger.info(f"Epoch {epoch}/{epochs} - "
                   f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                   f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_acc': val_acc
            }, os.path.join(output_dir, 'best_model.pt'))
            logger.info(f"Saved best model with val loss: {val_loss:.4f}")
    
    # Save final model
    torch.save({
        'epoch': epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': val_loss,
        'val_acc': val_acc
    }, os.path.join(output_dir, 'final_model.pt'))
    
    return model


def test_model():
    """Simple test function to verify the model implementation"""
    # Create a model
    model = RAVDESSEmotionModel(num_emotions=8)
    
    # Generate a random waveform (1 second at 16kHz)
    waveform = torch.randn(1, 16000)
    
    # Forward pass
    with torch.no_grad():
        emotion_probs = model(waveform)
    
    # Print results
    print(f"Emotion probabilities shape: {emotion_probs.shape}")
    
    # Test the recognizer
    print("\nTesting recognizer...")
    recognizer = RAVDESSRecognizer(model_path="nonexistent_path.pt")
    result = recognizer.predict(waveform)
    
    print(f"Emotion: {result['emotion']}")
    print(f"Confidence: {result['confidence']:.4f}")
    print(f"Probs: {result['probs']}")
    

if __name__ == "__main__":
    test_model() 