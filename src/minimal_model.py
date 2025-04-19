#!/usr/bin/env python3
"""
Minimal model compatible with the saved weights in models/simple_model/best_model.pt
This model structure matches the key names in the state dictionary.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class MinimalEmotionModel(nn.Module):
    """Emotion recognition model matching the saved model weights"""
    def __init__(self, num_emotions=7):
        super().__init__()
        
        # Feature encoder with keys matching saved weights
        self.encoder = nn.Module()
        self.encoder.encoder = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=8, stride=2, padding=4),  # encoder.encoder.0
            nn.BatchNorm1d(64),  # encoder.encoder.1
            nn.ReLU(),  # encoder.encoder.2 (no weights)
            nn.Conv1d(64, 128, kernel_size=8, stride=2, padding=4),  # encoder.encoder.3
            nn.BatchNorm1d(128),  # encoder.encoder.4
            nn.ReLU(),  # encoder.encoder.5 (no weights)
            nn.Conv1d(128, 256, kernel_size=8, stride=2, padding=4),  # encoder.encoder.6
            nn.BatchNorm1d(256),  # encoder.encoder.7
            nn.ReLU()  # encoder.encoder.8 (no weights)
        )
        
        # Pooling (no weights)
        self.adaptive_pool = nn.AdaptiveAvgPool1d(1)
        
        # Classifier layers - keys matching saved weights
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_emotions)
        
        # Voice activity detection branch
        self.vad_fc1 = nn.Linear(256, 64)
        self.vad_fc2 = nn.Linear(64, 1)
    
    def forward(self, x):
        """Forward pass with proper shape handling and branch structure"""
        # Handle tensor dimensions
        if x.dim() == 2:  # [batch_size, seq_len]
            x = x.unsqueeze(1)  # [batch_size, 1, seq_len]
        elif x.dim() == 3 and x.shape[1] != 1:
            x = x.transpose(1, 2)  # [batch_size, 1, seq_len]
        
        # Apply encoder
        x = self.encoder.encoder(x)
        
        # Global pooling
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)  # Flatten
        
        # Classification branch
        emotion = F.relu(self.fc1(x))
        emotion = F.dropout(emotion, p=0.5, training=self.training)
        emotion = F.relu(self.fc2(emotion))
        emotion = F.dropout(emotion, p=0.3, training=self.training)
        emotion_logits = self.fc3(emotion)
        emotion_probs = F.softmax(emotion_logits, dim=1)
        
        # VAD branch
        vad = F.relu(self.vad_fc1(x))
        vad = F.dropout(vad, p=0.3, training=self.training)
        vad_logits = self.vad_fc2(vad)
        vad_probs = torch.sigmoid(vad_logits)
        
        return emotion_probs, vad_probs

def load_model(model_path, device='cpu'):
    """Load model from saved weights path with proper error handling"""
    model = MinimalEmotionModel()
    
    try:
        # Load the state dict
        checkpoint = torch.load(model_path, map_location=device)
        
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
        
        # Load the state dict
        model.load_state_dict(state_dict)
        print(f"Successfully loaded model from {model_path}")
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Using untrained model")
    
    model.to(device)
    model.eval()
    return model

def test_model(model_path=None):
    """Test the model with some sample data"""
    # Create a model
    if model_path:
        model = load_model(model_path)
    else:
        model = MinimalEmotionModel()
        model.eval()
    
    # Create a sample input (1 second of audio at 16kHz)
    sample_input = torch.randn(1, 16000)
    
    # Run model inference
    with torch.no_grad():
        emotion_probs, vad_probs = model(sample_input)
    
    # Print results
    print(f"Emotion probabilities shape: {emotion_probs.shape}")
    print(f"Voice activity detection shape: {vad_probs.shape}")
    print("\nEmotion probabilities:")
    emotion_labels = ['neutral', 'happy', 'sad', 'angry', 'fear', 'disgust', 'surprise']
    for i, label in enumerate(emotion_labels):
        print(f"  {label}: {emotion_probs[0, i].item():.4f}")
    print(f"\nVoice activity: {vad_probs.item():.4f}")
    
    return model

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test Minimal Emotion Model")
    parser.add_argument("--model", type=str, help="Path to model weights")
    
    args = parser.parse_args()
    
    test_model(args.model) 