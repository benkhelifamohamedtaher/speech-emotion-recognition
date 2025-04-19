#!/usr/bin/env python3
"""
Fixed SimpleModel compatible with existing model weights.
This provides a model definition that matches the structure of the trained models.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleEmotionRecognitionModel(nn.Module):
    """Simple speech emotion recognition model compatible with saved weights"""
    def __init__(self, num_emotions=7):
        super().__init__()
        
        # Convolutional feature extraction layers
        self.conv1 = nn.Conv1d(1, 64, kernel_size=8, stride=2, padding=4)
        self.bn1 = nn.BatchNorm1d(64)
        self.pool1 = nn.MaxPool1d(kernel_size=4, stride=4)
        
        self.conv2 = nn.Conv1d(64, 128, kernel_size=8, stride=2, padding=4)
        self.bn2 = nn.BatchNorm1d(128)
        self.pool2 = nn.MaxPool1d(kernel_size=4, stride=4)
        
        self.conv3 = nn.Conv1d(128, 256, kernel_size=8, stride=2, padding=4)
        self.bn3 = nn.BatchNorm1d(256)
        self.pool3 = nn.AdaptiveAvgPool1d(1)
        
        # Classifier
        self.fc1 = nn.Linear(256, 128)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 64)
        self.dropout2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(64, num_emotions)
        
        # Voice activity detection
        self.vad_fc1 = nn.Linear(256, 64)
        self.vad_dropout = nn.Dropout(0.3)
        self.vad_fc2 = nn.Linear(64, 1)
    
    def forward(self, x):
        """Forward pass with shape handling for different input formats"""
        # Handle input shape
        if x.dim() == 2:  # [batch_size, seq_len]
            x = x.unsqueeze(1)  # Add channel dimension [batch_size, 1, seq_len]
        elif x.dim() == 3 and x.shape[1] != 1:
            x = x.transpose(1, 2)  # Transpose to [batch_size, 1, seq_len]
        elif x.dim() == 4:  # Unexpected dimension
            x = x.squeeze(2)  # Try to fix by removing an extra dimension
        
        # Apply convolutional layers
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        
        # Flatten features
        features = x.view(x.size(0), -1)
        
        # Emotion classification branch
        emotion = F.relu(self.fc1(features))
        emotion = self.dropout1(emotion)
        emotion = F.relu(self.fc2(emotion))
        emotion = self.dropout2(emotion)
        emotion_logits = self.fc3(emotion)
        emotion_probs = F.softmax(emotion_logits, dim=1)
        
        # Voice activity detection branch
        vad = F.relu(self.vad_fc1(features))
        vad = self.vad_dropout(vad)
        vad_logits = self.vad_fc2(vad)
        vad_probs = torch.sigmoid(vad_logits)
        
        return emotion_probs, vad_probs

def load_model(model_path, device='cpu'):
    """Load a saved model from path with proper handling"""
    model = SimpleEmotionRecognitionModel()
    
    try:
        # Load weights
        checkpoint = torch.load(model_path, map_location=device)
        
        # Handle different checkpoint formats
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
            
        print(f"Model loaded successfully from {model_path}")
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Using untrained model")
    
    model.to(device)
    model.eval()
    return model
    
# Simple testing function
def test_model():
    """Test the model with random input"""
    model = SimpleEmotionRecognitionModel()
    
    # Create random input
    x = torch.randn(2, 16000)  # Batch of 2, 1-second audio at 16kHz
    
    # Run model
    with torch.no_grad():
        emotion_probs, vad_probs = model(x)
    
    print(f"Emotion probs shape: {emotion_probs.shape}")
    print(f"VAD probs shape: {vad_probs.shape}")
    
if __name__ == "__main__":
    # Run simple test
    test_model() 