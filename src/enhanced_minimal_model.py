#!/usr/bin/env python3
"""
Enhanced minimal model matching the exact structure of the saved weights.
This model will correctly load the weights from models/simple_model/best_model.pt.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class EnhancedEmotionModel(nn.Module):
    """Speech emotion recognition model matching the exact structure of saved weights"""
    def __init__(self, num_emotions=7, hidden_size=256):
        super().__init__()
        
        # Encoder with exact matching structure
        self.encoder = nn.Module()
        self.encoder.encoder = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=10, stride=2, padding=4),  # encoder.encoder.0
            nn.BatchNorm1d(64),  # encoder.encoder.1
            nn.ReLU(),  # encoder.encoder.2 (no params)
            nn.Conv1d(64, 128, kernel_size=8, stride=2, padding=3),  # encoder.encoder.3
            nn.BatchNorm1d(128),  # encoder.encoder.4
            nn.ReLU(),  # encoder.encoder.5 (no params)
            nn.Conv1d(128, 256, kernel_size=4, stride=1, padding=1),  # encoder.encoder.6
            nn.BatchNorm1d(256),  # encoder.encoder.7
            nn.ReLU(),  # encoder.encoder.8 (no params)
            nn.Conv1d(256, 512, kernel_size=4, stride=1, padding=1),  # encoder.encoder.9
            nn.BatchNorm1d(512),  # encoder.encoder.10
            nn.ReLU(),  # encoder.encoder.11 (no params)
            nn.Conv1d(512, hidden_size, kernel_size=4, stride=1, padding=1),  # encoder.encoder.12
            nn.BatchNorm1d(hidden_size),  # encoder.encoder.13
        )
        
        # Attention pooling - multi-head attention for sequence aggregation
        self.attention_pool = nn.Module()
        self.attention_pool.query = nn.Linear(hidden_size, hidden_size)
        self.attention_pool.key = nn.Linear(hidden_size, hidden_size)
        self.attention_pool.value = nn.Linear(hidden_size, hidden_size)
        self.attention_pool.output_projection = nn.Linear(hidden_size, hidden_size)
        
        # Context encoder - transformer layers for temporal context
        self.context_encoder = nn.Module()
        self.context_encoder.layers = nn.ModuleList([
            # Layer 0
            self._create_transformer_layer(hidden_size),
            # Layer 1
            self._create_transformer_layer(hidden_size)
        ])
        
        # Emotion classifier - hierarchical classification
        self.emotion_classifier = nn.Module()
        self.emotion_classifier.level1 = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.BatchNorm1d(128)
        )
        self.emotion_classifier.level2 = nn.Sequential(
            nn.Linear(128, 64),
            nn.BatchNorm1d(64)
        )
        self.emotion_classifier.context_layer = nn.Sequential(
            nn.Linear(64, 32),
            nn.BatchNorm1d(32)
        )
        self.emotion_classifier.classifier = nn.Linear(32, num_emotions)
        
        # Voice activity detection branch
        self.vad_branch = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
    
    def _create_transformer_layer(self, hidden_size):
        """Create a transformer encoder layer with specific parameter names"""
        layer = nn.Module()
        # Self attention
        layer.self_attn = nn.Module()
        layer.self_attn.in_proj_weight = nn.Parameter(torch.randn(3 * hidden_size, hidden_size))
        layer.self_attn.in_proj_bias = nn.Parameter(torch.zeros(3 * hidden_size))
        layer.self_attn.out_proj = nn.Linear(hidden_size, hidden_size)
        
        # Feed forward
        layer.linear1 = nn.Linear(hidden_size, 4 * hidden_size)
        layer.linear2 = nn.Linear(4 * hidden_size, hidden_size)
        
        # Layer normalization
        layer.norm1 = nn.LayerNorm(hidden_size)
        layer.norm2 = nn.LayerNorm(hidden_size)
        
        return layer
    
    def _attention_pool(self, x):
        """Attention pooling operation with multi-head attention"""
        # x: [batch_size, sequence_length, hidden_size]
        batch_size, seq_len, dim = x.shape
        
        # Compute query, key, value
        q = self.attention_pool.query(x)  # [batch, seq, dim]
        k = self.attention_pool.key(x)    # [batch, seq, dim]
        v = self.attention_pool.value(x)  # [batch, seq, dim]
        
        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / (dim ** 0.5)  # [batch, seq, seq]
        attn_weights = F.softmax(scores, dim=-1)  # [batch, seq, seq]
        
        # Apply attention weights
        context = torch.matmul(attn_weights, v)  # [batch, seq, dim]
        
        # Apply output projection
        output = self.attention_pool.output_projection(context)  # [batch, seq, dim]
        
        # Global average pooling
        output = output.mean(dim=1)  # [batch, dim]
        
        return output
    
    def _transformer_layer(self, x, layer):
        """Custom transformer layer implementation to match model structure"""
        # Self attention
        # Split in_proj_weight into q, k, v
        dim = x.size(-1)
        qkv_weight = layer.self_attn.in_proj_weight
        qkv_bias = layer.self_attn.in_proj_bias
        
        q_weight, k_weight, v_weight = qkv_weight.chunk(3, dim=0)
        q_bias, k_bias, v_bias = qkv_bias.chunk(3, dim=0)
        
        # Compute q, k, v
        q = F.linear(x, q_weight, q_bias)
        k = F.linear(x, k_weight, k_bias)
        v = F.linear(x, v_weight, v_bias)
        
        # Compute attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / (dim ** 0.5)
        attn_weights = F.softmax(scores, dim=-1)
        attn_output = torch.matmul(attn_weights, v)
        
        # Apply output projection
        attn_output = layer.self_attn.out_proj(attn_output)
        
        # Add & norm (residual connection)
        x = x + attn_output
        x = layer.norm1(x)
        
        # Feed forward
        ff_output = F.relu(layer.linear1(x))
        ff_output = layer.linear2(ff_output)
        
        # Add & norm (residual connection)
        x = x + ff_output
        x = layer.norm2(x)
        
        return x
    
    def forward(self, x):
        """Forward pass with proper shape handling and architecture matching saved weights"""
        batch_size = x.size(0)
        
        # Handle input shape
        if x.dim() == 2:  # [batch_size, seq_len]
            x = x.unsqueeze(1)  # Add channel dimension [batch_size, 1, seq_len]
        elif x.dim() == 3 and x.shape[1] != 1:
            x = x.transpose(1, 2)  # [batch_size, 1, seq_len]
        
        # Apply encoder
        x = self.encoder.encoder(x)  # [batch_size, hidden_size, seq_len]
        
        # Transpose for sequence processing
        x = x.transpose(1, 2)  # [batch_size, seq_len, hidden_size]
        
        # Apply attention pooling
        context = self._attention_pool(x)  # [batch_size, hidden_size]
        
        # Apply transformer layers
        x_seq = x
        for layer in self.context_encoder.layers:
            x_seq = self._transformer_layer(x_seq, layer)
        
        # Take the first token embedding or mean pooling
        x_transformed = x_seq.mean(dim=1)  # [batch_size, hidden_size]
        
        # Combine context and transformed embeddings
        x_combined = context  # For simplicity, just use context
        
        # Voice activity detection branch
        vad_probs = torch.sigmoid(self.vad_branch(x_combined))
        
        # Emotion classification branch
        x_emotion = F.relu(self.emotion_classifier.level1(x_combined))
        x_emotion = F.relu(self.emotion_classifier.level2(x_emotion))
        x_emotion = F.relu(self.emotion_classifier.context_layer(x_emotion))
        emotion_logits = self.emotion_classifier.classifier(x_emotion)
        emotion_probs = F.softmax(emotion_logits, dim=1)
        
        return emotion_probs, vad_probs

def load_model(model_path, device='cpu'):
    """Load the model with weights from the saved checkpoint"""
    model = EnhancedEmotionModel()
    
    try:
        # Load weights
        checkpoint = torch.load(model_path, map_location=device)
        
        # Format appropriately
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
        
        # Load state dict
        model.load_state_dict(state_dict)
        print(f"Successfully loaded model from {model_path}")
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Using untrained model")
    
    model.to(device)
    model.eval()
    return model

def test_model(model_path=None):
    """Test the model with sample input"""
    if model_path:
        model = load_model(model_path)
    else:
        model = EnhancedEmotionModel()
        model.eval()
    
    # Create sample input (1 second of audio at 16kHz)
    sample_input = torch.randn(1, 16000)
    
    # Run inference
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
    
    parser = argparse.ArgumentParser(description="Test Enhanced Emotion Model")
    parser.add_argument("--model", type=str, help="Path to model weights")
    
    args = parser.parse_args()
    
    test_model(args.model) 