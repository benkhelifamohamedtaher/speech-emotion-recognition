import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Wav2Vec2Model, Wav2Vec2Config
from torch.nn import TransformerEncoder, TransformerEncoderLayer


class AttentionPooling1D(nn.Module):
    """
    Attention pooling layer for temporal data
    """
    def __init__(self, hidden_size=768):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
            nn.Softmax(dim=1)
        )
        
    def forward(self, x):
        # x shape: (batch_size, sequence_length, hidden_size)
        attention_weights = self.attention(x)  # (batch_size, sequence_length, 1)
        context_vector = torch.sum(attention_weights * x, dim=1)  # (batch_size, hidden_size)
        return context_vector


class SpeechEmotionRecognitionModel(nn.Module):
    """
    Speech Emotion Recognition model using Wav2Vec 2.0 as encoder
    with multi-task learning for emotion classification and VAD
    """
    def __init__(self, num_emotions=4, freeze_encoder=True):
        super().__init__()
        
        # Load pre-trained Wav2Vec 2.0 model
        self.encoder = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")
        
        # Freeze encoder parameters for transfer learning
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
        
        hidden_size = self.encoder.config.hidden_size
        
        # Emotion classification branch
        self.emotion_dense = nn.Linear(hidden_size, 128)
        self.emotion_attention = AttentionPooling1D(128)
        self.emotion_classifier = nn.Linear(128, num_emotions)
        
        # Voice Activity Detection (VAD) branch
        self.vad_dense = nn.Linear(hidden_size, 64)
        self.vad_classifier = nn.Linear(64, 1)
        
    def forward(self, input_values, attention_mask=None):
        # Get encoder outputs
        encoder_outputs = self.encoder(
            input_values, 
            attention_mask=attention_mask
        ).last_hidden_state  # (batch_size, sequence_length, hidden_size)
        
        # Emotion classification branch
        emotion_features = F.gelu(self.emotion_dense(encoder_outputs))
        emotion_context = self.emotion_attention(emotion_features)
        emotion_logits = self.emotion_classifier(emotion_context)
        emotion_probs = F.softmax(emotion_logits, dim=-1)
        
        # VAD branch
        vad_features = F.gelu(self.vad_dense(encoder_outputs))
        vad_logits = self.vad_classifier(vad_features)
        vad_probs = torch.sigmoid(vad_logits)
        
        return {
            "emotion_logits": emotion_logits,
            "emotion_probs": emotion_probs,
            "vad_logits": vad_logits,
            "vad_probs": vad_probs
        }


class RealTimeSpeechEmotionRecognizer:
    """
    Wrapper for real-time inference with the SER model
    """
    def __init__(self, model_path, device="cpu"):
        self.device = device
        self.model = SpeechEmotionRecognitionModel()
        self.model.load_state_dict(torch.load(model_path, map_location=device))
        self.model.to(device)
        self.model.eval()
        
        self.sample_rate = 16000
        self.emotions = ["angry", "happy", "sad", "neutral"]
        
    def preprocess_audio(self, waveform):
        """Preprocess audio waveform for model input"""
        # Convert to torch tensor if numpy array
        if not isinstance(waveform, torch.Tensor):
            waveform = torch.tensor(waveform, dtype=torch.float32)
            
        # Ensure correct dimensions (batch_size, sequence_length)
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
            
        # Normalize
        waveform = (waveform - waveform.mean()) / (waveform.std() + 1e-6)
        
        return waveform.to(self.device)
    
    def predict(self, waveform):
        """
        Predict emotion from audio waveform
        Args:
            waveform: Raw audio waveform (16kHz, mono)
        Returns:
            dict: Emotion probabilities and dominant emotion
        """
        with torch.no_grad():
            input_values = self.preprocess_audio(waveform)
            outputs = self.model(input_values)
            
            emotion_probs = outputs["emotion_probs"][0].cpu().numpy()
            vad_prob = outputs["vad_probs"][0].mean().cpu().numpy()
            
            # Only predict emotion if voice activity detected
            is_speech = vad_prob > 0.5
            
            # Get dominant emotion
            if is_speech:
                dominant_emotion_idx = emotion_probs.argmax().item()
                dominant_emotion = self.emotions[dominant_emotion_idx]
            else:
                dominant_emotion = "no_speech"
                
            return {
                "emotion_probs": emotion_probs,
                "vad_prob": vad_prob,
                "is_speech": is_speech,
                "dominant_emotion": dominant_emotion
            } 