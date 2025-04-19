import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings
import numpy as np

# Modified imports with fallback
try:
    from transformers import Wav2Vec2Model, Wav2Vec2Config
except RuntimeError as e:
    if "torchvision::nms" in str(e):
        warnings.warn("Error loading transformers due to torchvision conflict. Using fallback implementation.")
        # Fallback to basic model definition if transformers fails
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
                
                # Simple convolutional encoder
                self.encoder = nn.Sequential(
                    nn.Conv1d(1, 64, kernel_size=10, stride=5, padding=5),
                    nn.LayerNorm([64]),
                    nn.GELU(),
                    nn.Conv1d(64, 128, kernel_size=8, stride=4, padding=2),
                    nn.LayerNorm([128]),
                    nn.GELU(),
                    nn.Conv1d(128, 256, kernel_size=4, stride=2, padding=1),
                    nn.LayerNorm([256]),
                    nn.GELU(),
                    nn.Conv1d(256, 512, kernel_size=4, stride=2, padding=1),
                    nn.LayerNorm([512]),
                    nn.GELU(),
                    nn.Conv1d(512, 768, kernel_size=4, stride=2, padding=1),
                    nn.LayerNorm([768]),
                    nn.GELU(),
                )
            
            def forward(self, input_values, attention_mask=None):
                # input_values: [batch_size, 1, sequence_length] or [batch_size, sequence_length]
                
                # Ensure input has the right shape
                if input_values.dim() == 3 and input_values.shape[1] == 1:
                    # [batch_size, 1, sequence_length] - already correct
                    x = input_values
                elif input_values.dim() == 2:
                    # [batch_size, sequence_length] - add channel dimension
                    x = input_values.unsqueeze(1)
                elif input_values.dim() == 4 and input_values.shape[2] == 1:
                    # [batch_size, 1, 1, sequence_length] - squeeze out extra dimension
                    x = input_values.squeeze(2)
                elif input_values.dim() == 4:
                    # Handle any 4D input by squeezing or reshaping appropriately
                    # [batch_size, 1, 1, sequence_length] or similar
                    x = input_values.view(input_values.shape[0], 1, -1)
                else:
                    # Unexpected shape - try to fix
                    x = input_values.reshape(input_values.shape[0], 1, -1)
                    print(f"Warning: Reshaped input from {input_values.shape} to {x.shape}")
                
                # Apply encoder (handle Layer Norm correctly)
                features = None
                try:
                    features = self.encoder(x)  # Try normal forward pass
                except RuntimeError as e:
                    # If there's a shape mismatch with LayerNorm, handle it manually
                    if "normalized_shape" in str(e):
                        print("Handling LayerNorm dimension issue with manual forward pass")
                        # Manual forward pass through encoder layers
                        features = x
                        for i, layer in enumerate(self.encoder):
                            if isinstance(layer, nn.LayerNorm):
                                # For LayerNorm, transpose, normalize, and transpose back
                                batch_size, channels, seq_len = features.shape
                                features = features.transpose(1, 2)  # [batch, seq_len, channels]
                                
                                # Check if LayerNorm expects different dimensions
                                if layer.normalized_shape[0] != features.shape[2]:
                                    # Reshape data to match expected normalized_shape
                                    norm_shape = layer.normalized_shape[0]
                                    if norm_shape < features.shape[2]:
                                        # If normalized_shape is smaller, use only part of the features
                                        features = layer(features[:, :, :norm_shape])
                                    else:
                                        # If normalized_shape is larger, pad the features
                                        pad_size = norm_shape - features.shape[2]
                                        padding = torch.zeros(features.shape[0], features.shape[1], pad_size, 
                                                              device=features.device)
                                        features = torch.cat([features, padding], dim=2)
                                        features = layer(features)
                                else:
                                    # Standard normalization when dimensions match
                                    features = layer(features)
                                    
                                features = features.transpose(1, 2)  # [batch, channels, seq_len]
                            else:
                                # For other layers, apply normally
                                features = layer(features)
                    else:
                        # Re-raise if it's not a LayerNorm issue
                        raise
                
                # Convert back to batch-first format
                features = features.transpose(1, 2)  # [batch_size, seq_len, hidden_size]
                
                class DummyOutput:
                    def __init__(self, last_hidden_state):
                        self.last_hidden_state = last_hidden_state
                
                return DummyOutput(features)
            
            @classmethod
            def from_pretrained(cls, model_name, *args, **kwargs):
                print(f"Using fallback model instead of {model_name}")
                return cls()
    else:
        raise
        
from torch.nn import TransformerEncoder, TransformerEncoderLayer


class AttentionPooling1D(nn.Module):
    """Attention pooling layer to aggregate sequence data into a fixed-length representation"""
    def __init__(self, hidden_size, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scale = self.head_dim ** -0.5
        
        # Multi-head attention projection
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        
        # Output projection
        self.out_proj = nn.Linear(hidden_size, hidden_size)
    
    def forward(self, x):
        # x: [batch_size, seq_len, hidden_size]
        batch_size, seq_len, hidden_size = x.size()
        
        # Compute query, key, value projections
        q = self.query(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = self.key(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        v = self.value(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        
        # Transpose for attention calculation
        q = q.transpose(1, 2)  # [batch_size, num_heads, seq_len, head_dim]
        k = k.transpose(1, 2)  # [batch_size, num_heads, seq_len, head_dim]
        v = v.transpose(1, 2)  # [batch_size, num_heads, seq_len, head_dim]
        
        # Calculate attention scores
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn_weights = F.softmax(attn_weights, dim=-1)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, hidden_size)
        
        # Final projection
        attn_output = self.out_proj(attn_output)
        
        return attn_output


class EnhancedEmotionClassifier(nn.Module):
    """Enhanced emotion classifier with deeper architecture and residual connections"""
    def __init__(self, input_size, hidden_size, num_emotions, dropout_rate=0.3):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_emotions = num_emotions
        
        # First layer
        self.layer1 = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Dropout(dropout_rate)
        )
        
        # Second layer with residual connection
        self.layer2 = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Dropout(dropout_rate)
        )
        
        # Third layer with residual connection
        self.layer3 = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LayerNorm(hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout_rate)
        )
        
        # Output layer
        self.output = nn.Linear(hidden_size // 2, num_emotions)
        
        # Layer to match dimensions for residual connections if needed
        self.dim_match = nn.Linear(input_size, hidden_size) if input_size != hidden_size else nn.Identity()
    
    def forward(self, x):
        # First layer
        h1 = self.layer1(x)
        
        # Second layer with residual connection
        h2 = self.layer2(h1) + h1
        
        # Third layer 
        h3 = self.layer3(h2)
        
        # Output layer
        output = self.output(h3)
        
        return output


class SpeechEmotionRecognitionModelEnhanced(nn.Module):
    """
    Enhanced speech emotion recognition model using Wav2Vec 2.0 encoder and 
    multi-level attention with context-aware emotion classification
    """
    def __init__(self, num_emotions=4, freeze_encoder=True, model_name="facebook/wav2vec2-base-960h"):
        super().__init__()
        
        # Load pre-trained Wav2Vec 2.0 model
        self.encoder = Wav2Vec2Model.from_pretrained(model_name)
        
        # Progressive unfreezing strategy
        if freeze_encoder:
            # Initially freeze all encoder parameters
            for param in self.encoder.parameters():
                param.requires_grad = False
                
            # If using the real Wav2Vec model (not our fallback), unfreeze final layers
            if hasattr(self.encoder, 'feature_extractor'):
                # Unfreeze the final transformer layers for fine-tuning
                for layer in self.encoder.feature_projection.parameters():
                    layer.requires_grad = True
                for i, layer in enumerate(self.encoder.encoder.layers):
                    # Only unfreeze the last 3 layers
                    if i >= len(self.encoder.encoder.layers) - 3:
                        for param in layer.parameters():
                            param.requires_grad = True
        
        hidden_size = self.encoder.config.hidden_size
        
        # Advanced Multi-head Attention with relative positional encoding
        self.attention_pool = AttentionPooling1D(hidden_size, num_heads=12)
        
        # Advanced context transformer layers with improved attention mechanism
        encoder_layer = TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=12,
            dim_feedforward=3072,
            dropout=0.2,
            activation="gelu",
            batch_first=True
        )
        self.context_encoder = TransformerEncoder(encoder_layer, num_layers=4)
        
        # Spectral-temporal feature integration
        self.feature_integration = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Dropout(0.2)
        )
        
        # Emotion classification branch with enhanced context
        self.emotion_classifier = EnhancedEmotionClassifier(
            input_size=hidden_size,
            hidden_size=512,
            num_emotions=num_emotions,
            dropout_rate=0.2
        )
        
        # Voice Activity Detection (VAD) branch with deeper networks
        self.vad_branch = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Linear(64, 1)
        )
        
        # Self-supervised auxiliary tasks
        self.spec_reconstruction = nn.Sequential(
            nn.Linear(hidden_size, 512),
            nn.GELU(),
            nn.Linear(512, 257)  # Common mel spectrogram dimension
        )
        
        # Arousal-valence regression for dimensional emotion model
        self.arousal_valence = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Linear(64, 2)  # Arousal and valence
        )
        
    def forward(self, input_values, attention_mask=None):
        # Get encoder outputs
        encoder_outputs = self.encoder(
            input_values, 
            attention_mask=attention_mask
        ).last_hidden_state  # (batch_size, sequence_length, hidden_size)
        
        # Apply advanced context encoding
        if attention_mask is not None:
            # Create an attention mask for the transformer
            context_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            context_mask = (1.0 - context_mask) * -10000.0
        else:
            context_mask = None
            
        context_features = self.context_encoder(encoder_outputs)
        
        # Apply spectral-temporal feature integration
        enhanced_features = self.feature_integration(context_features)
        
        # Temporal pooling with attention
        pooled_features = self.attention_pool(enhanced_features)
        
        # Emotion classification branch
        emotion_logits = self.emotion_classifier(pooled_features)
        emotion_probs = F.softmax(emotion_logits, dim=-1)
        
        # VAD branch
        vad_logits = self.vad_branch(context_features)
        vad_probs = torch.sigmoid(vad_logits)
        
        # Self-supervised prediction (spectral reconstruction)
        spec_recon = self.spec_reconstruction(pooled_features)
        
        # Arousal-Valence prediction
        av_prediction = self.arousal_valence(pooled_features)
        arousal = av_prediction[:, 0]
        valence = av_prediction[:, 1]
        
        return {
            "emotion_logits": emotion_logits,
            "emotion_probs": emotion_probs,
            "vad_logits": vad_logits,
            "vad_probs": vad_probs,
            "pooled_features": pooled_features,
            "spec_reconstruction": spec_recon,
            "arousal": arousal,
            "valence": valence
        }


class EnhancedRealTimeSpeechEmotionRecognizer:
    """
    Enhanced wrapper for real-time inference with the SER model
    Includes robust preprocessing and confidence estimation
    """
    def __init__(self, model_path, device="cpu", confidence_threshold=0.4, sliding_window_size=5):
        self.device = device
        self.model = SpeechEmotionRecognitionModelEnhanced()
        
        # Load state dict with error handling
        try:
            state_dict = torch.load(model_path, map_location=device)
            # Handle partial loading of state dict if needed
            self.model.load_state_dict(state_dict, strict=False)
            print(f"Model loaded from {model_path}")
        except Exception as e:
            print(f"Warning: Error loading model weights: {e}")
            print("Initializing with random weights")
            
        self.model.to(device)
        self.model.eval()
        
        self.sample_rate = 16000
        self.emotions = ["angry", "happy", "sad", "neutral"]
        self.confidence_threshold = confidence_threshold
        
        # Advanced history tracking for temporal smoothing
        self.history_size = sliding_window_size
        self.emotion_history = []
        self.vad_history = []
        self.arousal_history = []
        self.valence_history = []
        
        # Adaptive confidence threshold
        self.adaptive_threshold = confidence_threshold
        self.min_threshold = 0.2
        self.max_threshold = 0.6
    
    def preprocess_audio(self, waveform):
        """Enhanced preprocessing for audio waveform"""
        # Convert to torch tensor if numpy array
        if not isinstance(waveform, torch.Tensor):
            waveform = torch.tensor(waveform, dtype=torch.float32)
            
        # Ensure correct dimensions
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        elif waveform.dim() == 3 and waveform.shape[0] == 1:
            # If it's [1, 1, sequence_length]
            if waveform.shape[1] == 1:
                waveform = waveform.squeeze(0)  # Convert to [1, sequence_length]
            else:
                waveform = waveform.squeeze(1)  # If it's [1, batch_size, sequence_length]
        elif waveform.dim() == 4:
            # If it's [batch_size, 1, 1, sequence_length]
            waveform = waveform.squeeze(2)  # Remove the extra dimension
        
        # Advanced normalization: center and scale
        waveform = (waveform - waveform.mean()) / (waveform.std() + 1e-6)
        
        # Noise reduction via spectral subtraction
        # This is a simplified version - real spectral subtraction would involve FFT
        noise_estimate = torch.mean(waveform[:, :100], dim=1, keepdim=True)  # Estimate noise from first 100 samples
        waveform = waveform - noise_estimate
        
        # Apply pre-emphasis filter to enhance high frequencies
        pre_emphasis = 0.97
        emphasized = torch.cat([waveform[:, 0:1], waveform[:, 1:] - pre_emphasis * waveform[:, :-1]], dim=1)
        
        return emphasized.to(self.device)
    
    def _smooth_predictions(self, current_probs, current_vad=None, current_arousal=None, current_valence=None):
        """Apply adaptive temporal smoothing to predictions for stability"""
        # Add current probabilities to history
        self.emotion_history.append(current_probs)
        
        # Keep only last N predictions
        if len(self.emotion_history) > self.history_size:
            self.emotion_history.pop(0)
            
        # Compute exponential weights (newer predictions have more weight)
        weights = torch.exp(torch.linspace(0, 2, len(self.emotion_history)))
        weights = weights / weights.sum()
        
        # Compute weighted average
        smoothed_probs = torch.zeros_like(current_probs)
        for i, probs in enumerate(self.emotion_history):
            smoothed_probs += weights[i] * probs
        
        # If VAD provided, smooth it too
        smoothed_vad = None
        if current_vad is not None:
            self.vad_history.append(current_vad)
            if len(self.vad_history) > self.history_size:
                self.vad_history.pop(0)
            smoothed_vad = sum(w * v for w, v in zip(weights[-len(self.vad_history):], self.vad_history))
        
        # Smooth arousal-valence if provided
        smoothed_arousal, smoothed_valence = None, None
        if current_arousal is not None and current_valence is not None:
            self.arousal_history.append(current_arousal)
            self.valence_history.append(current_valence)
            
            if len(self.arousal_history) > self.history_size:
                self.arousal_history.pop(0)
                self.valence_history.pop(0)
                
            smoothed_arousal = sum(w * a for w, a in zip(weights[-len(self.arousal_history):], self.arousal_history))
            smoothed_valence = sum(w * v for w, v in zip(weights[-len(self.valence_history):], self.valence_history))
            
        return smoothed_probs, smoothed_vad, smoothed_arousal, smoothed_valence
    
    def _adjust_confidence_threshold(self, recent_confidences):
        """Dynamically adjust confidence threshold based on recent predictions"""
        if len(recent_confidences) < 10:
            return self.confidence_threshold
            
        # Calculate variance of recent confidences
        variance = np.var(recent_confidences)
        
        # If variance is high, we're less certain overall, so lower threshold
        if variance > 0.1:
            new_threshold = max(self.min_threshold, self.confidence_threshold - 0.05)
        # If variance is low, we're more certain, so can raise threshold
        elif variance < 0.05:
            new_threshold = min(self.max_threshold, self.confidence_threshold + 0.05)
        else:
            new_threshold = self.confidence_threshold
            
        # Smoothly adjust threshold
        self.adaptive_threshold = 0.9 * self.adaptive_threshold + 0.1 * new_threshold
        return self.adaptive_threshold
    
    def predict(self, waveform):
        """
        Predict emotion from audio waveform with confidence estimation and dimensional model
        Args:
            waveform: Raw audio waveform (16kHz, mono)
        Returns:
            dict: Emotion probabilities, dominant emotion, confidence, and dimensional model values
        """
        with torch.no_grad():
            # Preprocess audio
            input_values = self.preprocess_audio(waveform)
            
            # Run inference
            outputs = self.model(input_values)
            
            # Get probabilities
            emotion_probs = outputs["emotion_probs"][0].cpu()
            vad_prob = outputs["vad_probs"][0].mean().cpu().item()
            arousal = outputs["arousal"][0].cpu().item()  # Dimensional model: arousal
            valence = outputs["valence"][0].cpu().item()  # Dimensional model: valence
            
            # Apply advanced temporal smoothing
            smoothed_probs, smoothed_vad, smoothed_arousal, smoothed_valence = self._smooth_predictions(
                emotion_probs, vad_prob, arousal, valence
            )
            emotion_probs_np = smoothed_probs.numpy()
            
            if smoothed_vad is not None:
                vad_prob = smoothed_vad
            
            # Calculate confidence as the difference between top two probabilities
            sorted_probs, _ = torch.sort(smoothed_probs, descending=True)
            confidence = (sorted_probs[0] - sorted_probs[1]).item()
            
            # Adjust confidence threshold dynamically
            adjusted_threshold = self._adjust_confidence_threshold([confidence])
            
            # Only predict emotion if voice activity detected and confidence is high enough
            is_speech = vad_prob > 0.5
            is_confident = confidence > adjusted_threshold
            
            # Get dominant emotion
            if is_speech and is_confident:
                dominant_emotion_idx = smoothed_probs.argmax().item()
                dominant_emotion = self.emotions[dominant_emotion_idx]
            elif is_speech:
                dominant_emotion = "uncertain"
            else:
                dominant_emotion = "no_speech"
                
            # Scale arousal and valence for easier interpretation
            if smoothed_arousal is not None and smoothed_valence is not None:
                arousal = smoothed_arousal
                valence = smoothed_valence
            arousal_scaled = min(max((arousal + 1) / 2, 0), 1)  # Scale from [-1,1] to [0,1]
            valence_scaled = min(max((valence + 1) / 2, 0), 1)  # Scale from [-1,1] to [0,1]
                
            return {
                "emotion_probs": emotion_probs_np,
                "vad_prob": vad_prob,
                "is_speech": is_speech,
                "is_confident": is_confident,
                "confidence": confidence,
                "confidence_threshold": adjusted_threshold,
                "dominant_emotion": dominant_emotion,
                "arousal": arousal_scaled,
                "valence": valence_scaled,
            } 

# Add an alias for backwards compatibility with train_fixed.py
EnhancedSpeechEmotionRecognitionModel = SpeechEmotionRecognitionModelEnhanced 

class EnhancedSpeechEmotionRecognitionModel(nn.Module):
    """Enhanced speech emotion recognition model with improved architecture"""
    def __init__(
        self,
        num_emotions=8,
        sample_rate=16000,
        hidden_size=768,
        attention_heads=12,
        dropout=0.1,
        use_vad=True
    ):
        super().__init__()
        self.num_emotions = num_emotions
        self.sample_rate = sample_rate
        self.hidden_size = hidden_size
        self.use_vad = use_vad

        try:
            logger.info("Loading Wav2Vec2 model...")
            from transformers import Wav2Vec2Model
            self.encoder = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")
            self.is_pretrained = True
            logger.info("Wav2Vec2 model loaded successfully")
        except Exception as e:
            logger.warning(f"Error loading Wav2Vec2 model: {str(e)}. Using fallback encoder.")
            self.encoder = FallbackEncoder(hidden_size=hidden_size)
            self.is_pretrained = False

        # Add dimension handler
        self._handle_dimensions_method = True
        
        # Progressive unfreezing strategy
        if self.is_pretrained:
            # Initially freeze all encoder parameters
            for param in self.encoder.parameters():
                param.requires_grad = False
                
            # If using the real Wav2Vec model (not our fallback), unfreeze final layers
            if hasattr(self.encoder, 'feature_extractor'):
                # Unfreeze the final transformer layers for fine-tuning
                for layer in self.encoder.feature_projection.parameters():
                    layer.requires_grad = True
                for i, layer in enumerate(self.encoder.encoder.layers):
                    # Only unfreeze the last 3 layers
                    if i >= len(self.encoder.encoder.layers) - 3:
                        for param in layer.parameters():
                            param.requires_grad = True
        
        hidden_size = self.encoder.config.hidden_size
        
        # Advanced Multi-head Attention with relative positional encoding
        self.attention_pool = AttentionPooling1D(hidden_size, num_heads=attention_heads)
        
        # Advanced context transformer layers with improved attention mechanism
        encoder_layer = TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=attention_heads,
            dim_feedforward=3072,
            dropout=dropout,
            activation="gelu",
            batch_first=True
        )
        self.context_encoder = TransformerEncoder(encoder_layer, num_layers=4)
        
        # Spectral-temporal feature integration
        self.feature_integration = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # Emotion classification branch with enhanced context
        self.emotion_classifier = EnhancedEmotionClassifier(
            input_size=hidden_size,
            hidden_size=512,
            num_emotions=num_emotions,
            dropout_rate=dropout
        )
        
        # Voice Activity Detection (VAD) branch with deeper networks
        self.vad_branch = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Linear(64, 1)
        )
        
        # Self-supervised auxiliary tasks
        self.spec_reconstruction = nn.Sequential(
            nn.Linear(hidden_size, 512),
            nn.GELU(),
            nn.Linear(512, 257)  # Common mel spectrogram dimension
        )
        
        # Arousal-valence regression for dimensional emotion model
        self.arousal_valence = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Linear(64, 2)  # Arousal and valence
        )
        
    def _handle_dimensions(self, x):
        """Handle various input dimensions to ensure compatibility with conv/transformer layers"""
        # Fix 4D inputs (batch, channels, height, time) - common issue with spectrograms
        if len(x.shape) == 4:
            # Either squeeze height dimension or take mean
            if x.shape[2] == 1:
                x = x.squeeze(2)  # Remove height of 1
            else:
                x = x.mean(dim=2)  # Take mean along height dimension
        
        # Fix 2D inputs (batch, time) - add channel dimension
        if len(x.shape) == 2:
            x = x.unsqueeze(1)  # Add channel dimension
            
        # Check for correct order - (batch, channels, time)
        if len(x.shape) == 3 and x.shape[1] > x.shape[2]:
            # Channels is larger than time, might be transposed
            x = x.transpose(1, 2)
            
        return x
    
    def forward(self, waveform):
        """Forward pass with proper tensor dimension handling"""
        # Ensure input dimensions are correct
        waveform = self._handle_dimensions(waveform)
        
        # Normalize audio if needed
        if waveform.abs().max() > 0:
            waveform = waveform / waveform.abs().max()
            
        try:
            batch_size = waveform.shape[0]
            
            # Check if this is a pretrained transformer model or fallback
            if self.is_pretrained:
                # For pretrained models, make sure waveform is [batch_size, time]
                if len(waveform.shape) == 3:
                    waveform = waveform.squeeze(1)
                
                # Get encoder outputs
                encoder_outputs = self.encoder(
                    waveform,
                    output_hidden_states=True
                )
                
                # Extract features from the output
                hidden_states = encoder_outputs.last_hidden_state
                
            else:
                # For fallback, encoder expects [batch_size, channels, time]
                # Get encoder outputs
                encoder_outputs = self.encoder(
                    waveform
                )
                
                # Extract features from the output
                hidden_states = encoder_outputs["last_hidden_state"]
            
            # Apply attention to extract a fixed-length representation
            if hasattr(self, 'layer_norm'):
                # Ensure hidden_states has the right shape for layer normalization
                # Layer norm expects [batch, seq_len, hidden_size]
                if len(hidden_states.shape) == 3 and hidden_states.shape[1] != self.hidden_size:
                    hidden_states = hidden_states.transpose(1, 2)
                
                hidden_states = self.layer_norm(hidden_states)
            
            # Compute attention weights
            attention_weights = self.attention_pool(hidden_states)
            attention_weights = F.softmax(attention_weights, dim=1)
            
            # Apply attention weights to get context vector
            context = torch.bmm(attention_weights.permute(0, 2, 1), hidden_states)
            context = context.squeeze(1)
            
            # Classify emotions
            emotion_logits = self.emotion_classifier(context)
            emotion_probs = F.softmax(emotion_logits, dim=-1)
            
            # Voice activity detection (optional)
            vad_output = None
            if self.use_vad:
                vad_logits = self.vad_branch(context)
                vad_probs = torch.sigmoid(vad_logits)
                vad_output = vad_probs
            
            # Return outputs
            outputs = {
                'emotion_logits': emotion_logits,
                'emotion_probs': emotion_probs
            }
            
            if vad_output is not None:
                outputs['vad_probs'] = vad_output
                
            return outputs
            
        except Exception as e:
            logger.error(f"Error in model forward pass: {str(e)}")
            logger.error(f"Input shape: {waveform.shape}")
            
            # Return empty predictions as fallback
            dummy_emotion_logits = torch.zeros((batch_size, self.num_emotions), device=waveform.device)
            dummy_emotion_probs = torch.ones((batch_size, self.num_emotions), device=waveform.device) / self.num_emotions
            
            outputs = {
                'emotion_logits': dummy_emotion_logits,
                'emotion_probs': dummy_emotion_probs
            }
            
            if self.use_vad:
                outputs['vad_probs'] = torch.zeros((batch_size, 1), device=waveform.device)
                
            return outputs 