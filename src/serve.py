import os
import time
import argparse
import json
import numpy as np
import soundfile as sf
from flask import Flask, request, jsonify
import torch
from model import RealTimeSpeechEmotionRecognizer


# Initialize Flask app
app = Flask(__name__)

# Global variables for model and settings
model = None
sample_rate = 16000
device = "cpu"
batch_processor = None


class BatchProcessor:
    """Process multiple audio files in a batch for more efficient inference"""
    
    def __init__(self, model_path, device="cpu", max_batch_size=16):
        self.recognizer = RealTimeSpeechEmotionRecognizer(model_path, device)
        self.max_batch_size = max_batch_size
        
    def process_batch(self, audio_list):
        """
        Process a batch of audio waveforms
        
        Args:
            audio_list: List of audio waveforms (numpy arrays)
            
        Returns:
            List of prediction results
        """
        # Preprocess all audio
        batch_size = len(audio_list)
        processed_audio = []
        
        for audio in audio_list:
            processed = self.recognizer.preprocess_audio(audio)
            processed_audio.append(processed)
        
        # Stack into a batch
        batch_input = torch.cat(processed_audio, dim=0)
        
        # Perform inference in a single batch
        with torch.no_grad():
            outputs = self.recognizer.model(batch_input)
            
            # Process results
            results = []
            for i in range(batch_size):
                emotion_probs = outputs["emotion_probs"][i].cpu().numpy()
                vad_prob = outputs["vad_probs"][i].mean().cpu().numpy()
                
                # Determine emotion
                is_speech = vad_prob > 0.5
                if is_speech:
                    dominant_emotion_idx = emotion_probs.argmax().item()
                    dominant_emotion = self.recognizer.emotions[dominant_emotion_idx]
                else:
                    dominant_emotion = "no_speech"
                
                results.append({
                    "emotion_probs": {emotion: float(emotion_probs[i]) 
                                     for i, emotion in enumerate(self.recognizer.emotions)},
                    "vad_prob": float(vad_prob),
                    "is_speech": bool(is_speech),
                    "dominant_emotion": dominant_emotion
                })
            
            return results


@app.route('/predict', methods=['POST'])
def predict():
    """
    Predict emotion from audio
    
    Expects audio data in the request file or a path to an audio file
    """
    global model, sample_rate
    
    # Get start time for latency measurement
    start_time = time.time()
    
    try:
        # Check if request has file or path
        if 'file' in request.files:
            # Get audio file from request
            audio_file = request.files['file']
            
            # Load audio using soundfile
            waveform, file_sr = sf.read(audio_file)
            
            # Convert to mono if stereo
            if len(waveform.shape) > 1:
                waveform = np.mean(waveform, axis=1)
            
            # Resample if needed
            if file_sr != sample_rate:
                # Simple resampling (for production, use a proper resampling method)
                import librosa
                waveform = librosa.resample(waveform, orig_sr=file_sr, target_sr=sample_rate)
        
        elif 'path' in request.form:
            # Get audio file path from request
            audio_path = request.form['path']
            
            # Check if file exists
            if not os.path.exists(audio_path):
                return jsonify({
                    "error": f"Audio file not found: {audio_path}"
                }), 404
            
            # Load audio using soundfile
            waveform, file_sr = sf.read(audio_path)
            
            # Convert to mono if stereo
            if len(waveform.shape) > 1:
                waveform = np.mean(waveform, axis=1)
            
            # Resample if needed
            if file_sr != sample_rate:
                # Simple resampling (for production, use a proper resampling method)
                import librosa
                waveform = librosa.resample(waveform, orig_sr=file_sr, target_sr=sample_rate)
        
        else:
            return jsonify({
                "error": "No audio file or path provided"
            }), 400
        
        # Make prediction
        result = model.predict(waveform)
        
        # Convert numpy types to Python types for JSON serialization
        response = {
            "emotion_probs": {emotion: float(prob) 
                             for emotion, prob in zip(model.emotions, result["emotion_probs"])},
            "vad_prob": float(result["vad_prob"]),
            "is_speech": bool(result["is_speech"]),
            "dominant_emotion": result["dominant_emotion"],
            "latency_ms": float((time.time() - start_time) * 1000)
        }
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({
            "error": str(e)
        }), 500


@app.route('/predict_batch', methods=['POST'])
def predict_batch():
    """
    Predict emotions from multiple audio files
    
    Expects JSON with a list of audio file paths
    """
    global batch_processor, sample_rate
    
    start_time = time.time()
    
    try:
        # Parse request JSON
        request_data = request.get_json()
        
        if not request_data or 'files' not in request_data:
            return jsonify({
                "error": "No audio files provided"
            }), 400
        
        # Get list of audio file paths
        audio_paths = request_data['files']
        
        if not audio_paths:
            return jsonify({
                "error": "Empty list of audio files"
            }), 400
        
        # Load all audio files
        audio_list = []
        file_names = []
        
        for path in audio_paths:
            # Check if file exists
            if not os.path.exists(path):
                return jsonify({
                    "error": f"Audio file not found: {path}"
                }), 404
            
            # Load audio using soundfile
            waveform, file_sr = sf.read(path)
            
            # Convert to mono if stereo
            if len(waveform.shape) > 1:
                waveform = np.mean(waveform, axis=1)
            
            # Resample if needed
            if file_sr != sample_rate:
                import librosa
                waveform = librosa.resample(waveform, orig_sr=file_sr, target_sr=sample_rate)
            
            audio_list.append(waveform)
            file_names.append(os.path.basename(path))
        
        # Process batch
        results = batch_processor.process_batch(audio_list)
        
        # Prepare response
        response = {
            "results": [
                {
                    "file": file_name,
                    "prediction": result
                }
                for file_name, result in zip(file_names, results)
            ],
            "latency_ms": float((time.time() - start_time) * 1000),
            "avg_latency_per_file_ms": float((time.time() - start_time) * 1000 / len(audio_list))
        }
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({
            "error": str(e)
        }), 500


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "ok",
        "model": "speech_emotion_recognition",
        "version": "1.0.0"
    })


def main():
    global model, sample_rate, device, batch_processor
    
    parser = argparse.ArgumentParser(description='Speech Emotion Recognition API Server')
    
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to trained model')
    parser.add_argument('--device', type=str, default='cpu',
                        help='Device to run inference on (cpu or cuda)')
    parser.add_argument('--sample_rate', type=int, default=16000,
                        help='Audio sample rate')
    parser.add_argument('--host', type=str, default='0.0.0.0',
                        help='Host to run the server on')
    parser.add_argument('--port', type=int, default=5000,
                        help='Port to run the server on')
    parser.add_argument('--debug', action='store_true',
                        help='Run in debug mode')
    
    args = parser.parse_args()
    
    # Set global variables
    sample_rate = args.sample_rate
    device = args.device
    
    # Initialize model
    print(f"Loading model from {args.model_path}")
    model = RealTimeSpeechEmotionRecognizer(args.model_path, device)
    
    # Initialize batch processor
    batch_processor = BatchProcessor(args.model_path, device)
    
    # Start server
    print(f"Starting server at http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == '__main__':
    main() 