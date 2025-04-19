import os
import argparse
import time
import numpy as np
import torch
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

from model import SpeechEmotionRecognitionModel, RealTimeSpeechEmotionRecognizer
from data_utils import create_dataloaders


def evaluate_model(model_path, dataset_root, batch_size=16, device='cpu', 
                  sample_rate=16000, max_length=48000):
    """
    Evaluate trained model on test set
    
    Args:
        model_path: Path to trained model
        dataset_root: Path to dataset root
        batch_size: Batch size for evaluation
        device: Device to run evaluation on
        sample_rate: Audio sample rate
        max_length: Max audio length in samples
        
    Returns:
        dict: Evaluation metrics
    """
    print(f"Evaluating model: {model_path}")
    
    # Create dataloaders (only need test loader)
    _, test_loader = create_dataloaders(
        dataset_root=dataset_root,
        batch_size=batch_size,
        target_sr=sample_rate,
        max_length=max_length,
        apply_augmentation=False
    )
    
    print(f"Evaluating on {len(test_loader.dataset)} samples")
    
    # Load model
    model = SpeechEmotionRecognitionModel()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    # Evaluation
    all_emotion_preds = []
    all_emotion_probs = []
    all_emotion_labels = []
    latencies = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            waveforms = batch['waveform'].to(device)
            emotion_labels = batch['label'].to(device)
            
            # Measure inference time
            start_time = time.time()
            outputs = model(waveforms)
            end_time = time.time()
            
            # Calculate latency (in ms)
            batch_latency = (end_time - start_time) * 1000 / waveforms.size(0)
            latencies.append(batch_latency)
            
            # Collect predictions and labels
            emotion_probs = outputs['emotion_probs'].cpu().numpy()
            emotion_preds = np.argmax(emotion_probs, axis=1)
            
            all_emotion_probs.extend(emotion_probs)
            all_emotion_preds.extend(emotion_preds)
            all_emotion_labels.extend(emotion_labels.cpu().numpy())
    
    # Calculate metrics
    accuracy = accuracy_score(all_emotion_labels, all_emotion_preds)
    f1 = f1_score(all_emotion_labels, all_emotion_preds, average='macro')
    conf_matrix = confusion_matrix(all_emotion_labels, all_emotion_preds)
    class_report = classification_report(
        all_emotion_labels, 
        all_emotion_preds,
        target_names=["angry", "happy", "sad", "neutral"],
        output_dict=True
    )
    
    # Calculate average latency
    avg_latency = np.mean(latencies)
    p95_latency = np.percentile(latencies, 95)
    
    # Calculate inference speed in frames per second
    frames_per_second = 1000 / avg_latency
    
    # Print results
    print("\nEvaluation Results:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 Score (macro): {f1:.4f}")
    print(f"Average Latency: {avg_latency:.2f}ms")
    print(f"P95 Latency: {p95_latency:.2f}ms")
    print(f"Inference Speed: {frames_per_second:.2f} frames/sec")
    
    print("\nConfusion Matrix:")
    print(conf_matrix)
    
    print("\nClassification Report:")
    for emotion in ["angry", "happy", "sad", "neutral"]:
        precision = class_report[emotion]['precision']
        recall = class_report[emotion]['recall']
        f1 = class_report[emotion]['f1-score']
        print(f"{emotion.ljust(8)}: Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}")
    
    # Return metrics
    return {
        'accuracy': accuracy,
        'f1_score': f1,
        'avg_latency': avg_latency,
        'p95_latency': p95_latency,
        'frames_per_second': frames_per_second,
        'confusion_matrix': conf_matrix,
        'classification_report': class_report
    }


def evaluate_streaming_performance(model_path, sample_rate=16000, chunk_size=0.5, buffer_size=3.0, 
                                   num_chunks=10, device='cpu'):
    """
    Evaluate streaming inference performance with simulated audio chunks
    
    Args:
        model_path: Path to trained model
        sample_rate: Audio sample rate
        chunk_size: Size of each audio chunk in seconds
        buffer_size: Size of audio buffer in seconds
        num_chunks: Number of chunks to simulate
        device: Device to run evaluation on
        
    Returns:
        dict: Performance metrics
    """
    print("\nEvaluating Streaming Performance:")
    print(f"Sample Rate: {sample_rate} Hz")
    print(f"Chunk Size: {chunk_size} seconds")
    print(f"Buffer Size: {buffer_size} seconds")
    
    # Initialize model
    recognizer = RealTimeSpeechEmotionRecognizer(model_path, device)
    
    # Convert sizes to samples
    chunk_samples = int(sample_rate * chunk_size)
    buffer_samples = int(sample_rate * buffer_size)
    
    # Create simulated audio buffer
    audio_buffer = np.zeros(buffer_samples, dtype=np.float32)
    
    # Simulate different emotion types in chunks
    emotion_types = ["angry", "happy", "sad", "neutral"]
    
    # Measure processing times
    processing_times = []
    
    for i in range(num_chunks):
        # Simulate new audio chunk with emotion-like patterns
        emotion_idx = i % len(emotion_types)
        
        # Create synthetic emotion pattern (just for simulation)
        chunk = np.random.randn(chunk_samples) * 0.1  # Base noise
        
        # Add synthetic patterns for different emotions (simplified)
        if emotion_types[emotion_idx] == "angry":
            # Higher amplitude, more high frequency components
            chunk = chunk * 1.5 + np.sin(np.linspace(0, 100, chunk_samples)) * 0.5
        elif emotion_types[emotion_idx] == "happy":
            # More rhythmic pattern
            chunk = chunk + np.sin(np.linspace(0, 50, chunk_samples)) * 0.8
        elif emotion_types[emotion_idx] == "sad":
            # Lower amplitude, smoother
            chunk = chunk * 0.7 + np.sin(np.linspace(0, 10, chunk_samples)) * 0.4
        # neutral remains mostly noise
        
        # Update buffer with new chunk
        audio_buffer = np.roll(audio_buffer, -chunk_samples)
        audio_buffer[-chunk_samples:] = chunk
        
        # Measure prediction time
        start_time = time.time()
        result = recognizer.predict(audio_buffer)
        end_time = time.time()
        
        # Calculate processing time (in ms)
        proc_time = (end_time - start_time) * 1000
        processing_times.append(proc_time)
        
        print(f"Chunk {i+1}: Dominant Emotion={result['dominant_emotion']}, "
              f"Processing Time={proc_time:.2f}ms")
    
    # Calculate statistics
    avg_proc_time = np.mean(processing_times)
    p95_proc_time = np.percentile(processing_times, 95)
    max_proc_time = np.max(processing_times)
    
    print("\nStreaming Performance Results:")
    print(f"Average Processing Time: {avg_proc_time:.2f}ms")
    print(f"P95 Processing Time: {p95_proc_time:.2f}ms")
    print(f"Max Processing Time: {max_proc_time:.2f}ms")
    print(f"Real-time Factor: {avg_proc_time / (chunk_size * 1000):.4f}x")
    
    # Return metrics
    return {
        'avg_processing_time': avg_proc_time,
        'p95_processing_time': p95_proc_time,
        'max_processing_time': max_proc_time,
        'real_time_factor': avg_proc_time / (chunk_size * 1000)
    }


def main():
    parser = argparse.ArgumentParser(description='Evaluate Speech Emotion Recognition Model')
    
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to trained model')
    parser.add_argument('--dataset_root', type=str, default='./dataset',
                        help='Path to dataset root')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size for evaluation')
    parser.add_argument('--device', type=str, default='cpu',
                        help='Device to run evaluation on')
    parser.add_argument('--sample_rate', type=int, default=16000,
                        help='Audio sample rate')
    parser.add_argument('--max_length', type=int, default=48000,
                        help='Max audio length in samples')
    parser.add_argument('--streaming', action='store_true',
                        help='Evaluate streaming performance')
    parser.add_argument('--output_file', type=str, default=None,
                        help='Path to save evaluation results')
    
    args = parser.parse_args()
    
    # Evaluate model
    metrics = evaluate_model(
        model_path=args.model_path,
        dataset_root=args.dataset_root,
        batch_size=args.batch_size,
        device=args.device,
        sample_rate=args.sample_rate,
        max_length=args.max_length
    )
    
    # Evaluate streaming performance
    if args.streaming:
        streaming_metrics = evaluate_streaming_performance(
            model_path=args.model_path,
            sample_rate=args.sample_rate,
            device=args.device
        )
        metrics['streaming'] = streaming_metrics
    
    # Save results if output file specified
    if args.output_file:
        import json
        
        # Convert numpy types to Python types for JSON serialization
        for key, value in metrics.items():
            if isinstance(value, np.ndarray):
                metrics[key] = value.tolist()
            elif isinstance(value, np.generic):
                metrics[key] = value.item()
                
        # Handle nested dictionaries
        if 'classification_report' in metrics:
            for emotion, values in metrics['classification_report'].items():
                if isinstance(values, dict):
                    for k, v in values.items():
                        if isinstance(v, np.generic):
                            metrics['classification_report'][emotion][k] = v.item()
        
        with open(args.output_file, 'w') as f:
            json.dump(metrics, f, indent=4)
        
        print(f"Evaluation results saved to {args.output_file}")


if __name__ == '__main__':
    main() 