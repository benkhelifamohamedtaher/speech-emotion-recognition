import os
import argparse
import torch
import onnx
import onnxruntime as ort
import numpy as np
from model import SpeechEmotionRecognitionModel, RealTimeSpeechEmotionRecognizer


def export_to_onnx(model_path, output_path, max_length=48000, device='cpu', optimize=True):
    """
    Export PyTorch model to ONNX format
    
    Args:
        model_path: Path to PyTorch model
        output_path: Path to save ONNX model
        max_length: Maximum audio length in samples
        device: Device to load model on
        optimize: Whether to optimize the ONNX model
    
    Returns:
        Path to saved ONNX model
    """
    print(f"Loading PyTorch model from {model_path}")
    
    # Load PyTorch model
    model = SpeechEmotionRecognitionModel()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    # Create dummy input
    dummy_input = torch.randn(1, 1, max_length, device=device)
    
    # Export to ONNX
    print(f"Exporting model to ONNX: {output_path}")
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        input_names=['input'],
        output_names=['emotion_logits', 'emotion_probs', 'vad_logits', 'vad_probs'],
        dynamic_axes={
            'input': {0: 'batch_size', 2: 'sequence_length'},
            'emotion_logits': {0: 'batch_size'},
            'emotion_probs': {0: 'batch_size'},
            'vad_logits': {0: 'batch_size', 1: 'sequence_length'},
            'vad_probs': {0: 'batch_size', 1: 'sequence_length'}
        },
        opset_version=12,
        verbose=False
    )
    
    # Verify ONNX model
    print("Verifying ONNX model...")
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    
    if optimize:
        from onnxruntime.transformers import optimizer
        from onnxruntime.transformers.onnx_model_bert import BertOptimizationOptions
        
        # Optimize for inference
        print("Optimizing ONNX model...")
        opt_options = BertOptimizationOptions('bert')
        opt_options.enable_gelu = True
        opt_options.enable_layer_norm = True
        opt_options.enable_attention = True
        
        opt_model = optimizer.optimize_model(
            output_path,
            'bert',
            num_heads=12,
            hidden_size=768,
            optimization_options=opt_options
        )
        opt_model.save_model_to_file(output_path)
        
    print("ONNX export completed successfully")
    
    return output_path


def test_onnx_inference(onnx_path, test_input=None):
    """
    Test ONNX model inference
    
    Args:
        onnx_path: Path to ONNX model
        test_input: Test input tensor (optional)
    
    Returns:
        Inference results
    """
    print(f"Testing ONNX model: {onnx_path}")
    
    # Create ONNX Runtime session
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    ort_session = ort.InferenceSession(onnx_path, options)
    
    # Create test input if not provided
    if test_input is None:
        test_input = np.random.randn(1, 1, 16000).astype(np.float32)
    
    # Run inference
    ort_inputs = {ort_session.get_inputs()[0].name: test_input}
    ort_outputs = ort_session.run(None, ort_inputs)
    
    # Get output names
    output_names = [output.name for output in ort_session.get_outputs()]
    
    # Print results
    print(f"ONNX Model Input Shape: {test_input.shape}")
    for i, output in enumerate(ort_outputs):
        print(f"Output '{output_names[i]}' Shape: {output.shape}")
    
    # Check emotion probabilities
    emotion_probs_idx = output_names.index('emotion_probs')
    emotion_probs = ort_outputs[emotion_probs_idx]
    
    emotions = ["angry", "happy", "sad", "neutral"]
    dominant_emotion = emotions[np.argmax(emotion_probs)]
    
    print(f"Emotion Probabilities: {emotion_probs.flatten()}")
    print(f"Dominant Emotion: {dominant_emotion}")
    
    # Measure inference time
    import time
    start_time = time.time()
    for _ in range(10):
        ort_session.run(None, ort_inputs)
    end_time = time.time()
    avg_time = (end_time - start_time) / 10
    
    print(f"Average Inference Time: {avg_time*1000:.2f}ms")
    print(f"Frames per Second: {1.0/avg_time:.2f}")
    
    return ort_outputs


def create_tflite_model(onnx_path, output_path):
    """
    Convert ONNX model to TFLite format
    
    Note: This requires TensorFlow and ONNX-TensorFlow packages
    """
    try:
        import tensorflow as tf
        import onnx_tf
    except ImportError:
        print("Error: TensorFlow or ONNX-TensorFlow not found.")
        print("Please install with: pip install tensorflow onnx-tf")
        return None
    
    print(f"Converting ONNX model to TFLite: {output_path}")
    
    # Load ONNX model
    onnx_model = onnx.load(onnx_path)
    
    # Convert to TensorFlow
    tf_rep = onnx_tf.backend.prepare(onnx_model)
    
    # Save TensorFlow model
    tf_model_path = output_path.replace('.tflite', '_tf')
    tf_rep.export_graph(tf_model_path)
    
    # Convert to TFLite
    converter = tf.lite.TFLiteConverter.from_saved_model(tf_model_path)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
    
    # Save TFLite model
    with open(output_path, 'wb') as f:
        f.write(tflite_model)
    
    print(f"TFLite conversion completed: {output_path}")
    
    return output_path


def main():
    parser = argparse.ArgumentParser(description='Export Speech Emotion Recognition Model')
    
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to trained PyTorch model')
    parser.add_argument('--output_dir', type=str, default='./exports',
                        help='Directory to save exported models')
    parser.add_argument('--format', type=str, default='onnx', choices=['onnx', 'tflite', 'all'],
                        help='Export format')
    parser.add_argument('--max_length', type=int, default=48000,
                        help='Maximum audio length in samples (3 seconds at 16kHz)')
    parser.add_argument('--optimize', action='store_true',
                        help='Optimize exported model for inference')
    parser.add_argument('--test', action='store_true',
                        help='Test exported model inference')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Export to ONNX
    if args.format in ['onnx', 'all']:
        onnx_path = os.path.join(args.output_dir, 'model.onnx')
        onnx_path = export_to_onnx(
            model_path=args.model_path,
            output_path=onnx_path,
            max_length=args.max_length,
            optimize=args.optimize
        )
        
        if args.test:
            test_onnx_inference(onnx_path)
    
    # Export to TFLite
    if args.format in ['tflite', 'all']:
        if args.format == 'tflite' and 'onnx_path' not in locals():
            # Export ONNX first
            onnx_path = os.path.join(args.output_dir, 'model.onnx')
            onnx_path = export_to_onnx(
                model_path=args.model_path,
                output_path=onnx_path,
                max_length=args.max_length,
                optimize=args.optimize
            )
        
        tflite_path = os.path.join(args.output_dir, 'model.tflite')
        create_tflite_model(onnx_path, tflite_path)


if __name__ == '__main__':
    main() 