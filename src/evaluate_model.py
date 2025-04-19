#!/usr/bin/env python3
"""
Script to evaluate a trained speech emotion recognition model on test data
"""

import os
import argparse
import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, f1_score

# Import model and data utils
from model_enhanced import SpeechEmotionRecognitionModelEnhanced
from data_utils import create_dataloaders


def plot_confusion_matrix(cm, class_names, output_path=None):
    """Plot confusion matrix with seaborn"""
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Confusion matrix saved to {output_path}")
    else:
        plt.show()
    plt.close()


def plot_emotion_distribution(true_labels, pred_labels, class_names, output_path=None):
    """Plot distribution of true and predicted emotions"""
    plt.figure(figsize=(12, 6))
    
    # Count occurrences
    true_counts = [sum(1 for label in true_labels if label == i) for i in range(len(class_names))]
    pred_counts = [sum(1 for label in pred_labels if label == i) for i in range(len(class_names))]
    
    x = np.arange(len(class_names))
    width = 0.35
    
    plt.bar(x - width/2, true_counts, width, label='True')
    plt.bar(x + width/2, pred_counts, width, label='Predicted')
    
    plt.xticks(x, class_names)
    plt.ylabel('Count')
    plt.title('Emotion Distribution')
    plt.legend()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Emotion distribution plot saved to {output_path}")
    else:
        plt.show()
    plt.close()


def evaluate_model(args):
    """
    Evaluate trained model on test data
    
    Args:
        args: Command line arguments
    """
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu')
    print(f"Using device: {device}")
    
    # Create output directory if needed
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
    
    # Create test dataloader
    _, test_loader = create_dataloaders(
        dataset_root=args.dataset_root,
        batch_size=args.batch_size,
        target_sr=args.sample_rate,
        max_length=args.max_length,
        apply_augmentation=False,
        num_workers=args.num_workers
    )
    
    print(f"Evaluating on {len(test_loader.dataset)} samples")
    
    # Load model
    model = SpeechEmotionRecognitionModelEnhanced(num_emotions=4)
    
    # Load weights
    try:
        state_dict = torch.load(args.model_path, map_location=device)
        model.load_state_dict(state_dict, strict=False)
        print(f"Model loaded from {args.model_path}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    model.to(device)
    model.eval()
    
    # Prepare for evaluation
    all_preds = []
    all_labels = []
    all_probs = []
    class_names = ["angry", "happy", "sad", "neutral"]
    
    # Evaluate
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            waveforms = batch['waveform'].to(device)
            labels = batch['label'].to(device)
            
            # Forward pass
            outputs = model(waveforms)
            probs = outputs['emotion_probs']
            
            # Get predictions
            preds = probs.argmax(dim=1)
            
            # Store results
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='macro')
    cm = confusion_matrix(all_labels, all_preds)
    report = classification_report(all_labels, all_preds, target_names=class_names)
    
    # Print results
    print("\nEvaluation Results:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 Score: {accuracy:.4f}")
    print("\nClassification Report:")
    print(report)
    
    # Save results to file if output directory specified
    if args.output_dir:
        # Save confusion matrix plot
        cm_path = os.path.join(args.output_dir, "confusion_matrix.png")
        plot_confusion_matrix(cm, class_names, cm_path)
        
        # Save emotion distribution plot
        dist_path = os.path.join(args.output_dir, "emotion_distribution.png")
        plot_emotion_distribution(all_labels, all_preds, class_names, dist_path)
        
        # Save metrics to text file
        metrics_path = os.path.join(args.output_dir, "eval_metrics.txt")
        with open(metrics_path, 'w') as f:
            f.write("Evaluation Results\n")
            f.write("=================\n\n")
            f.write(f"Model: {args.model_path}\n")
            f.write(f"Dataset: {args.dataset_root}\n\n")
            f.write(f"Accuracy: {accuracy:.4f}\n")
            f.write(f"F1 Score: {f1:.4f}\n\n")
            f.write("Classification Report:\n")
            f.write(report)
            f.write("\n")
            f.write("Confusion Matrix:\n")
            f.write(str(cm))
        
        print(f"Evaluation results saved to {metrics_path}")


def main():
    parser = argparse.ArgumentParser(description='Evaluate Speech Emotion Recognition Model')
    
    # Model parameters
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to trained model file')
    parser.add_argument('--dataset_root', type=str, default='./processed_dataset',
                        help='Path to the dataset root')
    parser.add_argument('--sample_rate', type=int, default=16000,
                        help='Audio sample rate')
    parser.add_argument('--max_length', type=int, default=48000,
                        help='Max audio length in samples (3 seconds at 16kHz)')
    
    # Evaluation parameters
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for evaluation')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Directory to save evaluation results')
    parser.add_argument('--no_cuda', action='store_true',
                        help='Disable CUDA even if available')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='Number of worker threads for dataloading')
    
    args = parser.parse_args()
    evaluate_model(args)


if __name__ == '__main__':
    main() 