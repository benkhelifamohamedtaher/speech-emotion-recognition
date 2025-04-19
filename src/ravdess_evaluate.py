#!/usr/bin/env python3
"""
Evaluation script for RAVDESS speech emotion recognition model
Tests the model on the test set and reports detailed metrics
"""

import os
import sys
import json
import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.metrics import precision_recall_fscore_support

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

# Import our modules
from ravdess_model import AdvancedSpeechEmotionRecognizer
from ravdess_dataset import RAVDESSDataset

def load_model(model_path, device='cuda'):
    """
    Load the trained model from checkpoint
    """
    print(f"Loading model from {model_path}")
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location='cpu')
    
    # Get model arguments
    args = checkpoint['args'] if 'args' in checkpoint else {}
    
    # Get config file if it exists
    config_path = os.path.join(os.path.dirname(model_path), 'config.json')
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = json.load(f)
    else:
        config = {}
    
    # Combine config and args
    combined_config = {**config, **args}
    
    # Get number of classes
    num_classes = combined_config.get('num_classes', 8)
    
    # Get other model parameters
    wav2vec_model = combined_config.get('wav2vec_model', 'facebook/wav2vec2-base')
    context_layers = combined_config.get('context_layers', 2)
    attention_heads = combined_config.get('attention_heads', 4)
    dropout_rate = combined_config.get('dropout_rate', 0.3)
    use_gender_branch = combined_config.get('use_gender_branch', True)
    use_spectrogram_branch = combined_config.get('use_spectrogram_branch', True)
    
    # Create model
    model = AdvancedSpeechEmotionRecognizer(
        num_emotions=num_classes,
        wav2vec_model_name=wav2vec_model,
        context_layers=context_layers,
        attention_heads=attention_heads,
        dropout_rate=dropout_rate,
        use_gender_branch=use_gender_branch,
        use_spectrogram_branch=use_spectrogram_branch
    )
    
    # Load model weights
    model.load_state_dict(checkpoint['model'])
    
    # Move model to device
    model = model.to(device)
    model.eval()
    
    return model, combined_config


def evaluate_model(model, dataset_root, device='cuda', batch_size=32, subset=None):
    """
    Evaluate the model on the test set
    """
    # Create test dataset
    test_dataset = RAVDESSDataset(
        root_dir=dataset_root,
        split='test',
        sample_rate=16000,
        max_duration=5.0,
        transforms=None,
        audio_only=True,
        speech_only=True,
        cache_waveforms=False,
        subset=subset
    )
    
    # Create test dataloader
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=RAVDESSDataset.collate_fn
    )
    
    # Get emotion mapping
    emotion_mapping = RAVDESSDataset.get_emotion_mapping()
    
    # Initialize variables
    all_preds = []
    all_targets = []
    all_probs = []
    all_files = []
    
    # Evaluate model
    print(f"Evaluating model on {len(test_dataset)} test samples...")
    
    with torch.no_grad():
        for batch in tqdm(test_loader):
            # Get data
            waveforms = batch['waveform'].to(device)
            emotion_targets = batch['emotion'].to(device)
            metadata = batch['metadata']
            
            # Forward pass
            outputs = model(waveforms)
            emotion_logits = outputs['emotion_logits']
            emotion_probs = F.softmax(emotion_logits, dim=1)
            
            # Get predictions
            emotion_preds = torch.argmax(emotion_probs, dim=1)
            
            # Store results
            all_preds.extend(emotion_preds.cpu().numpy())
            all_targets.extend(emotion_targets.cpu().numpy())
            all_probs.extend(emotion_probs.cpu().numpy())
            all_files.extend([m['file_path'] for m in metadata])
    
    # Convert to numpy arrays
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    all_probs = np.array(all_probs)
    
    # Calculate accuracy
    accuracy = accuracy_score(all_targets, all_preds)
    print(f"Test accuracy: {accuracy:.4f} ({100 * accuracy:.2f}%)")
    
    # Calculate confusion matrix
    cm = confusion_matrix(all_targets, all_preds)
    
    # Get class names
    class_names = [emotion_mapping[i] for i in range(len(emotion_mapping))
                   if i in emotion_mapping]
    
    # Calculate precision, recall, and F1 score
    precision, recall, f1, support = precision_recall_fscore_support(
        all_targets, all_preds, average=None
    )
    
    # Print per-class metrics
    print("\nPer-class metrics:")
    print(f"{'Emotion':<10} {'Precision':<10} {'Recall':<10} {'F1':<10} {'Support':<10}")
    print("-" * 50)
    for i, class_name in enumerate(class_names):
        print(f"{class_name:<10} {precision[i]:.4f}      {recall[i]:.4f}     {f1[i]:.4f}     {support[i]}")
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(all_targets, all_preds, target_names=class_names))
    
    # Plot confusion matrix
    plot_confusion_matrix(cm, class_names)
    
    # Plot ROC curves
    plot_roc_curves(all_targets, all_probs, class_names)
    
    # Return results
    return {
        'accuracy': accuracy,
        'confusion_matrix': cm,
        'class_names': class_names,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'support': support,
        'all_preds': all_preds,
        'all_targets': all_targets,
        'all_probs': all_probs,
        'all_files': all_files
    }


def plot_confusion_matrix(cm, class_names):
    """
    Plot confusion matrix
    """
    plt.figure(figsize=(10, 8))
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    
    # Save figure
    plt.savefig('confusion_matrix.png')
    print("Confusion matrix saved to confusion_matrix.png")
    
    # Close figure
    plt.close()


def plot_roc_curves(all_targets, all_probs, class_names):
    """
    Plot ROC curves
    """
    from sklearn.metrics import roc_curve, auc
    from sklearn.preprocessing import label_binarize
    
    # Binarize the labels
    n_classes = len(class_names)
    
    try:
        y_bin = label_binarize(all_targets, classes=range(n_classes))
        
        # Compute ROC curve and ROC area for each class
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_bin[:, i], all_probs[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
        
        # Plot ROC curves
        plt.figure(figsize=(10, 8))
        
        for i, color, cls in zip(range(n_classes), 
                               plt.cm.rainbow(np.linspace(0, 1, n_classes)), 
                               class_names):
            plt.plot(fpr[i], tpr[i], color=color, lw=2,
                    label=f'{cls} (AUC = {roc_auc[i]:.2f})')
        
        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curves')
        plt.legend(loc="lower right")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        
        # Save figure
        plt.savefig('roc_curves.png')
        print("ROC curves saved to roc_curves.png")
        
        # Close figure
        plt.close()
    except Exception as e:
        print(f"Error plotting ROC curves: {e}")


def analyze_errors(results):
    """
    Analyze errors in the model predictions
    """
    # Get data
    all_preds = results['all_preds']
    all_targets = results['all_targets']
    all_files = results['all_files']
    class_names = results['class_names']
    
    # Find errors
    errors = np.where(all_preds != all_targets)[0]
    
    # Print error summary
    print(f"\nFound {len(errors)} errors out of {len(all_targets)} samples ({100 * len(errors) / len(all_targets):.2f}%)")
    
    # Analyze most common error patterns
    error_pairs = [(all_targets[i], all_preds[i]) for i in errors]
    error_counts = {}
    
    for true, pred in error_pairs:
        pair = (true, pred)
        if pair not in error_counts:
            error_counts[pair] = 0
        error_counts[pair] += 1
    
    # Sort by frequency
    sorted_errors = sorted(error_counts.items(), key=lambda x: x[1], reverse=True)
    
    # Print most common errors
    print("\nMost common errors:")
    for (true, pred), count in sorted_errors[:10]:
        print(f"  {class_names[true]} misclassified as {class_names[pred]}: {count} times")
    
    # Return detailed error information
    return {
        'error_indices': errors,
        'error_files': [all_files[i] for i in errors],
        'error_true': [all_targets[i] for i in errors],
        'error_pred': [all_preds[i] for i in errors],
        'error_counts': error_counts
    }


def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description="Evaluate RAVDESS emotion recognition model")
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to the trained model checkpoint')
    parser.add_argument('--dataset_root', type=str, required=True,
                        help='Path to the RAVDESS dataset directory')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use for evaluation')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for evaluation')
    parser.add_argument('--emotion_subset', type=str, default=None,
                        choices=['basic4', 'basic6', None],
                        help='Subset of emotions to use')
    
    args = parser.parse_args()
    
    # Load model
    model, config = load_model(args.model_path, args.device)
    
    # Get emotion subset from config if not specified
    if args.emotion_subset is None and 'emotion_subset' in config:
        args.emotion_subset = config['emotion_subset']
    
    # Evaluate model
    results = evaluate_model(
        model, 
        args.dataset_root, 
        args.device, 
        args.batch_size,
        args.emotion_subset
    )
    
    # Analyze errors
    error_analysis = analyze_errors(results)
    
    print("\nEvaluation completed!")


if __name__ == "__main__":
    main() 