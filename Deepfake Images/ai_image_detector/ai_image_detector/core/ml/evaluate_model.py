"""
Model Evaluation Script
Evaluates trained model on test dataset and generates metrics
"""

import torch
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import os
import sys

sys.path.append(os.path.dirname(__file__))
import cv2

from core.ml.train_model import AIImageClassifier, get_transforms
from core.analyzers.frequency_analyzer import FrequencyAnalyzer
from core.analyzers.noise_analyzer import NoiseAnalyzer
from core.analyzers.metadata_analyzer import MetadataAnalyzer
from core.analyzers.pixel_analyzer import PixelStatisticsAnalyzer
from core.config import Config

WEIGHTS = Config.WEIGHTS
DATASET_DIR = Config.DATASETS_FOLDER
MODEL_SAVE_PATH = os.path.join(Config.MODELS_FOLDER, 'best_model.pth')

def evaluate_model(model_path, test_data_dir, device='cpu'):
    """
    Evaluate model on test dataset
    
    Args:
        model_path: Path to trained model
        test_data_dir: Directory containing test data
        device: 'cpu' or 'cuda'
    """
    
    print("="*70)
    print("MODEL EVALUATION")
    print("="*70)
    
    # Load model
    print("\nLoading model...")
    model = AIImageClassifier(num_classes=5, pretrained=False)
    trainer = ModelTrainer(model, device=device)
    
    try:
        trainer.load_model(model_path)
        print(f"✓ Model loaded from: {model_path}")
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        return
    
    # Prepare test dataset
    print("\nPreparing test dataset...")
    test_paths, test_labels = prepare_dataset(test_data_dir)
    
    if len(test_paths) == 0:
        print("✗ No test images found!")
        print(f"Please add test images to: {test_data_dir}")
        return
    
    print(f"✓ Found {len(test_paths)} test images")
    
    test_dataset = ImageDataset(test_paths, test_labels, 
                                transform=get_transforms(train=False))
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    
    # Evaluate
    print("\nEvaluating model...")
    model.eval()
    
    all_predictions = []
    all_labels = []
    all_probabilities = []
    
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Testing"):
            images = images.to(device)
            outputs = model(images)
            probabilities = torch.softmax(outputs, dim=1)
            _, predictions = outputs.max(1)
            
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
    
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    all_probabilities = np.array(all_probabilities)
    
    # Calculate metrics
    accuracy = np.mean(all_predictions == all_labels) * 100
    
    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)
    print(f"\nOverall Accuracy: {accuracy:.2f}%")
    
    # Classification report
    print("\n" + "-"*70)
    print("Classification Report:")
    print("-"*70)
    print(classification_report(all_labels, all_predictions, 
                                target_names=trainer.class_names,
                                digits=3))
    
    # Confusion matrix
    print("\n" + "-"*70)
    print("Confusion Matrix:")
    print("-"*70)
    cm = confusion_matrix(all_labels, all_predictions)
    print(cm)
    
    # Per-class accuracy
    print("\n" + "-"*70)
    print("Per-Class Accuracy:")
    print("-"*70)
    for i, class_name in enumerate(trainer.class_names):
        class_mask = all_labels == i
        if class_mask.sum() > 0:
            class_acc = np.mean(all_predictions[class_mask] == all_labels[class_mask]) * 100
            print(f"{class_name:20s}: {class_acc:6.2f}%")
        else:
            print(f"{class_name:20s}: No samples")
    
    # Plot confusion matrix
    try:
        plot_confusion_matrix(cm, trainer.class_names)
        print("\n✓ Confusion matrix saved to: confusion_matrix.png")
    except Exception as e:
        print(f"\n✗ Could not save confusion matrix: {e}")
    
    # Calculate confidence statistics
    print("\n" + "-"*70)
    print("Confidence Statistics:")
    print("-"*70)
    
    max_probs = np.max(all_probabilities, axis=1)
    correct_mask = all_predictions == all_labels
    
    print(f"Average confidence (all):     {np.mean(max_probs)*100:.2f}%")
    print(f"Average confidence (correct): {np.mean(max_probs[correct_mask])*100:.2f}%")
    print(f"Average confidence (wrong):   {np.mean(max_probs[~correct_mask])*100:.2f}%")
    
    print("\n" + "="*70)

def plot_confusion_matrix(cm, class_names):
    """Plot and save confusion matrix"""
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()

def main():
    """Main evaluation function"""
    
    # Configuration
    MODEL_PATH = '../models/best_model.pth'
    TEST_DATA_DIR = '../datasets/test'
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"Device: {DEVICE}")
    
    # Check if model exists
    if not os.path.exists(MODEL_PATH):
        print(f"\n✗ Model not found at: {MODEL_PATH}")
        print("Please train the model first using train_model.py")
        return
    
    # Check if test data exists
    if not os.path.exists(TEST_DATA_DIR):
        print(f"\n✗ Test directory not found: {TEST_DATA_DIR}")
        print("Please create test dataset")
        return
    
    # Run evaluation
    evaluate_model(MODEL_PATH, TEST_DATA_DIR, DEVICE)

if __name__ == '__main__':
    main()
