"""
Deep Learning Model Training Script
Trains EfficientNet model to classify real vs AI-generated images
"""

import os
import json
import numpy as np
from PIL import Image
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving plots
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, auc, classification_report
from sklearn.preprocessing import label_binarize

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from efficientnet_pytorch import EfficientNet
import kornia.augmentation as K

from core.config import Config

DATASET_DIR = Config.DATASETS_FOLDER
MODEL_SAVE_PATH = os.path.join(Config.MODELS_FOLDER, 'best_model.pth')
TRAINING = True

class ImageDataset(Dataset):
    """Dataset for real vs AI images"""
    
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

class AIImageClassifier(nn.Module):
    """EfficientNet-based classifier for AI image detection"""
    
    def __init__(self, num_classes=5, pretrained=True):
        super(AIImageClassifier, self).__init__()
        
        # Load pre-trained EfficientNet (Default: B3 for better capacity)
        if pretrained:
            self.backbone = EfficientNet.from_pretrained('efficientnet-b1')
        else:
            self.backbone = EfficientNet.from_name('efficientnet-b1')
        
        # Get the number of features from the last layer
        num_features = self.backbone._fc.in_features
        
        # Replace classifier
        self.backbone._fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, x):
        return self.backbone(x)

class ModelTrainer:
    """Handles model training and evaluation"""
    
    def __init__(self, model, device='cpu'):
        self.model = model.to(device)
        self.device = device
        self.class_names = ['Real', 'Stable_Diffusion', 'Midjourney', 'DALLE', 'Unknown']
        
    def train(self, train_loader, val_loader, epochs=10, learning_rate=0.001):
        """Train the model"""
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', 
                                                         factor=0.5, patience=3)
        
        best_val_loss = float('inf')
        train_losses = []
        val_losses = []
        train_accs = []
        val_accs = []
        
        for epoch in range(epochs):
            print(f"\nEpoch {epoch+1}/{epochs}")
            print("-" * 50)
            
            # Training phase
            train_loss, train_acc = self._train_epoch(train_loader, criterion, optimizer)
            train_losses.append(train_loss)
            train_accs.append(train_acc)
            
            # Validation phase
            val_loss, val_acc = self._validate_epoch(val_loader, criterion)
            val_losses.append(val_loss)
            val_accs.append(val_acc)
            
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            
            # Learning rate scheduling
            scheduler.step(val_loss)
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save_model('best_model.pth')
                print("✓ Saved best model")
        
        # Save training history
        history = {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'train_accs': train_accs,
            'val_accs': val_accs
        }
        
        return history
    
    def _train_epoch(self, train_loader, criterion, optimizer):
        """Train for one epoch using Mixed Precision and FGSM Adversarial Auto-Defense"""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())
        
        # JPEG Compression augmentation inside the training loop so it applies on GPU batch
        jpeg_aug = K.RandomJPEG(jpeg_quality=(30., 90.), p=0.5).to(self.device)
        
        pbar = tqdm(train_loader, desc="Training")
        for inputs, labels in pbar:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            
            # Apply GPU-based augmentations (like JPEG compression)
            with torch.no_grad():
                inputs = jpeg_aug(inputs)
            
            # --- START ADVERSARIAL TRAINING (FGSM) ---
            # Tell PyTorch to compute gradients for the inputs
            inputs.requires_grad = True
            
            # 1. Clean image forward pass
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                outputs_clean = self.model(inputs)
                loss_clean = criterion(outputs_clean, labels)
            
            # 2. Calculate gradients of the loss with respect to the input pixels
            optimizer.zero_grad()
            scaler.scale(loss_clean).backward(retain_graph=True)
            
            # 3. Create adversarial images by adding standard FGSM noise
            with torch.no_grad():
                epsilon = 0.03 # Perturbation strength
                inputs_adv = inputs + epsilon * inputs.grad.sign()
                inputs_adv = torch.clamp(inputs_adv, -3.0, 3.0) # Prevent exploding values
            
            # Reset gradients and detach inputs to perform the final backward pass cleanly
            inputs.requires_grad = False
            optimizer.zero_grad()
            
            # 4. Adversarial image forward pass
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                outputs_adv = self.model(inputs_adv)
                loss_adv = criterion(outputs_adv, labels)
                
            # 5. Combine losses to simultaneously learn clean features and adversarial robustness
            total_loss = (loss_clean + loss_adv) / 2.0
            
            # Mixed Precision Final Backward pass
            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            # Metrics (Based on clean images)
            running_loss += total_loss.item()
            _, predicted = outputs_clean.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            pbar.set_postfix({'loss': running_loss/(pbar.n+1), 
                            'acc': 100.*correct/total})
        
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100. * correct / total
        
        return epoch_loss, epoch_acc
    
    def _validate_epoch(self, val_loader, criterion):
        """Validate for one epoch"""
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc="Validation"):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        epoch_loss = running_loss / len(val_loader)
        epoch_acc = 100. * correct / total
        
        return epoch_loss, epoch_acc
    
    def predict(self, image_path):
        """Predict class for a single image"""
        self.model.eval()
        
        transform = get_transforms(train=False)
        image = Image.open(image_path).convert('RGB')
        image = transform(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(image)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, predicted = probabilities.max(1)
        
        return {
            'class': self.class_names[predicted.item()],
            'confidence': confidence.item() * 100,
            'probabilities': {
                self.class_names[i]: probabilities[0][i].item() * 100 
                for i in range(len(self.class_names))
            }
        }
    
    def save_model(self, path):
        """Save model checkpoint"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'class_names': self.class_names
        }, path)
    
    def load_model(self, path):
        """Load model checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.class_names = checkpoint['class_names']
        self.model.eval()

def get_transforms(train=True):
    """Get image transforms"""
    # Increased input size for B3 model
    img_size = 300
    
    if train:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=5)], p=0.3),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])

def prepare_dataset(data_dir):
    """
    Prepare dataset from directory structure:
    data_dir/
        real/
        stable_diffusion/
        midjourney/
        dalle/
        unknown/
    """
    image_paths = []
    labels = []
    
    class_names = ['real', 'stable_diffusion', 'midjourney', 'dalle', 'unknown']
    
    for class_idx, class_name in enumerate(class_names):
        class_dir = os.path.join(data_dir, class_name)
        if not os.path.exists(class_dir):
            print(f"Warning: {class_dir} not found")
            continue
            
        for img_name in os.listdir(class_dir):
            if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_paths.append(os.path.join(class_dir, img_name))
                labels.append(class_idx)
    
    return image_paths, labels

def generate_training_plots(history, trainer, val_loader, class_names, save_dir):
    """Generate ROC curve, accuracy/loss curves, and confusion matrix after training"""
    os.makedirs(save_dir, exist_ok=True)
    
    # --- 1. ACCURACY & LOSS CURVES ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    epochs_range = range(1, len(history['train_losses']) + 1)
    
    ax1.plot(epochs_range, history['train_losses'], 'b-o', label='Train Loss')
    ax1.plot(epochs_range, history['val_losses'], 'r-o', label='Validation Loss')
    ax1.set_title('Training & Validation Loss', fontsize=14)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(epochs_range, history['train_accs'], 'b-o', label='Train Accuracy')
    ax2.plot(epochs_range, history['val_accs'], 'r-o', label='Validation Accuracy')
    ax2.set_title('Training & Validation Accuracy', fontsize=14)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'accuracy_loss_curves.png'), dpi=150)
    plt.close()
    print(f"[OK] Saved accuracy/loss curves to {save_dir}/accuracy_loss_curves.png")
    
    # --- 2. CONFUSION MATRIX ---
    trainer.model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for inputs, labels in tqdm(val_loader, desc="Generating Confusion Matrix"):
            inputs = inputs.to(trainer.device)
            outputs = trainer.model(inputs)
            probs = torch.softmax(outputs, dim=1)
            _, preds = outputs.max(1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    # Get unique classes present in the data
    unique_classes = np.unique(all_labels)
    present_class_names = [class_names[i] for i in unique_classes]
    
    cm = confusion_matrix(all_labels, all_preds, labels=unique_classes)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.set_title('Confusion Matrix', fontsize=14)
    plt.colorbar(im)
    
    ax.set_xticks(range(len(present_class_names)))
    ax.set_yticks(range(len(present_class_names)))
    ax.set_xticklabels(present_class_names, rotation=45, ha='right')
    ax.set_yticklabels(present_class_names)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    
    # Add text annotations
    for i in range(len(present_class_names)):
        for j in range(len(present_class_names)):
            ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                    color='white' if cm[i, j] > cm.max() / 2 else 'black')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'confusion_matrix.png'), dpi=150)
    plt.close()
    print(f"[OK] Saved confusion matrix to {save_dir}/confusion_matrix.png")
    
    # --- 3. ROC CURVE ---
    n_classes = len(unique_classes)
    if n_classes == 2:
        # Binary ROC
        fpr, tpr, _ = roc_curve(all_labels, all_probs[:, 1] if all_probs.shape[1] > 1 else all_probs[:, 0])
        roc_auc = auc(fpr, tpr)
        
        fig, ax = plt.subplots(figsize=(7, 6))
        ax.plot(fpr, tpr, 'b-', linewidth=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.5)
        ax.set_title('ROC Curve', fontsize=14)
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3)
    else:
        # Multi-class ROC (One-vs-Rest)
        labels_bin = label_binarize(all_labels, classes=unique_classes)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        colors = ['#667eea', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6']
        
        for i, cls_idx in enumerate(unique_classes):
            col_idx = list(unique_classes).index(cls_idx)
            fpr, tpr, _ = roc_curve(labels_bin[:, col_idx], all_probs[:, cls_idx])
            roc_auc = auc(fpr, tpr)
            color = colors[i % len(colors)]
            ax.plot(fpr, tpr, color=color, linewidth=2,
                    label=f'{class_names[cls_idx]} (AUC = {roc_auc:.4f})')
        
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.5)
        ax.set_title('ROC Curve (One-vs-Rest)', fontsize=14)
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'roc_curve.png'), dpi=150)
    plt.close()
    print(f"[OK] Saved ROC curve to {save_dir}/roc_curve.png")
    
    # --- 4. DATASET DISTRIBUTION PIE CHART ---
    unique, counts = np.unique(all_labels, return_counts=True)
    fig, ax = plt.subplots(figsize=(7, 6))
    wedges, texts, autotexts = ax.pie(
        counts, labels=[class_names[u] for u in unique],
        autopct='%1.1f%%', startangle=90,
        colors=['#667eea', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6'][:len(unique)]
    )
    ax.set_title('Dataset Class Distribution', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'dataset_distribution.png'), dpi=150)
    plt.close()
    print(f"[OK] Saved dataset distribution chart to {save_dir}/dataset_distribution.png")


def main():
    """Main training function"""
    
    # Configuration
    DATA_DIR = os.path.join(DATASET_DIR, 'train')
    BATCH_SIZE = 16
    EPOCHS = 10
    LEARNING_RATE = 0.001
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    MODELS_DIR = os.path.join(DATASET_DIR, '..', 'models')
    PLOTS_DIR = os.path.join(MODELS_DIR, 'plots')
    
    print(f"Using device: {DEVICE}")
    print(f"Training for {EPOCHS} epochs with batch size {BATCH_SIZE}")
    
    # Prepare dataset
    print("\nPreparing dataset...")
    image_paths, labels = prepare_dataset(DATA_DIR)
    
    if len(image_paths) == 0:
        print("Error: No images found in dataset directory!")
        print(f"Please organize your data in: {DATA_DIR}")
        print("Structure: data_dir/real/, data_dir/stable_diffusion/, etc.")
        return
    
    print(f"Found {len(image_paths)} images")
    
    # Split into train and validation
    split_idx = int(0.8 * len(image_paths))
    train_paths = image_paths[:split_idx]
    train_labels = labels[:split_idx]
    val_paths = image_paths[split_idx:]
    val_labels = labels[split_idx:]
    
    print(f"Training samples: {len(train_paths)}")
    print(f"Validation samples: {len(val_paths)}")
    
    # Create datasets and loaders
    train_dataset = ImageDataset(train_paths, train_labels, 
                                 transform=get_transforms(train=True))
    val_dataset = ImageDataset(val_paths, val_labels, 
                               transform=get_transforms(train=False))
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, 
                             shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, 
                           shuffle=False, num_workers=4)
    
    # Create model
    print("\nInitializing model...")
    model = AIImageClassifier(num_classes=5, pretrained=True)
    trainer = ModelTrainer(model, device=DEVICE)
    
    # Train model
    print("\nStarting training...")
    history = trainer.train(train_loader, val_loader, 
                           epochs=EPOCHS, learning_rate=LEARNING_RATE)
    
    # Save final model
    os.makedirs(MODELS_DIR, exist_ok=True)
    trainer.save_model(os.path.join(MODELS_DIR, 'final_model.pth'))
    
    # Save training history
    with open(os.path.join(MODELS_DIR, 'training_history.json'), 'w') as f:
        json.dump(history, f)
    
    # Generate evaluation plots
    print("\nGenerating evaluation plots...")
    generate_training_plots(
        history, trainer, val_loader,
        trainer.class_names, PLOTS_DIR
    )
    
    print("\n[OK] Training complete!")
    print(f"Best model saved to: {MODELS_DIR}/best_model.pth")
    print(f"Final model saved to: {MODELS_DIR}/final_model.pth")
    print(f"Evaluation plots saved to: {PLOTS_DIR}/")

if __name__ == '__main__':
    main()
