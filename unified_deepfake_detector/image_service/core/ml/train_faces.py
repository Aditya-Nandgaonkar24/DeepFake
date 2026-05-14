"""
Training script for 140K Real & Fake Faces dataset.
Binary classification: Real (0) vs Fake (1)
Images are 256x256 high resolution.
"""
import os
import sys
import json
import numpy as np
from PIL import Image
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, auc

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from efficientnet_pytorch import EfficientNet

# Speed optimization: use fastest convolution algorithms
torch.backends.cudnn.benchmark = True

# ====================== CONFIG ======================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))
DATASET_DIR = os.path.join(PROJECT_DIR, 'datasets', 'faces', 'real_vs_fake', 'real-vs-fake')
MODELS_DIR = os.path.join(PROJECT_DIR, 'models')
PLOTS_DIR = os.path.join(MODELS_DIR, 'plots')

# ====================== MODEL ======================
class FaceDetector(nn.Module):
    def __init__(self, num_classes=2, pretrained=True):
        super().__init__()
        if pretrained:
            self.backbone = EfficientNet.from_pretrained('efficientnet-b0')
        else:
            self.backbone = EfficientNet.from_name('efficientnet-b0')
        
        in_features = self.backbone._fc.in_features
        self.backbone._fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        return self.backbone(x)

# ====================== DATASET ======================
class FaceDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        try:
            image = Image.open(self.image_paths[idx]).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image, self.labels[idx]
        except:
            # Return a blank image on error
            image = Image.new('RGB', (256, 256), (128, 128, 128))
            if self.transform:
                image = self.transform(image)
            return image, self.labels[idx]

def get_transforms(train=True):
    img_size = 224  # Reduced from 256 to save VRAM
    if train:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

# ====================== DATA LOADING ======================
def load_split(split_dir, max_per_class=None):
    """Load images from a split directory (train/test/valid)"""
    paths = []
    labels = []
    
    # Try both folder naming conventions
    real_dir = os.path.join(split_dir, 'real')
    fake_dir = os.path.join(split_dir, 'fake')
    
    if not os.path.exists(real_dir):
        real_dir = os.path.join(split_dir, '0_real')
        fake_dir = os.path.join(split_dir, '1_fake')
    
    for class_label, class_dir in [(0, real_dir), (1, fake_dir)]:
        if not os.path.exists(class_dir):
            print(f"  Warning: {class_dir} not found")
            continue
        
        files = [f for f in os.listdir(class_dir) 
                 if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        if max_per_class and len(files) > max_per_class:
            import random
            random.shuffle(files)
            files = files[:max_per_class]
        
        for fname in files:
            paths.append(os.path.join(class_dir, fname))
            labels.append(class_label)
    
    return paths, labels

# ====================== TRAINING ======================
CHECKPOINT_PATH = os.path.join(MODELS_DIR, 'checkpoint.pth')

def train_model(train_loader, val_loader, device, epochs=3, resume=False):
    model = FaceDetector(num_classes=2, pretrained=True).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2)
    scaler = torch.amp.GradScaler('cuda', enabled=(device == 'cuda'))
    
    best_val_loss = float('inf')
    start_epoch = 0
    history = {'train_losses': [], 'val_losses': [], 'train_accs': [], 'val_accs': []}
    
    # Resume from checkpoint if requested
    if resume and os.path.exists(CHECKPOINT_PATH):
        print(f"[RESUME] Loading checkpoint from {CHECKPOINT_PATH}")
        ckpt = torch.load(CHECKPOINT_PATH, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        start_epoch = ckpt['epoch'] + 1
        best_val_loss = ckpt['best_val_loss']
        history = ckpt.get('history', history)
        print(f"[RESUME] Continuing from epoch {start_epoch + 1}, best_val_loss={best_val_loss:.4f}")
    
    total_epochs = start_epoch + epochs
    
    for epoch in range(start_epoch, total_epochs):
        # --- TRAIN ---
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{total_epochs} [Train]")
        for inputs, labels in pbar:
            inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)  # Faster than zero_grad()
            
            with torch.amp.autocast('cuda', enabled=(device == 'cuda')):
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            pbar.set_postfix(loss=f"{loss.item():.3f}", acc=f"{100.*correct/total:.1f}%")
        
        train_loss = running_loss / len(train_loader)
        train_acc = 100. * correct / total
        
        # --- VALIDATE ---
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{total_epochs} [Val]"):
                inputs, labels = inputs.to(device), labels.to(device)
                with torch.amp.autocast('cuda', enabled=(device == 'cuda')):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
        
        val_loss /= len(val_loader)
        val_acc = 100. * val_correct / val_total
        
        history['train_losses'].append(train_loss)
        history['val_losses'].append(val_loss)
        history['train_accs'].append(train_acc)
        history['val_accs'].append(val_acc)
        
        print(f"\nEpoch {epoch+1}: Train Loss={train_loss:.4f} Acc={train_acc:.1f}% | Val Loss={val_loss:.4f} Acc={val_acc:.1f}%")
        
        scheduler.step(val_loss)
        
        # Save best model (for inference)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            os.makedirs(MODELS_DIR, exist_ok=True)
            torch.save({
                'model_state_dict': model.state_dict(),
                'class_names': ['Real', 'Fake']
            }, os.path.join(MODELS_DIR, 'best_model.pth'))
            print(f"  [SAVED] Best model (val_loss={val_loss:.4f})")
        
        # Save full checkpoint (for resuming training)
        os.makedirs(MODELS_DIR, exist_ok=True)
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'epoch': epoch,
            'best_val_loss': best_val_loss,
            'history': history,
            'class_names': ['Real', 'Fake']
        }, CHECKPOINT_PATH)
    
    return model, history

# ====================== EVALUATION PLOTS ======================
def generate_plots(model, val_loader, history, device):
    os.makedirs(PLOTS_DIR, exist_ok=True)
    
    # 1. Accuracy & Loss curves
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    epochs_range = range(1, len(history['train_losses']) + 1)
    
    ax1.plot(epochs_range, history['train_losses'], 'b-o', label='Train Loss')
    ax1.plot(epochs_range, history['val_losses'], 'r-o', label='Val Loss')
    ax1.set_title('Loss Curve'); ax1.set_xlabel('Epoch'); ax1.legend(); ax1.grid(True, alpha=0.3)
    
    ax2.plot(epochs_range, history['train_accs'], 'b-o', label='Train Acc')
    ax2.plot(epochs_range, history['val_accs'], 'r-o', label='Val Acc')
    ax2.set_title('Accuracy Curve'); ax2.set_xlabel('Epoch'); ax2.set_ylabel('%'); ax2.legend(); ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'accuracy_loss_curves.png'), dpi=150)
    plt.close()
    print("[OK] Saved accuracy_loss_curves.png")
    
    # 2. Confusion Matrix + ROC
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    
    with torch.no_grad():
        for inputs, labels in tqdm(val_loader, desc="Evaluating"):
            inputs = inputs.to(device)
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)
            _, preds = outputs.max(1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())  # P(Fake)
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(cm, cmap=plt.cm.Blues)
    ax.set_title('Confusion Matrix'); plt.colorbar(im)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                    color='white' if cm[i, j] > cm.max()/2 else 'black', fontsize=16)
    ax.set_xticks([0,1]); ax.set_yticks([0,1])
    ax.set_xticklabels(['Real', 'Fake']); ax.set_yticklabels(['Real', 'Fake'])
    ax.set_xlabel('Predicted'); ax.set_ylabel('True')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'confusion_matrix.png'), dpi=150)
    plt.close()
    print("[OK] Saved confusion_matrix.png")
    
    # ROC Curve
    fpr, tpr, _ = roc_curve(all_labels, all_probs)
    roc_auc = auc(fpr, tpr)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, 'b-', linewidth=2, label=f'ROC (AUC = {roc_auc:.4f})')
    ax.plot([0,1], [0,1], 'k--', alpha=0.5)
    ax.set_title('ROC Curve'); ax.set_xlabel('FPR'); ax.set_ylabel('TPR')
    ax.legend(loc='lower right'); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'roc_curve.png'), dpi=150)
    plt.close()
    print(f"[OK] Saved roc_curve.png (AUC = {roc_auc:.4f})")
    
    # Dataset distribution pie chart
    unique, counts = np.unique(all_labels, return_counts=True)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.pie(counts, labels=['Real', 'Fake'], autopct='%1.1f%%', 
           colors=['#2ecc71', '#e74c3c'], startangle=90)
    ax.set_title('Dataset Distribution')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'dataset_distribution.png'), dpi=150)
    plt.close()
    print("[OK] Saved dataset_distribution.png")

# ====================== MAIN ======================
def main():
    import argparse
    parser = argparse.ArgumentParser(description='Train Face Detector')
    parser.add_argument('--resume', action='store_true', help='Resume training from last checkpoint')
    parser.add_argument('--epochs', type=int, default=1, help='Number of epochs to train (default: 1)')
    args = parser.parse_args()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    print(f"Dataset: {DATASET_DIR}")
    if args.resume:
        print("[MODE] Resume training from checkpoint")
    
    EPOCHS = args.epochs
    BATCH_SIZE = 8  # Small batch for 4GB VRAM
    MAX_PER_CLASS = None  # Set to e.g. 5000 for quick test
    
    # Check dataset structure
    train_dir = os.path.join(DATASET_DIR, 'train')
    val_dir = os.path.join(DATASET_DIR, 'valid')
    test_dir = os.path.join(DATASET_DIR, 'test')
    
    if not os.path.exists(train_dir):
        # Try alternative structure
        for d in os.listdir(DATASET_DIR):
            print(f"  Found: {d}")
        print(f"\nExpected: {train_dir}")
        print("Please check the dataset structure.")
        return
    
    # Load data
    print("\nLoading training data...")
    train_paths, train_labels = load_split(train_dir, MAX_PER_CLASS)
    print(f"  Train: {len(train_paths)} images")
    
    print("Loading validation data...")
    val_paths, val_labels = load_split(val_dir, MAX_PER_CLASS)
    print(f"  Val: {len(val_paths)} images")
    
    # Create data loaders
    train_dataset = FaceDataset(train_paths, train_labels, get_transforms(train=True))
    val_dataset = FaceDataset(val_paths, val_labels, get_transforms(train=False))
    
    num_w = 0  # Set to 0 to avoid Windows multiprocessing resource errors (WinError 1450)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=num_w, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=num_w, pin_memory=True)
    
    # Train
    print(f"\nTraining for {EPOCHS} epochs{'(resume)' if args.resume else ''}...")
    model, history = train_model(train_loader, val_loader, device, EPOCHS, resume=args.resume)
    
    # Save history
    os.makedirs(MODELS_DIR, exist_ok=True)
    with open(os.path.join(MODELS_DIR, 'training_history.json'), 'w') as f:
        json.dump(history, f)
    
    # Generate plots
    print("\nGenerating evaluation plots...")
    generate_plots(model, val_loader, history, device)
    
    print(f"\n{'='*50}")
    print(f"[DONE] Training complete!")
    print(f"  Model: {MODELS_DIR}/best_model.pth")
    print(f"  Plots: {PLOTS_DIR}/")
    print(f"{'='*50}")

if __name__ == '__main__':
    main()
