import os
import signal
import sys
import numpy as np
import pandas as pd
import argparse
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from models.fusion import SpatialTemporalFusion
from utils.dataset import DeepfakeDataset
from sklearn.metrics import f1_score as compute_f1
from config import (SEQUENCE_LENGTH, DEFAULT_BATCH_SIZE, DEFAULT_ACCUMULATE_STEPS,
                    DEFAULT_LR, DEFAULT_EPOCHS, DEFAULT_PATIENCE, DEFAULT_WEIGHT_DECAY,
                    GRAD_CLIP_MAX_NORM, NUM_WORKERS, DATASET_PREPROCESSED, LOG_DIR,
                    USE_AUDIO_BRANCH)

# Global flag for graceful exit
pause_training = False

def signal_handler(sig, frame):
    global pause_training
    if not pause_training:
        print("\n[INFO] Pause command (Ctrl+C) received! The system will finish the current epoch, save the checkpoint, and then exit gracefully...")
        pause_training = True
    else:
        print("\n[WARNING] Force quitting immediately! State may be lost.")
        sys.exit(1)

def plot_metrics(history, output_dir):
    epochs = range(1, len(history['train_loss']) + 1)
    
    plt.figure(figsize=(16, 5))
    
    # Loss plot
    plt.subplot(1, 3, 1)
    plt.plot(epochs, history['train_loss'], label='Train Loss', color='#1f77b4', marker='o', linewidth=2)
    plt.plot(epochs, history['val_loss'], label='Val Loss', color='#ff7f0e', marker='x', linewidth=2)
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # Accuracy plot
    plt.subplot(1, 3, 2)
    if 'train_acc' in history and len(history['train_acc']) > 0:
        plt.plot(epochs, history['train_acc'], label='Train Accuracy', color='#1f77b4', marker='o', linewidth=2)
    plt.plot(epochs, history['val_acc'], label='Val Accuracy', color='#2ca02c', marker='s', linewidth=2)
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # F1 Score plot
    plt.subplot(1, 3, 3)
    plt.plot(epochs, history['val_f1'], label='Val F1', color='#9467bd', marker='D', linewidth=2)
    plt.title('F1 Score')
    plt.xlabel('Epoch')
    plt.ylabel('F1')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    # Save the plot
    plot_path = os.path.join(output_dir, 'training_evaluation_plots.png')
    plt.savefig(plot_path, dpi=300)
    plt.close()

def train_model(args):
    global pause_training
    
    # Register Ctrl+C interception for graceful pausing
    signal.signal(signal.SIGINT, signal_handler)
    
    # Fix #7: Reproducibility seeding
    torch.manual_seed(42)
    np.random.seed(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Executing Temporal-Spatial AI Processor on: {device}")

    train_dataset = DeepfakeDataset(os.path.join(args.data_dir, 'train'), sequence_length=SEQUENCE_LENGTH, phase='train')
    val_dataset = DeepfakeDataset(os.path.join(args.data_dir, 'val'), sequence_length=SEQUENCE_LENGTH, phase='val')

    # Boosted workers + pin_memory for faster CPU->GPU transfer
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

    model = SpatialTemporalFusion(
        seq_length=SEQUENCE_LENGTH,
        freeze_spatial=True,
        use_audio=args.use_audio,
    ).to(device)
    
    # Dynamic class imbalance handling: compute pos_weight from actual dataset
    num_real = sum(1 for l in train_dataset.labels if l == 0.0)
    num_fake = sum(1 for l in train_dataset.labels if l == 1.0)
    if num_fake > 0 and num_real > 0:
        pw = num_real / num_fake
        print(f"[INFO] Dataset balance: {num_real} real, {num_fake} fake => pos_weight={pw:.4f}")
    else:
        pw = 1.0
    pos_weight = torch.tensor([pw]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=DEFAULT_WEIGHT_DECAY)
    
    # LR warmup for 1 epoch before ReduceLROnPlateau to prevent catastrophic forgetting
    warmup_iters = max(1, len(train_loader) // args.accumulate_steps)
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, total_iters=warmup_iters)
    plateau_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=2, factor=0.5)
    scaler = torch.amp.GradScaler(device)

    start_epoch = 0
    best_val_f1 = 0.0
    early_stop_patience = args.patience
    epochs_no_improve = 0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [], 'val_f1': []}

    # Resumption Strategy
    if args.resume and os.path.exists(args.resume):
        print(f"Loading state dict from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        plateau_scheduler.load_state_dict(checkpoint['plateau_scheduler_state_dict'])
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        start_epoch = checkpoint['epoch'] + 1
        best_val_f1 = checkpoint.get('best_val_f1', 0.0)
        history = checkpoint.get('history', history)
        
        # Checking if user bumped the epochs parameter
        if args.epochs <= start_epoch:
            print(f"The model was already trained up to epoch {start_epoch}. Specify a higher --epochs to train further.")
            return

    os.makedirs(args.log_dir, exist_ok=True)

    for epoch in range(start_epoch, args.epochs):
        if pause_training:
            print("\nGraceful pause executed correctly before mapping a new epoch. Goodbye!")
            break
            
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Train]")
        optimizer.zero_grad()
        
        for step, (frames, labels, masks, audio_features) in enumerate(pbar):
            if pause_training:
                # Break mid-batch loops
                break
                
            frames = frames.to(device)
            labels = labels.to(device)
            masks = masks.to(device)
            audio_features = audio_features.to(device)
            
            with torch.amp.autocast(str(device)):
                logits = model(frames, mask=masks, audio_features=audio_features)
                loss = criterion(logits, labels)
                
                # Training Accuracy Calc for presentation
                probs_t = torch.sigmoid(logits)
                preds_t = probs_t >= 0.5
                train_correct += (preds_t == labels).sum().item()
                train_total += labels.size(0)
                
                # Gradient accumulation calculation for simulated large batches on cheap hardware
                loss = loss / args.accumulate_steps
                
            scaler.scale(loss).backward()
            
            if (step + 1) % args.accumulate_steps == 0 or (step + 1) == len(train_loader):
                # Fix v2#5: Gradient clipping to prevent GRU exploding gradients
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP_MAX_NORM)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                
                # Apply warmup only during first epoch, directly after optimizer steps
                if epoch == start_epoch:
                    warmup_scheduler.step()
            
            actual_loss = loss.item() * args.accumulate_steps
            train_loss += actual_loss
            pbar.set_postfix({'loss': actual_loss})
            
        if pause_training and step < len(train_loader) - 1:
            print("\n[INFO] Saving partial state dump for later resumption...")
            checkpoint = {
                'epoch': epoch - 1 if epoch > 0 else 0,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'plateau_scheduler_state_dict': plateau_scheduler.state_dict(),
                'scaler_state_dict': scaler.state_dict(),
                'best_val_f1': best_val_f1,
                'history': history,
                'args': vars(args)
            }
            torch.save(checkpoint, os.path.join(args.log_dir, 'latest_checkpoint.pth'))
            sys.exit(0)
            
        train_loss /= len(train_loader)
        train_acc = train_correct / train_total if train_total > 0 else 0
        
        # Validation Loop
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        all_val_preds = []
        all_val_labels = []
        
        with torch.no_grad():
            vbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Val]")
            for frames, labels, masks, audio_features in vbar:
                frames = frames.to(device)
                labels = labels.to(device)
                masks = masks.to(device)
                audio_features = audio_features.to(device)
                
                with torch.amp.autocast(str(device)):
                    logits = model(frames, mask=masks, audio_features=audio_features)
                    loss = criterion(logits, labels)
                    
                val_loss += loss.item()
                probs = torch.sigmoid(logits)
                preds = probs >= 0.5
                correct += (preds == labels).sum().item()
                total += labels.size(0)
                all_val_preds.extend(probs.cpu().numpy().tolist())
                all_val_labels.extend(labels.cpu().numpy().tolist())
                
        val_loss /= len(val_loader)
        val_acc = correct / total
        
        # Fix v2#6: F1-score for reliable metric under class imbalance
        all_preds_bin = [1 if p >= 0.5 else 0 for p in all_val_preds]
        val_f1 = compute_f1(all_val_labels, all_preds_bin)
        
        plateau_scheduler.step(val_loss)
        
        history['train_loss'].append(train_loss)
        if 'train_acc' not in history: history['train_acc'] = []
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_f1'].append(val_f1)
        
        print(f"--> [Epoch {epoch+1} Conclusion] Train Loss={train_loss:.4f}, Train Acc={train_acc:.4f}, Val Loss={val_loss:.4f}, Val Acc={val_acc:.4f}, Val F1={val_f1:.4f}")
        
        # Output Logging
        df = pd.DataFrame(history)
        df.to_csv(os.path.join(args.log_dir, 'training_log.csv'), index=False)
        plot_metrics(history, args.log_dir)
        
        # Always save latest state for perfect resumption
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'plateau_scheduler_state_dict': plateau_scheduler.state_dict(),
            'scaler_state_dict': scaler.state_dict(),
            'best_val_f1': best_val_f1,
            'history': history,
            'args': vars(args)
        }
        torch.save(checkpoint, os.path.join(args.log_dir, 'latest_checkpoint.pth'))
        
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            # Save raw model logic for ONNX extraction 
            torch.save(model.state_dict(), os.path.join(args.log_dir, 'best_model.pth'))
            epochs_no_improve = 0
            print("----> New Validation Peak (Highest F1)! Raw architecture successfully mapped and saved.")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= early_stop_patience:
                print(f"Early cessation barrier ({early_stop_patience}) breached! Halting execution.")
                break

    # Once loop completes or halts
    best_model_path = os.path.join(args.log_dir, 'best_model.pth')
    if os.path.exists(best_model_path):
        print("\n========================================================")
        print("[INFO] Training protocol concluded. Automatically evaluating best state against Test Set...")
        print("========================================================\n")
        from test import evaluate_model
        evaluate_model(args.data_dir, best_model_path, output_dir=args.log_dir, num_workers=NUM_WORKERS, use_audio=args.use_audio)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default=DATASET_PREPROCESSED, help='Dataset absolute directory')
    parser.add_argument('--log_dir', type=str, default=LOG_DIR, help='Directory holding log matrices')
    parser.add_argument('--epochs', type=int, default=DEFAULT_EPOCHS, help='Total maximum execution epoch cap')
    parser.add_argument('--batch_size', type=int, default=DEFAULT_BATCH_SIZE, help='Physical batch sizing')
    parser.add_argument('--accumulate_steps', type=int, default=DEFAULT_ACCUMULATE_STEPS, help='Gradient accumulation steps for low-GPU architectures')
    parser.add_argument('--patience', type=int, default=DEFAULT_PATIENCE, help='Tolerance boundary for validation stagnation')
    parser.add_argument('--lr', type=float, default=DEFAULT_LR, help='Optimizer curve gradient learning rate')
    parser.add_argument('--resume', type=str, default=None, help='Target .pth index checkpoint if executing a resumable state')
    parser.add_argument('--use_audio', action='store_true', default=USE_AUDIO_BRANCH, help='Enable the optional audio-visual branch')
    args = parser.parse_args()
    
    train_model(args)
