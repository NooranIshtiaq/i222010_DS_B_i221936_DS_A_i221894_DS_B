import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from pathlib import Path
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from m_fusher_model import MFusHER
from ravdess_dataset import RAVDESSDataset, create_ravdess_loaders


class RAVDESSTrainer:
    """Trainer optimized for RAVDESS dataset."""
    
    def _init_(self, model, train_loader, val_loader, test_loader, config):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.config = config
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)
        print(f"Using device: {self.device}")
        
        # Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config['lr'],
            weight_decay=config['weight_decay']
        )
        
        # Scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=config['epochs'], eta_min=1e-6
        )
        
        # Loss - CrossEntropy for single-label classification
        self.criterion = nn.CrossEntropyLoss()
        
        self.best_acc = 0.0
        self.history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    
    def train_epoch(self):
        self.model.train()
        total_loss = 0
        all_preds, all_labels = [], []
        
        pbar = tqdm(self.train_loader, desc="Training")
        for batch in pbar:
            audio = batch['audio'].to(self.device)
            video = batch['video'].to(self.device)
            text = batch['text'].to(self.device)
            labels = batch['emotion_id'].to(self.device)
            
            self.optimizer.zero_grad()
            
            # Forward pass - get logits
            logits = self.model(audio=audio, visual=video, text=text)
            
            # Loss (CrossEntropy expects class indices, not one-hot)
            loss = self.criterion(logits, labels)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_loss = total_loss / len(self.train_loader)
        accuracy = accuracy_score(all_labels, all_preds)
        
        return avg_loss, accuracy
    
    @torch.no_grad()
    def evaluate(self, loader, return_preds=False):
        self.model.eval()
        total_loss = 0
        all_preds, all_labels = [], []
        
        for batch in loader:
            audio = batch['audio'].to(self.device)
            video = batch['video'].to(self.device)
            text = batch['text'].to(self.device)
            labels = batch['emotion_id'].to(self.device)
            
            logits = self.model(audio=audio, visual=video, text=text)
            loss = self.criterion(logits, labels)
            
            total_loss += loss.item()
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
        
        avg_loss = total_loss / len(loader)
        accuracy = accuracy_score(all_labels, all_preds)
        
        if return_preds:
            return avg_loss, accuracy, all_preds, all_labels
        return avg_loss, accuracy
    
    def train(self, epochs):
        print(f"\nTraining for {epochs} epochs...")
        print(f"Train samples: {len(self.train_loader.dataset)}")
        print(f"Val samples: {len(self.val_loader.dataset)}")
        print(f"Test samples: {len(self.test_loader.dataset)}")
        
        for epoch in range(1, epochs + 1):
            print(f"\nEpoch {epoch}/{epochs}")
            
            train_loss, train_acc = self.train_epoch()
            val_loss, val_acc = self.evaluate(self.val_loader)
            
            self.scheduler.step()
            
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_acc'].append(val_acc)
            
            print(f"Train Loss: {train_loss:.4f} | Acc: {train_acc:.4f}")
            print(f"Val Loss: {val_loss:.4f} | Acc: {val_acc:.4f}")
            
            if val_acc > self.best_acc:
                self.best_acc = val_acc
                torch.save(self.model.state_dict(), 'best_ravdess_model.pth')
                print(f"✓ New best model saved! (Acc: {val_acc:.4f})")
        
        # Final test evaluation
        self.model.load_state_dict(torch.load('best_ravdess_model.pth'))
        test_loss, test_acc, preds, labels = self.evaluate(self.test_loader, return_preds=True)
        
        print("\n" + "="*60)
        print("FINAL TEST RESULTS")
        print("="*60)
        print(f"Test Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")
        print(f"Test Loss: {test_loss:.4f}")
        
        # Classification report
        emotions = RAVDESSDataset.EMOTIONS
        print("\nClassification Report:")
        print(classification_report(labels, preds, target_names=emotions))
        
        # Save confusion matrix
        self.plot_confusion_matrix(labels, preds, emotions)
        self.plot_training_curves()
        
        return test_acc
    
    def plot_confusion_matrix(self, labels, preds, class_names):
        cm = confusion_matrix(labels, preds)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=class_names, yticklabels=class_names)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix - RAVDESS')
        plt.tight_layout()
        plt.savefig('confusion_matrix_ravdess.png', dpi=150)
        print("Confusion matrix saved to confusion_matrix_ravdess.png")
    
    def plot_training_curves(self):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        ax1.plot(self.history['train_loss'], label='Train')
        ax1.plot(self.history['val_loss'], label='Validation')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Loss Curve')
        ax1.legend()
        ax1.grid(True)
        
        ax2.plot(self.history['train_acc'], label='Train')
        ax2.plot(self.history['val_acc'], label='Validation')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.set_title('Accuracy Curve')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig('training_curves_ravdess.png', dpi=150)
        print("Training curves saved to training_curves_ravdess.png")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Train M-fusHER on RAVDESS')
    parser.add_argument('--data_dir', type=str, default='data/RAVDESS',
                        help='Path to RAVDESS dataset')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--use_video', action='store_true',
                        help='Use video modality (slower)')
    args = parser.parse_args()
    
    print("="*60)
    print("M-fusHER Training on RAVDESS")
    print("="*60)
    
    # Check dataset
    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f"\n❌ Dataset not found at {data_dir}")
        print("\nTo download RAVDESS dataset:")
        print("1. Go to: https://www.kaggle.com/datasets/uwrfkaggler/ravdess-emotional-speech-audio")
        print("2. Download and extract to data/RAVDESS/")
        print("\nAlternative: Run with synthetic data for testing:")
        print("   python train.py --epochs 10")
        return
    
    print(f"\nConfiguration:")
    print(f"  Data directory: {args.data_dir}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Use video: {args.use_video}")
    
    # Create dataloaders
    print("\nLoading dataset...")
    train_loader, val_loader, test_loader = create_ravdess_loaders(
        args.data_dir,
        batch_size=args.batch_size,
        use_video=args.use_video
    )
    
    # Create model (8 emotions for RAVDESS)
    print("\nCreating model...")
    model = MFusHER(
        num_emotions=8,  # RAVDESS has 8 emotions
        d_model=256,
        num_heads=4,
        d_ff=1024,  # Smaller for faster training
        num_encoder_layers=2,  # Fewer layers for smaller dataset
        num_decoder_layers=2,
        dropout=0.2,
        use_pretrained_text=False
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")
    
    # Train
    config = {
        'lr': args.lr,
        'weight_decay': 1e-4,
        'epochs': args.epochs
    }
    
    trainer = RAVDESSTrainer(model, train_loader, val_loader, test_loader, config)
    final_acc = trainer.train(args.epochs)
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)
    print(f"Final Test Accuracy: {final_acc*100:.2f}%")
    print("\nSaved files:")
    print("  - best_ravdess_model.pth")
    print("  - confusion_matrix_ravdess.png")
    print("  - training_curves_ravdess.png")


if _name_ == "_main_":
    main()