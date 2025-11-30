"""Training pipeline for contrastive learning."""

import os
import random
import logging
from typing import List, Dict, Optional, Tuple
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, LinearLR, SequentialLR
from tqdm import tqdm

from config import Config, TrainingConfig, AugmentationConfig
from model import ContrastivePaperModel, NTXentLoss
from augmentations import create_augmenter, create_strong_augmenter, TextAugmenter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class ContrastiveDataset(Dataset):
    """Dataset for contrastive learning with text augmentation."""
    
    def __init__(
        self,
        texts: List[str],
        tokenizer,
        augmenter_1: TextAugmenter,
        augmenter_2: TextAugmenter,
        max_length: int = 128
    ):
        self.texts = texts
        self.tokenizer = tokenizer
        self.augmenter_1 = augmenter_1
        self.augmenter_2 = augmenter_2
        self.max_length = max_length
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        text = self.texts[idx]
        
        # Create two augmented views
        view_1 = self.augmenter_1(text)
        view_2 = self.augmenter_2(text)
        
        # Tokenize both views
        encoded_1 = self.tokenizer(
            view_1,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        encoded_2 = self.tokenizer(
            view_2,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids_1': encoded_1['input_ids'].squeeze(0),
            'attention_mask_1': encoded_1['attention_mask'].squeeze(0),
            'input_ids_2': encoded_2['input_ids'].squeeze(0),
            'attention_mask_2': encoded_2['attention_mask'].squeeze(0),
        }


class Trainer:
    """Trainer for contrastive learning."""
    
    def __init__(
        self,
        model: ContrastivePaperModel,
        config: Config,
        texts: List[str],
        val_texts: Optional[List[str]] = None
    ):
        self.model = model
        self.config = config
        self.train_config = config.training
        self.device = torch.device(
            self.train_config.device if torch.cuda.is_available() else "cpu"
        )
        
        self.model.to(self.device)
        
        # Create augmenters
        self.augmenter_1 = create_augmenter(config.augmentation)
        self.augmenter_2 = create_strong_augmenter(config.augmentation)
        
        # Create dataset and dataloader
        self.train_dataset = ContrastiveDataset(
            texts=texts,
            tokenizer=model.tokenizer,
            augmenter_1=self.augmenter_1,
            augmenter_2=self.augmenter_2,
            max_length=512
        )
        
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.train_config.batch_size,
            shuffle=True,
            num_workers=self.train_config.num_workers,
            pin_memory=True,
            drop_last=True
        )
        
        # Validation data
        self.val_texts = val_texts
        
        # Loss function
        self.criterion = NTXentLoss(temperature=self.train_config.temperature)
        
        # Optimizer
        self.optimizer = AdamW(
            model.parameters(),
            lr=self.train_config.learning_rate,
            weight_decay=self.train_config.weight_decay
        )
        
        # Learning rate scheduler
        total_steps = len(self.train_loader) * self.train_config.num_epochs
        warmup_steps = min(self.train_config.warmup_steps, total_steps // 2)
        
        warmup_scheduler = LinearLR(
            self.optimizer,
            start_factor=0.1,
            end_factor=1.0,
            total_iters=max(1, warmup_steps)
        )
        
        cosine_steps = max(1, total_steps - warmup_steps)
        cosine_scheduler = CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=cosine_steps,
            T_mult=1
        )
        
        self.scheduler = SequentialLR(
            self.optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[max(1, warmup_steps)]
        )
        
        # Checkpoint directory
        self.checkpoint_dir = Path(self.train_config.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Training state
        self.global_step = 0
        self.best_loss = float('inf')
        
        # Optional: Weights & Biases logging
        self.use_wandb = self.train_config.use_wandb
        if self.use_wandb:
            try:
                import wandb
                wandb.init(
                    project=self.train_config.wandb_project,
                    config=vars(config)
                )
            except ImportError:
                logger.warning("wandb not installed, disabling logging")
                self.use_wandb = False
                
    def train_epoch(self, epoch: int) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(
            self.train_loader,
            desc=f"Epoch {epoch + 1}/{self.train_config.num_epochs}",
            leave=True
        )
        
        for batch in pbar:
            # Move to device
            input_ids_1 = batch['input_ids_1'].to(self.device)
            attention_mask_1 = batch['attention_mask_1'].to(self.device)
            input_ids_2 = batch['input_ids_2'].to(self.device)
            attention_mask_2 = batch['attention_mask_2'].to(self.device)
            
            # Use automatic mixed precision for memory efficiency
            with torch.amp.autocast(device_type='cuda', enabled=(self.device.type == 'cuda')):
                # Forward pass for both views
                z_1, _ = self.model(input_ids_1, attention_mask_1)
                z_2, _ = self.model(input_ids_2, attention_mask_2)
                
                # Compute loss
                loss = self.criterion(z_1, z_2)
            
            # Backward pass
            loss = loss / self.train_config.gradient_accumulation_steps
            loss.backward()
            
            if (self.global_step + 1) % self.train_config.gradient_accumulation_steps == 0:
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.train_config.max_grad_norm
                )
                
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
            
            # Logging
            total_loss += loss.item() * self.train_config.gradient_accumulation_steps
            num_batches += 1
            self.global_step += 1
            
            pbar.set_postfix({
                'loss': f"{loss.item() * self.train_config.gradient_accumulation_steps:.4f}",
                'lr': f"{self.scheduler.get_last_lr()[0]:.2e}"
            })
            
            if self.global_step % self.train_config.log_every_n_steps == 0:
                avg_loss = total_loss / num_batches
                if self.use_wandb:
                    import wandb
                    wandb.log({
                        'train/loss': avg_loss,
                        'train/lr': self.scheduler.get_last_lr()[0],
                        'global_step': self.global_step
                    })
                    
        return total_loss / num_batches
    
    def save_checkpoint(self, epoch: int, loss: float, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'loss': loss,
            'config': self.config,
        }
        
        # Save regular checkpoint
        checkpoint_path = self.checkpoint_dir / f"checkpoint_epoch_{epoch + 1}.pt"
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Saved checkpoint to {checkpoint_path}")
        
        # Save best model
        if is_best:
            best_path = self.checkpoint_dir / "best_model.pt"
            torch.save(checkpoint, best_path)
            logger.info(f"Saved best model to {best_path}")
            
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.global_step = checkpoint['global_step']
        
        logger.info(f"Loaded checkpoint from {checkpoint_path}")
        return checkpoint['epoch']
    
    def train(self):
        """Full training loop."""
        logger.info(f"Starting training on {self.device}")
        logger.info(f"Training {len(self.train_dataset)} samples for {self.train_config.num_epochs} epochs")
        
        for epoch in range(self.train_config.num_epochs):
            # Train epoch
            train_loss = self.train_epoch(epoch)
            
            logger.info(f"Epoch {epoch + 1}/{self.train_config.num_epochs} - Loss: {train_loss:.4f}")
            
            # Save checkpoint
            is_best = train_loss < self.best_loss
            if is_best:
                self.best_loss = train_loss
                
            if (epoch + 1) % self.train_config.save_every_n_epochs == 0 or is_best:
                self.save_checkpoint(epoch, train_loss, is_best)
                
        # Save final model
        self.save_checkpoint(
            self.train_config.num_epochs - 1,
            train_loss,
            is_best=False
        )
        
        logger.info("Training complete!")
        return self.model


def train_model(
    texts: List[str],
    config: Optional[Config] = None,
    val_texts: Optional[List[str]] = None
) -> ContrastivePaperModel:
    """
    Convenience function to train the contrastive model.
    
    Args:
        texts: List of paper texts (title + abstract)
        config: Configuration object
        val_texts: Optional validation texts
        
    Returns:
        Trained model
    """
    if config is None:
        config = Config()
        
    set_seed(config.seed)
    
    model = ContrastivePaperModel(config.model)
    trainer = Trainer(model, config, texts, val_texts)
    
    return trainer.train()


if __name__ == "__main__":
    # Test training with dummy data
    config = Config()
    config.training.num_epochs = 2
    config.training.batch_size = 8
    config.training.device = "cpu"
    
    # Dummy texts
    texts = [
        "Deep learning for computer vision. We present novel methods for image classification using convolutional neural networks.",
        "Natural language processing with transformers. Our approach uses attention mechanisms for text understanding.",
        "Reinforcement learning for robotics. We train agents to perform complex manipulation tasks.",
    ] * 10
    
    set_seed(config.seed)
    
    model = ContrastivePaperModel(config.model)
    trainer = Trainer(model, config, texts)
    
    # Train for one epoch
    loss = trainer.train_epoch(0)
    print(f"Training loss: {loss:.4f}")
