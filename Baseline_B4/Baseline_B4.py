import os
import sys
from PIL import __version__ as PILLOW_VERSION
from PIL import Image
import cv2
import numpy as np
import torch
import torch.nn as nn
import albumentations as albu
from albumentations.pytorch import ToTensorV2
from torch.cuda.amp import autocast, GradScaler
import torchvision.models as models
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import logging
from datetime import datetime
import torchvision.transforms as transforms
from pathlib import Path
import pickle
from typing import List,Tuple

sys.path.append('D:/pycharm project/slidesdeep/15 Final Project/warmup-code')

from data.data_loader import get_B4_loaders
from data.boxinfo import BoxInfo

from helper import check, setup_logger


root_dataset = 'D:/volleyball-datasets'




class ResNetSequenceClassifier(nn.Module):
    def __init__(self, num_classes=8):
        super(ResNetSequenceClassifier,self).__init__()
        self.base_model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

        layers = list(self.base_model.children())[:-1]

        # Create truncated model
        self.truncated_model = nn.Sequential(*layers)

        # for param in self.truncated_model.parameters():
        #     param.requires_grad=False

        self.hidden_size=512
        self.num_layers=1

        self.lstm=nn.LSTM(2048,self.hidden_size,num_layers=self.num_layers,batch_first=True)

        self.fc = nn.Sequential(
            nn.Linear(2048 + self.hidden_size, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),  
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.5),  
            nn.Linear(128, num_classes)
        )

    def forward(self, x):  # x: (B, 9, 3, H, W)
        B, seq, C, H, W = x.shape
        x = x.view(B * seq, C, H, W)
        #print(x.shape)

        x = self.truncated_model(x)  # (B*seq, 2048, 1, 1)
        #print(feats.shape)

        x = x.view(B, seq, -1)  # (B*seq,2048)
        #print(feats.shape)

        lstm_out, _ = self.lstm(x)  # (B*seq, 500)

        x = torch.cat([x, lstm_out], dim=2)
        #print(lstm_out.shape)

        out=self.fc(x[:, -1 , :]) # B *1
        #print(out.shape)

        return out



def validate_model(model, val_loader, criterion, device):
    """Function to validate the model and return validation loss and accuracy"""
    model.eval()
    val_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(val_loader):
            data = data.to(device)
            target = target.to(device)
            output = model(data)

            # Calculate validation loss
            loss = criterion(output, target)
            val_loss += loss.item()

            # Calculate accuracy
            # _, predicted = torch.max(output, 1)
            # total += target.size(0)
            # correct += (target == predicted).sum().item()

            _, predicted = output.max(1)
            _, target_class = target.max(1)
            total += target.size(0)
            correct += predicted.eq(target_class).sum().item()

    avg_val_loss = val_loss / len(val_loader)
    accuracy = (correct / total) * 100
    logger.info(f"Validation completed - Loss: {avg_val_loss:.4f}, Accuracy: {accuracy:.2f}%")

    return avg_val_loss, accuracy


if __name__ == '__main__':

    videos_root = f'{root_dataset}/videos'
    logger= setup_logger()
    logger.info("Starting Baseline_B4 Training")


    train=[1,3,6,7,10,13,15,16,18,22,23,31,32,36,38,39,40,41,42,48,50,52,53,54]
    val=[0,2,8,12,17,19,24,26,27,28,30,33,46,49,51]

    logger.info(f"Training videos: {len(train)} videos")
    logger.info(f"Validation videos: {len(val)} videos")
    logger.info(f"Training video IDs: {train}")
    logger.info(f"Validation video IDs: {val}")


    train_transforms =albu.Compose([
        albu.Resize(224, 224),
        
        albu.OneOf([ albu.GaussianBlur(blur_limit=(3, 7)), albu.ColorJitter(brightness=0.2),
            albu.RandomBrightnessContrast(),albu.GaussNoise()], p=0.5),
        
        albu.OneOf([albu.HorizontalFlip(),albu.VerticalFlip(),], p=0.05),
        albu.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]),
        
        ToTensorV2()
    ])

    val_transforms =  albu.Compose([
        albu.Resize(224, 224),
        albu.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]
        ),
        ToTensorV2()
    ])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ResNetSequenceClassifier(8)
    model = model.to(device)

    # Initialize optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001,weight_decay=0.1)

    # Initialize scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        patience=5,
        factor=0.1,
    )

    train_dataset = get_B4_loaders(
        videos_path=f"{root_dataset}/videos",
        annot_path=f"{root_dataset}/annot_all.pkl",
        split=train,
        transform=train_transforms)

    val_dataset = get_B4_loaders(
        videos_path=f"{root_dataset}/videos",
        annot_path=f"{root_dataset}/annot_all.pkl",
        split=val,
        transform=val_transforms)
    
    logger.info(f"Training dataset size: {len(train_dataset)}")
    logger.info(f"Validation dataset size: {len(val_dataset)}")

    print(f"Training dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")


    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4,pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=2,pin_memory=True)
    

    total_train = len(train_dataset)
    labels=[label.argmax(1).item() for batch in train_loader for label in batch[1]]
    classes=torch.bincount(torch.tensor(labels))
    classes_weights = total_train / (len(classes) * classes)
    classes_weights = classes_weights.to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.15,weight=classes_weights)

    # Training loop with validation and scheduler
    num_epochs = 40
    best_val_loss = float('inf')
    scaler = GradScaler()

    checkpoint_dir = "checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)

    logger.info(f"Checkpoint directory created: {checkpoint_dir}")
    logger.info(f"Starting training for {num_epochs} epochs")

    for epoch in range(num_epochs):
        logger.info(f"Starting epoch {epoch + 1}/{num_epochs}")
        # Training phase
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        correct=0

        for batch_idx, (data, target) in enumerate(train_loader):
            data = data.to(device)
            target = target.to(device)

            optimizer.zero_grad()
            with autocast(dtype=torch.float16):
                output = model(data)
                loss = criterion(output, target)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()

            # Calculate training accuracy
            target_class = target.argmax(1)
            # _, predicted = torch.max(output, 1)
            predictedd = output.argmax(1)
            train_total += target.size(0)
            # train_correct += (target == predicted).sum().item()
            correct += predictedd.eq(target_class).sum().item()

            # Log progress every 100 batches
            if (batch_idx + 1) % 100 == 0:
                current_acc = (correct / train_total) * 100
                logger.info(f"  Batch {batch_idx + 1}/{len(train_loader)} - "
                           f"Loss: {loss.item():.4f}, Acc: {current_acc:.2f}%")

        # Calculate average training loss and accuracy
        avg_train_loss = train_loss / len(train_loader)
        train_accuracy = (correct / train_total) * 100

        # Validation phase
        logger.info("Running validation...")
        val_loss, val_accuracy = validate_model(model, val_loader, criterion, device)

        # Update scheduler based on validation loss
        old_lr = optimizer.param_groups[0]['lr']
        scheduler.step(val_loss)
        new_lr = optimizer.param_groups[0]['lr']
        
        if old_lr != new_lr:
            logger.info(f"Learning rate reduced from {old_lr:.6f} to {new_lr:.6f}")

        
        if (epoch+1) % 10 == 0 or epoch < num_epochs:
            # Save checkpoint
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_accuracy': val_accuracy
            }
        
            checkpoint_path = f"{checkpoint_dir}/checkpoint_epoch_{epoch + 1}.pth"
            torch.save(checkpoint, checkpoint_path)
            logger.info(f"Checkpoint saved for epoch {epoch + 1} at {checkpoint_path}")

        # Get current learning rate
        current_lr = optimizer.param_groups[0]['lr']

        # Log epoch results
        logger.info(f"Epoch {epoch + 1}/{num_epochs} Results:")
        logger.info(f"  Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.2f}%")
        logger.info(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.2f}%")
        logger.info(f"  Learning Rate: {current_lr:.6f}")
        logger.info("-" * 60)

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_Baseline_B4_model.pth')
            logger.info(f"New best model saved with validation loss: {val_loss:.4f}")
            logger.info(f"Best model saved to: best_Baseline_B4_model.pth")

    # Final evaluation
    logger.info("Training completed!")
    logger.info(f"Best validation loss achieved: {best_val_loss:.4f}")

    # Load best model for final evaluation
    logger.info("Loading best model for final evaluation...")
    model.load_state_dict(torch.load('best_Baseline_B4_model.pth'))
    final_val_loss, final_val_accuracy = validate_model(model, val_loader, criterion, device)
    logger.info(f"Final validation accuracy: {final_val_accuracy:.2f}%")
    # logger.info("training completed successfully")aluation
    logger.info("Loading best model for final evaluation...")
    model.load_state_dict(torch.load('best_Baseline_B4_model.pth'))
    final_val_loss, final_val_accuracy = validate_model(model, val_loader, criterion, device)
    logger.info(f"Final validation accuracy: {final_val_accuracy:.2f}%")
    logger.info("training completed successfully")