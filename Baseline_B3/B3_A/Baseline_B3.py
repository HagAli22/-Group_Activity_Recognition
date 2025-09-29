import os
import sys
from PIL import __version__ as PILLOW_VERSION
from PIL import Image
import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
import torchvision.models as models
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import pickle
sys.path.append('D:/pycharm project/slidesdeep/15 Final Project/warmup-code')

from data.data_loader import get_B3_A_loaders
from data.boxinfo import BoxInfo

from helper import check, setup_logger
root_dataset = 'D:/volleyball-datasets'


class ResNetSequenceClassifier(nn.Module):
    def __init__(self, num_classes=9):
        super().__init__()
        self.base_model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.in_feature=self.base_model.fc.in_features

        self.base_model.fc=nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_features=self.base_model.fc.in_features, out_features=num_classes)
        )


    def forward(self, x):  # x: (B, 9, 3, H, W)
        out = self.base_model(x)  # (B, num_classes)
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

    return avg_val_loss, accuracy


if __name__ == '__main__':
    
    videos_root = f'{root_dataset}/videos'

    train=[1,3,6,7,10,13,15,16,18,22,23,31,32,36,38,39,40,41,42,48,50,52,53,54]
    val=[0,2,8,12,17,19,24,26,27,28,30,33,46,49,51]
    # train = [7]
    # val = [10]


    # Enhanced data augmentation for volleyball dataset
    train_preprocess = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomResizedCrop(224, scale=(0.7, 1.0), ratio=(0.8, 1.2)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(
            brightness=0.3, 
            contrast=0.3, 
            saturation=0.2, 
            hue=0.05
        ),
        transforms.RandomApply([
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0))
        ], p=0.2),
        transforms.RandomApply([
            transforms.RandomAdjustSharpness(sharpness_factor=1.5)
        ], p=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.1, scale=(0.02, 0.1), ratio=(0.3, 3.3))
    ])

    val_preprocess = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ResNetSequenceClassifier(9)
    model = model.to(device)

    # Initialize optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001)
    criterion = nn.CrossEntropyLoss()

    # Initialize scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        patience=3,
        factor=0.1,
        verbose=True,
    )

    train_dataset = get_B3_A_loaders(
        videos_path=f"{root_dataset}/videos_sample",
        annot_path=f"{root_dataset}/annot_all2.pkl",
        split=train,
        transform=train_preprocess)

    val_dataset = get_B3_A_loaders(
        videos_path=f"{root_dataset}/videos_sample",
        annot_path=f"{root_dataset}/annot_all2.pkl",
        split=val,
        transform=val_preprocess)
    print(f"Training dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=True)

    # Training loop with validation and scheduler
    num_epochs = 10
    best_val_loss = float('inf')
    scaler = GradScaler()

    for epoch in range(num_epochs):
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

        # Calculate average training loss and accuracy
        avg_train_loss = train_loss / len(train_loader)
        train_accuracy = (correct / train_total) * 100

        # Validation phase
        val_loss, val_accuracy = validate_model(model, val_loader, criterion, device)

        # Update scheduler based on validation loss
        scheduler.step(val_loss)

        # Get current learning rate
        current_lr = optimizer.param_groups[0]['lr']

        print(f"Epoch: {epoch + 1}/{num_epochs}")
        print(f"Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.2f}%")
        print(f"Learning Rate: {current_lr:.6f}")
        print("-" * 50)

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_volleyball_model.pth')
            print(f"New best model saved with validation loss: {val_loss:.4f}")

    # Final evaluation
    print("Training completed!")
    print(f"Best validation loss: {best_val_loss:.4f}")

    # Load best model for final evaluation
    model.load_state_dict(torch.load('best_volleyball_model.pth'))
    final_val_loss, final_val_accuracy = validate_model(model, val_loader, criterion, device)
    print(f"Final validation accuracy: {final_val_accuracy:.2f}%")