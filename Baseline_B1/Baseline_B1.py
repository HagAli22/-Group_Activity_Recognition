import os
import sys
from PIL import __version__ as PILLOW_VERSION
import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

sys.path.append('D:/pycharm project/slidesdeep/15 Final Project/Group-Activity-Recognition')

from data.data_loader import get_B1_loaders
from data.boxinfo import BoxInfo
from helper import check,load_yaml_config
from B1_eval import Group_Classifer

CONFIG_PATH="configs/b1_config.yaml"


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
    # Check for availability of CUDA (GPU)
    check()

    # Load B1 Configuration
    print("Loading Baseline 1 Training Configuration...")
    config = load_yaml_config(CONFIG_PATH)

    
    # Extract configuration parameters
    root_dataset = config.data['dataset_root']
    videos_root = config.data['videos_path']
    train_split = config.data['train_split']
    val_split = config.data['val_split']
    
    print(f"Dataset Root: {root_dataset}")
    print(f"Videos Path: {videos_root}")
    print(f"Training Split: {len(train_split)} videos")
    print(f"Validation Split: {len(val_split)} videos")

    # Create data transforms using config parameters
    train_preprocess = transforms.Compose([
        transforms.Resize((config.data['image_size'], config.data['image_size'])),
        
        transforms.RandomApply([
            transforms.GaussianBlur(kernel_size=(3, getattr(config.data, 'gaussian_blur_kernel', 7))),
            transforms.ColorJitter(brightness=0.2),
            transforms.RandomAdjustSharpness(sharpness_factor=getattr(config.data, 'sharpness_factor', 2.0)),
        ], p=config.data['color_jitter_prob']),

        transforms.RandomApply([
            transforms.RandomHorizontalFlip(p=1.0),
            transforms.RandomVerticalFlip(p=getattr(config.data, 'vertical_flip_prob', 0.05) * 20),  # Convert to full probability
        ], p=getattr(config.data, 'vertical_flip_prob', 0.05)),
        
        transforms.ToTensor(),
        
        transforms.Normalize(mean=config.data['mean'], std=config.data['std'])
    ])
    p=getattr(config.data, 'vertical_flip_prob', 0.05)
    print("p",p)
    
    val_preprocess = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop((config.data['image_size'], config.data['image_size'])),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.data['mean'], std=config.data['std']),
    ])

    # Device configuration from config
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Initialize model with config parameters
    model = Group_Classifer(config.model['num_classes'])
    model = model.to(device)
    print(f"Model initialized with {config.model['num_classes']} classes")

    # Initialize optimizer with config parameters
    if config.training['optimizer'].lower() == 'adamw':
        optimizer = torch.optim.AdamW(
            model.parameters(), 
            lr=config.training['learning_rate'],
            weight_decay=config.training['weight_decay']
        )
    elif config.training['optimizer'].lower() == 'adam':
        optimizer = torch.optim.Adam(
            model.parameters(), 
            lr=config.training['learning_rate'],
            weight_decay=config.training['weight_decay']
        )
    else:
        optimizer = torch.optim.SGD(
            model.parameters(), 
            lr=config.training['learning_rate'],
            weight_decay=config.training['weight_decay'],
            momentum=config.training['momentum']
        )
    
    criterion = nn.CrossEntropyLoss()
    print(f"Optimizer: {config.training['optimizer'].upper()}, LR: {config.training['learning_rate']}")

    # Initialize scheduler with config parameters
    if config.training['use_scheduler']:
        if config.training['scheduler_type'] == 'reduce_on_plateau':
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='min',
                patience=config.training['scheduler_patience'],
                factor=config.training['scheduler_factor'],
                verbose=True,
            )
        elif config.training['scheduler_type'] == 'cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=config.training['num_epochs']
            )
        else:
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=config.training['scheduler_patience'],
                gamma=config.training['scheduler_factor']
            )
        print(f"Scheduler: {config.training['scheduler_type']}")
    else:
        scheduler = None

    # Create datasets using config parameters
    train_dataset = get_B1_loaders(
        videos_path=config.data['videos_path'],
        annot_path=config.data['annot_path'],
        split=train_split,
        transform=train_preprocess)

    val_dataset = get_B1_loaders(
        videos_path=config.data['videos_path'],
        annot_path=config.data['annot_path'],
        split=val_split,
        transform=val_preprocess)
    
    print(f"Training dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")

    # Create data loaders using config parameters
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.data['batch_size'], 
        shuffle=config.data['shuffle_train'],
        num_workers=config.data['num_workers'],
        pin_memory=config.data['pin_memory']
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config.data['batch_size'], 
        shuffle=False,
        num_workers=config.data['num_workers'],
        pin_memory=config.data['pin_memory']
    )

    # Training loop with validation and scheduler
    num_epochs = config.training['num_epochs']
    best_val_loss = float('inf')
    
    # Initialize mixed precision scaler if enabled
    scaler = GradScaler() if config.training['use_amp'] else None
    
    print(f"Starting training for {num_epochs} epochs...")
    print(f"Mixed Precision: {'Enabled' if config.training['use_amp'] else 'Disabled'}")

    model_name=config.training['model_name']

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
            
            # Use mixed precision if enabled
            if config.training['use_amp'] and scaler is not None:
                with autocast(dtype=torch.float16):
                    output = model(data)
                    loss = criterion(output, target)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()

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
        if scheduler is not None:
            if config.training['scheduler_type'] == 'reduce_on_plateau':
                scheduler.step(val_loss)
            else:
                scheduler.step()

        # Get current learning rate
        current_lr = optimizer.param_groups[0]['lr']

        print(f"Epoch: {epoch + 1}/{num_epochs}")
        print(f"Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.2f}%")
        print(f"Learning Rate: {current_lr:.6f}")
        print("-" * 50)

        # Save best model using config paths
        if config.training['save_best_model'] and val_loss < best_val_loss:
            best_val_loss = val_loss
            
            # Create checkpoint directory if it doesn't exist
            checkpoint_dir = config.training['checkpoint_dir']
            os.makedirs(checkpoint_dir, exist_ok=True)
            
            # Save model with config-specified name
            model_save_path = os.path.join(checkpoint_dir, f'best_{model_name}.pth')
            torch.save(model.state_dict(), model_save_path)
            print(f"New best model saved: {model_save_path}")
            print(f"Validation loss: {val_loss:.4f}")

    # Final evaluation
    print("\nTraining completed!")
    print(f"Best validation loss: {best_val_loss:.4f}")

    # Load best model for final evaluation using config path
    if config.training['save_best_model']:
        best_model_path = os.path.join(config.training['checkpoint_dir'], f'best_{model_name}.pth')
        if os.path.exists(best_model_path):
            model.load_state_dict(torch.load(best_model_path, map_location=device))
            final_val_loss, final_val_accuracy = validate_model(model, val_loader, criterion, device)
            print(f"Final validation accuracy: {final_val_accuracy:.2f}%")
            print(f"Best model saved at: {best_model_path}")
        else:
            print(f"Best model not found at: {best_model_path}")
            print("Using current model state for final evaluation")
            final_val_loss, final_val_accuracy = validate_model(model, val_loader, criterion, device)
            print(f"Final validation accuracy: {final_val_accuracy:.2f}%")
    
    # Save final model if configured
    if config.training['save_last_model']:
        final_model_path = os.path.join(config.training['checkpoint_dir'], f'final_{model_name}.pth')
        torch.save(model.state_dict(), final_model_path)
        print(f"Final model saved at: {final_model_path}")
    
    print(f"\nTraining session completed successfully!")
    print(f"Experiment: {config['experiment_name']}")
    print(f"Check results in: {config.training['checkpoint_dir']}")