
import os
import sys
import torch
import torch.nn as nn
import albumentations as albu
from albumentations.pytorch import ToTensorV2
from torch.cuda.amp import autocast, GradScaler
import torchvision.models as models
from torch.utils.data import DataLoader

sys.path.append('D:/pycharm project/slidesdeep/15 Final Project/Group-Activity-Recognition')

from data.boxinfo import BoxInfo
from data.data_loader import get_B5_B_loaders
from helper import check, setup_logger , load_yaml_config
from eval_model import Person_Classifer,Group_Classifer
root_dataset = 'D:/volleyball-datasets'
CONFIG_PATH = 'configs/b5_config.yaml'

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
            # target = target.view(-1, target.shape[-1])
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
    check()

    # Load B5 step_B Configuration
    print("Loading Baseline B5 step_B Training Configuration...")
    config = load_yaml_config(CONFIG_PATH)

    # Extract configuration parameters
    root_dataset = config.data['dataset_root']
    videos_root = config.data['videos_path']
    train_split = config.data['train_split']
    val_split = config.data['val_split']

    logger= setup_logger()
    logger.info("Starting Baseline_B5_stepB Training")

    logger.info(f"Training videos: {len(train_split)} videos")
    logger.info(f"Validation videos: {len(val_split)} videos")
    logger.info(f"Training video IDs: {train_split}")
    logger.info(f"Validation video IDs: {val_split}")


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
    logger.info(f"Using device: {device}")

    Person=Person_Classifer(num_classes=config.model['person_activity']['num_classes'])
    Person.load_state_dict(torch.load('Baseline_B5/results/best_Baseline_B5_A_model.pth'))

    # Initialize model with config parameters
    model = Group_Classifer(Person_Classifer=Person,num_classes=config.model['group_activity']['num_classes'])
    model = model.to(device)
    print(f"Model initialized with {config.model['group_activity']['num_classes']} classes")

    # Initialize optimizer with config parameters
    if config.training['group_activity']['optimizer'].lower() == 'adamw':
        optimizer = torch.optim.AdamW(
            model.parameters(), 
            lr=config.training['group_activity']['learning_rate'],
            weight_decay=config.training['group_activity']['weight_decay']
        )
    elif config.training['group_activity']['optimizer'].lower() == 'adam':
        optimizer = torch.optim.Adam(
            model.parameters(), 
            lr=config.training['group_activity']['learning_rate'],
            weight_decay=config.training['group_activity']['weight_decay']
        )
    else:
        optimizer = torch.optim.SGD(
            model.parameters(), 
            lr=config.training['group_activity']['learning_rate'],
            weight_decay=config.training['group_activity']['weight_decay'],
            momentum=config.training['group_activity']['momentum']
        )

    print(f"Optimizer: {config.training['group_activity']['optimizer'].upper()}, LR: {config.training['group_activity']['learning_rate']}")

    # Initialize scheduler with config parameters
    if config.training['group_activity']['use_scheduler']:
        if config.training['group_activity']['scheduler_type'] == 'reduce_on_plateau':
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='min',
                patience=config.training['group_activity']['scheduler_patience'],
                factor=config.training['group_activity']['scheduler_factor'],
                verbose=True,
            )
        elif config.training['group_activity']['scheduler_type'] == 'cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=config.training['group_activity']['num_epochs']
            )
        else:
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=config.training['group_activity']['scheduler_patience'],
                gamma=config.training['group_activity']['scheduler_factor']
            )
        print(f"Scheduler: {config.training['group_activity']['scheduler_type']}")
    else:
        scheduler = None


    # Create datasets using config parameters
    train_dataset = get_B5_B_loaders(
        videos_path=config.data['videos_path'],
        annot_path=config.data['annot_path'],
        split=train_split,
        transform=train_transforms)

    val_dataset = get_B5_B_loaders(
        videos_path=config.data['videos_path'],
        annot_path=config.data['annot_path'],
        split=val_split,
        transform=val_transforms)
    
    logger.info(f"Training dataset size: {len(train_dataset)}")
    logger.info(f"Validation dataset size: {len(val_dataset)}")
    print(f"Training dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")

    # Create data loaders using config parameters
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.model['group_activity']['batch_size'],
        shuffle=config.model['group_activity']['shuffle_train'],
        num_workers=config.model['group_activity']['num_workers'],
        pin_memory=config.model['group_activity']['pin_memory']
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config.model['group_activity']['batch_size'], 
        shuffle=False,
        num_workers=config.model['group_activity']['num_workers'],
        pin_memory=config.model['group_activity']['pin_memory']
    )

    num_epochs = config.training['group_activity']['num_epochs']
    best_val_loss = float('inf')
    
    criterion = nn.CrossEntropyLoss()

    # Initialize mixed precision scaler if enabled
    scaler = GradScaler() if config.training['group_activity']['use_amp'] else None
    
    print(f"Starting training for {num_epochs} epochs...")
    print(f"Mixed Precision: {'Enabled' if config.training['group_activity']['use_amp'] else 'Disabled'}")

    model_name=config.training['group_activity']['model_name']


    checkpoint_dir = config.training['group_activity']['checkpoint_dir']
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

            # Use mixed precision if enabled
            if config.training['group_activity']['use_amp'] and scaler is not None:
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

        
        if (epoch+1) % 5 == 0 or epoch < num_epochs:
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
            model_save_path = os.path.join(checkpoint_dir, f'best_{model_name}.pth')
            torch.save(model.state_dict(), model_save_path)
            logger.info(f"New best model saved with validation loss: {val_loss:.4f}")
            logger.info(f"Best model saved to: best_Baseline_B5_B_model.pth")

    # Final evaluation
    logger.info("Training completed!")
    logger.info(f"Best validation loss achieved: {best_val_loss:.4f}")

    # Load best model for final evaluation
    logger.info("Loading best model for final evaluation...")
    model.load_state_dict(torch.load('best_Baseline_B5_B_model.pth'))
    final_val_loss, final_val_accuracy = validate_model(model, val_loader, criterion, device)
    logger.info(f"Final validation accuracy: {final_val_accuracy:.2f}%")
    