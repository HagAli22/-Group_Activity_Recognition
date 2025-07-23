import os
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
from data_loader import *
root_dataset = 'D:/volleyball-datasets'


def check():
    print('torch: version', torch.__version__)
    # Check for availability of CUDA (GPU)
    if torch.cuda.is_available():
        print("CUDA is available.")
        # Get the number of GPU devices
        num_devices = torch.cuda.device_count()
        print(f"Number of GPU devices: {num_devices}")

        # Print details for each CUDA device
        for i in range(num_devices):
            print(f"Device {i}: {torch.cuda.get_device_name(i)}")
    else:
        print("CUDA is not available. Using CPU.")

    # Get the name of the current device
    current_device = torch.cuda.current_device() if torch.cuda.is_available() else "CPU"
    print(f"Current device: {current_device}")


class VolleyballDataset(Dataset):
    def __init__(self, videos_path,annot_path, split, transform=None):
        self.samples = []
        self.transform = transform
        self.categories_dct = {
            'waiting': 0,
            'setting': 1,
            'digging': 2,
            'falling': 3,
            'spiking': 4,
            'blocking': 5,
            'jumping': 6,
            'moving': 7,
            'standing': 8
        }
        # print("Available videos in data:", list(dataset_dict.keys()))
        # print("Available labels in labels_dict:", list(label_dict.keys()))

        self.data=[]
        with open(annot_path,'rb')as file:
            videos_annot=pickle.load(file)


        for idx in split:
            video_annot=videos_annot[str(idx)]

            for clip in video_annot.keys():
                frames_data = video_annot[str(clip)]['frame_boxes_dct']

                dir_frames = list(video_annot[str(clip)]['frame_boxes_dct'].items())

                for frame_id,boxes in dir_frames:

                    #if str(clip) == str(frame_id):
                        #print("###",frame_id, str(clip))'
                    image_path=f'{videos_path}/{str(idx)}/{str(clip)}/{frame_id}.jpg'
                    image_path=os.path.join(image_path)
                    image=Image.open(image_path).convert('RGB')


                    for box_info in boxes:
                        x1,y1,x2,y2=box_info.box
                        category = box_info.category
                        #print("category",category)
                        cropred_image=image.crop((x1,y1,x2,y2))
                        cropred_image=self.transform(cropred_image)

                        self.data.append(
                            {
                                'frame_path': f'{videos_path}/{str(idx)}/{str(clip)}/{frame_id}.jpg',
                                'cropred_image': cropred_image,
                                'category': category
                            }
                        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample=self.data[idx]
        num_classes=len(self.categories_dct)
        labels=torch.zeros(num_classes)
        category=self.categories_dct[sample['category']]

        labels[self.categories_dct[sample['category']]]=1

        cropred_image=sample['cropred_image']

        return cropred_image, labels



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
    check()
    videos_root = f'{root_dataset}/videos'

    # train=[1,3,6,7,10,13,15,16,18,22,23,31,32,36,38,39,40,41,42,48,50,52,53,54]
    # val=[0,2,8,12,17,19,24,26,27,28,30,33,46,49,51]
    train = [7]
    val = [10]


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

    train_dataset = VolleyballDataset(
        videos_path=f"{root_dataset}/videos_sample",
        annot_path=f"{root_dataset}/annot_all2.pkl",
        split=train,
        transform=train_preprocess)

    val_dataset = VolleyballDataset(
        videos_path=f"{root_dataset}/videos_sample",
        annot_path=f"{root_dataset}/annot_all2.pkl",
        split=val,
        transform=val_preprocess)
    print(f"Training dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=True)

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