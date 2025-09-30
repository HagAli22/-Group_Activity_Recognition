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
from sklearn.metrics import classification_report, confusion_matrix


sys.path.append('D:/pycharm project/slidesdeep/15 Final Project/warmup-code')

from data.data_loader import get_B4_loaders

from evaluation.eval import plot_confusion_matrix, plot_class_accuracies


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


def test_model(model, test_loader, criterion, device):
    """Test the model and return detailed metrics"""
    model.eval()
    
    all_predictions = []
    all_targets = []
    correct = 0
    total = 0
    test_loss = 0
    
    category_correct = {i: 0 for i in range(8)}
    category_total = {i: 0 for i in range(8)}
    
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data = data.to(device)
            target = target.to(device)
            
            output = model(data)
            
            # Calculate loss
            loss = criterion(output, target)
            test_loss += loss.item()
            
            # Get predictions
            _, predicted = output.max(1)
            _, target_class = target.max(1)
            
            # Store for detailed analysis (convert to list once)
            all_predictions.extend(predicted.cpu().numpy().tolist())
            all_targets.extend(target_class.cpu().numpy().tolist())
            
            # Calculate overall accuracy
            total += target.size(0)
            correct += predicted.eq(target_class).sum().item()
            
            # Calculate per-class accuracy
            for i in range(len(target_class)):
                true_class = target_class[i].item()
                pred_class = predicted[i].item()
                category_total[true_class] += 1
                if true_class == pred_class:
                    category_correct[true_class] += 1
    
    # Calculate metrics
    overall_accuracy = (correct / total) * 100
    average_loss = test_loss / len(test_loader)
    
    # Calculate per-class accuracy
    class_accuracies = {}
    categories_dct = {
        'l-pass': 0, 'r-pass': 1, 'l-spike': 2, 'r_spike': 3,
        'l_set': 4, 'r_set': 5, 'l_winpoint': 6, 'r_winpoint': 7
    }
    idx_to_category = {v: k for k, v in categories_dct.items()}
    
    for class_idx, class_name in idx_to_category.items():
        if category_total[class_idx] > 0:
            class_accuracies[class_name] = (category_correct[class_idx] / category_total[class_idx]) * 100
        else:
            class_accuracies[class_name] = 0.0
    
    return overall_accuracy, average_loss, class_accuracies, all_predictions, all_targets


if __name__ == '__main__':
    # Configuration
    root_dataset = 'D:/volleyball-datasets'  # Change this to your dataset path
    model_path = 'Baseline_B4/resulates/best_Baseline_B4_model.pth'  # Path to your saved model
    
    # Test split - use different videos than training/validation
    test = [0, 2, 8, 12, 17, 19, 24, 26, 27, 28, 30, 33, 46, 49, 51]  # Example test split
    
    # Test preprocessing (same as validation)
    test_preprocess = albu.Compose([
        albu.Resize(224, 224),
        albu.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]
        ),
        ToTensorV2()
    ])
    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    model = ResNetSequenceClassifier(8)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    print("Model loaded successfully!")

    # Define criterion for loss calculation
    criterion = nn.CrossEntropyLoss()
    
    # Create test dataset and dataloader
    test_dataset = get_B4_loaders(
        videos_path=f"{root_dataset}/videos",
        annot_path=f"{root_dataset}/annot_all.pkl",
        split=test,
        transform=test_preprocess
    )
    
    print(f"Test dataset size: {len(test_dataset)}")
    
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=2,pin_memory=True)
    
    # Test the model
    print("Starting model evaluation...")
    overall_accuracy, average_loss, class_accuracies, predictions, targets = test_model(model, test_loader, criterion, device)
    
    # Calculate overall F1 scores
    from sklearn.metrics import f1_score
    overall_f1_macro = f1_score(targets, predictions, average='macro')
    overall_f1_weighted = f1_score(targets, predictions, average='weighted')
    
    # Print results
    print("\n" + "="*60)
    print("TEST RESULTS")
    print("="*60)
    print(f"Overall Test Accuracy: {overall_accuracy:.2f}%")
    print(f"Average Loss: {average_loss:.4f}")
    print(f"Overall F1 Score (Macro): {overall_f1_macro:.4f}")
    print(f"Overall F1 Score (Weighted): {overall_f1_weighted:.4f}")
    print("\nPer-Class Accuracies:")
    print("-" * 30)
    
    for class_name, accuracy in class_accuracies.items():
        print(f"{class_name:12}: {accuracy:6.2f}%")
    
    # Generate detailed classification report
    categories_dct = {
        'l-pass': 0, 'r-pass': 1, 'l-spike': 2, 'r_spike': 3,
        'l_set': 4, 'r_set': 5, 'l_winpoint': 6, 'r_winpoint': 7
    }
    class_names = list(categories_dct.keys())
    
    print("\n" + "="*60)
    print("DETAILED CLASSIFICATION REPORT")
    print("="*60)
    report = classification_report(targets, predictions, target_names=class_names, digits=3)
    print(report)
    
    # Plot results
    plot_class_accuracies(class_accuracies,save_path='Baseline_B4/resulates/')
    plot_confusion_matrix(targets, predictions, class_names,save_path='Baseline_B4/resulates/')
    
    # Save results to file
    results_dir = 'Baseline_B4/resulates/'
    os.makedirs(results_dir, exist_ok=True)
    
    with open(os.path.join(results_dir, 'test_results.txt'), 'w') as f:
        f.write("VOLLEYBALL MODEL TEST RESULTS\n")
        f.write("="*50 + "\n\n")
        f.write(f"Overall Test Accuracy: {overall_accuracy:.2f}%\n\n")
        f.write("Per-Class Accuracies:\n")
        f.write("-" * 30 + "\n")
        for class_name, accuracy in class_accuracies.items():
            f.write(f"{class_name:12}: {accuracy:6.2f}%\n")
        f.write("\n" + "="*50 + "\n")
        f.write("DETAILED CLASSIFICATION REPORT\n")
        f.write("="*50 + "\n")
        f.write(report)
    
    print(f"\nResults saved to 'test_results.txt'")
    print("Confusion matrix saved as 'confusion_matrix.png'")
    print("Class accuracies plot saved as 'class_accuracies.png'")
    print("\nTesting completed successfully!")


    '''
    
    ============================================================
    TEST RESULTS
    ============================================================
    Overall Test Accuracy: 74.79%
    Average Loss: 0.9381
    Overall F1 Score (Macro): 0.7652
    Overall F1 Score (Weighted): 0.7487
        
    '''


