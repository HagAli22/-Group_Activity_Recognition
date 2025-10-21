import os
import sys
import torch
import torch.nn as nn
import albumentations as albu
from albumentations.pytorch import ToTensorV2
import torchvision.models as models
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix


sys.path.append('D:/pycharm project/slidesdeep/15 Final Project/Group-Activity-Recognition')

from data.data_loader import get_B5_B_loaders
from data.boxinfo import BoxInfo
from evaluation.eval import plot_confusion_matrix, plot_class_accuracies,create_text_results
from helper import load_yaml_config

CONFIG_PATH="configs/b5_config.yaml"

class Person_Classifer(nn.Module):
    def __init__(self,num_classes=9):
        super(Person_Classifer,self).__init__()
        self.base_model=models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.feature_extraction=nn.Sequential(*list(self.base_model.children())[:-1])

        # for param in self.feature_extraction.parameters():
        #     param.requires_grad=False

        '''
        x = b * T * 9 * 3 * H * W

        x =ResNetfeature_extraction(x)   x -->  b * T * 9 * 2048 

        x = LSTM(x)        x[:, -1 , :] --> b * T * 128

        x = AdaptiveMaxPool2d(x[:, -1 , :])   x--> b * 2048

        x = classifier(x)   x --> b * (num_classes=8) 
   
        '''

        self.hidden_size=128
        self.num_layers=1

        self.lstm=nn.LSTM(2048,self.hidden_size,num_layers=self.num_layers,batch_first=True)


        # self.pool=nn.AdaptiveMaxPool2d((1,2048))

        self.classifier=nn.Sequential(
            nn.Linear(128,512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512,num_classes)
        )


    def forward(self, x):  # x: (B, T, seq , 3, H, W)
        B, T, seq, C, H, W = x.shape
        x = x.view(B * T * seq, C, H, W)

        feats = self.feature_extraction(x)  # (B*T, 2048, 1, 1)

        x = feats.view(B*T,seq, -1)  #(B*T,seq,2048)

        x , _ =self.lstm(x)  # (B*T, seq, 128)
        x=x[:, -1 , :]  # (B*T, 128)

        x=self.classifier(x)  # (B*T, num_classes)

        return x

class Group_Classifer(nn.Module):
    def __init__(self, Person_Classifer , num_classes=8):
        super(Group_Classifer,self).__init__()

        self.feature_extraction=Person_Classifer.feature_extraction

        self.lstm=Person_Classifer.lstm

        for model in [self.feature_extraction , self.lstm ]:

            for param in model.parameters():

                param.requires_grad=False


        # self.base_model=models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        # self.feature_extraction=nn.Sequential(*list(self.base_model.children())[:-1])

        # for param in self.feature_extraction.parameters():
        #     param.requires_grad=False

        '''
        x = b * T * 9 * 3 * H * W

        x =ResNetfeature_extraction(x)   x -->  b * T * 9 * 2048 

        x = LSTM(x)        x[:, -1 , :] --> b * T * 2048

        x = AdaptiveMaxPool2d(x[:, -1 , :])   x--> b * 2048

        x = classifier(x)   x --> b * (num_classes=8) 
   
        '''

        # self.hidden_size=512
        # self.num_layers=1

        # self.lstm=nn.LSTM(2048,self.hidden_size,num_layers=self.num_layers,batch_first=True)


        self.pool=nn.AdaptiveMaxPool2d((1,2048))

        self.classifier=nn.Sequential(
            nn.Linear(2048,512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512,num_classes)
        )


    def forward(self, x):  # x: (B, T, seq , 3, H, W)
        B, T, seq, C, H, W = x.shape
        x = x.view(B * T * seq, C, H, W)

        feats = self.feature_extraction(x)  # (B*T, 2048, 1, 1)

        x = feats.view(B*T,seq, -1)  #(B*T,seq,2048)

        x2 , _ =self.lstm(x)  # (B*T, seq, 512)

        x=torch.cat([x,x2],dim=2)
        x=x.contiguous()
        x=x[:, -1 , :]  # (B*T, 512)

        x=x.view(B,T,-1) # (B , T , 512)

        x=self.pool(x) # (B , 1 , 2048)

        x=x.squeeze(dim=1) #( B , 2048)


        x=self.classifier(x)  # (B, num_classes)

        return x


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
    # Load B5 Configuration
    print("Loading Baseline 5 Configuration...")
    config = load_yaml_config(CONFIG_PATH)
    
    # Extract configuration parameters
    root_dataset = config.data['dataset_root']
    model_path = os.path.join(config.evaluation['results_dir'], 'best_Baseline_B5_B_model.pth')
    test_split = config.data['test_split']
    batch_size = config.evaluation['batch_size']
    
    print(f"Dataset Root: {root_dataset}")
    print(f"Model Path: {model_path}")
    print(f"Test Split: {test_split}")
    
    # Test preprocessing using config parameters
    test_preprocess = albu.Compose([
        albu.Resize(224, 224),
        albu.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]
        ),
        ToTensorV2()
    ])
    
    # Device configuration from config
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model with config parameters
    model = Group_Classifer(num_classes=config.model['group_activity']['num_classes'])
    
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print("Model loaded successfully!")
    else:
        print(f"Model file not found: {model_path}")
        print("Please ensure the model has been trained and saved.")
        sys.exit(1)
    
    model = model.to(device)

    # Define criterion for loss calculation
    criterion = nn.CrossEntropyLoss()
    
    # Create test dataset and dataloader using config parameters
    test_dataset = get_B5_B_loaders(
        videos_path=config.data['videos_path'],
        annot_path=config.data['annot_path'],
        split=test_split,
        transform=test_preprocess
    )
    
    print(f"Test dataset size: {len(test_dataset)}")
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=config.evaluation['batch_size'],
        shuffle=False, 
        num_workers=config.model['group_activity']['num_workers'],
        pin_memory=config.model['group_activity']['pin_memory']
    )
    
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
    
    # Generate detailed classification report using config class names
    class_names = config.model['group_activity']['class_names']
    
    print("\n" + "="*60)
    print("DETAILED CLASSIFICATION REPORT")
    print("="*60)
    report = classification_report(targets, predictions, target_names=class_names, digits=3)
    print(report)

    # Use config paths for saving results
    save_path = config.evaluation['results_dir']
    plots_path = config.evaluation['plots_dir']
    
    # Create directories if they don't exist
    os.makedirs(save_path, exist_ok=True)
    os.makedirs(plots_path, exist_ok=True)
    
    print(f"Saving results to: {save_path}")
    print(f"Saving plots to: {plots_path}")
    
    # Plot results using config parameters
    if config.evaluation['plot_class_accuracies']:
        plot_class_accuracies(class_accuracies, save_path=save_path)
        
    if config.evaluation['plot_confusion_matrix']:
        plot_confusion_matrix(targets, predictions, class_names, save_path=save_path)
        
    if config.evaluation['save_classification_report']:
        create_text_results(overall_accuracy, class_accuracies, report, save_path=save_path)
    
    print(f"Evaluation completed successfully!")
    print(f"Results saved in: {save_path}")
    print(f"Check the results directory for detailed analysis.")


    '''
    
    ============================================================
    TEST RESULTS
    ============================================================
    Overall Test Accuracy: 76.74%
    Average Loss: 0.6231
    Overall F1 Score (Macro): 0.770
    Overall F1 Score (Weighted): 0.769
        
    '''


