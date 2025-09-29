import os
import sys
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append('D:/pycharm project/slidesdeep/15 Final Project/Group-Activity-Recognition')


def plot_confusion_matrix(y_true, y_pred, class_names, save_path='Baseline_B4/resulates/'):
    """Plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    
    # Create directory if it doesn't exist
    os.makedirs(save_path, exist_ok=True)

    len_class_names=len(class_names)
    
    plt.figure(figsize=(10, len_class_names))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
    plt.show()

def plot_class_accuracies(class_accuracies, save_path='Baseline_B4/resulates/'):
    """Plot per-class accuracies"""
    classes = list(class_accuracies.keys())
    accuracies = list(class_accuracies.values())
    
    # Create directory if it doesn't exist
    os.makedirs(save_path, exist_ok=True)
    
    plt.figure(figsize=(12, 6))
    bars = plt.bar(classes, accuracies, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', 
                                              '#FFEAA7', '#DDA0DD', '#98D8C8', '#F7DC6F'])
    
    # Add value labels on bars
    for bar, acc in zip(bars, accuracies):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                f'{acc:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    plt.title('Per-Class Accuracy', fontsize=16, fontweight='bold')
    plt.xlabel('Classes', fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.xticks(rotation=45)
    plt.ylim(0, 100)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'class_accuracies.png'), dpi=300, bbox_inches='tight')
    plt.show()
    """Plot per-class accuracies"""
    classes = list(class_accuracies.keys())
    accuracies = list(class_accuracies.values())
    
    plt.figure(figsize=(12, 6))
    bars = plt.bar(classes, accuracies, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', 
                                              '#FFEAA7', '#DDA0DD', '#98D8C8', '#F7DC6F'])
    
    # Add value labels on bars
    for bar, acc in zip(bars, accuracies):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                f'{acc:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    plt.title('Per-Class Accuracy', fontsize=16, fontweight='bold')
    plt.xlabel('Classes', fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.xticks(rotation=45)
    plt.ylim(0, 100)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig('Baseline_B4/resulates/class_accuracies.png', dpi=300, bbox_inches='tight')
    plt.show()


def create_text_results(overall_accuracy, class_accuracies, report, save_path='Baseline_B4/resulates/'):
    """Save results to a text file"""
    os.makedirs(save_path, exist_ok=True)
    
    with open(os.path.join(save_path, 'test_results.txt'), 'w') as f:
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
    
    print(f"\nResults saved to '{os.path.join(save_path, 'test_results.txt')}'")