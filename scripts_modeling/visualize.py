import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (confusion_matrix, roc_curve, precision_recall_curve, 
                             f1_score, precision_score, recall_score)
import seaborn as sns
import os

from config import CONFIG
from data_loader import get_dataloaders
from model import FraudDetector
from utils import setup_logger

def get_predictions(model, loader, device):
    model.eval()
    all_probs, all_labels = [], []
    
    with torch.no_grad():
        for X, y in loader:
            X = X.to(device)
            probs = model(X)
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(y.numpy())
    
    return np.array(all_probs), np.array(all_labels)

def plot_confusion_matrix(labels, preds, save_path):
    cm = confusion_matrix(labels, preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Legitimate', 'Fraud'],
                yticklabels=['Legitimate', 'Fraud'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

def plot_roc_curve(labels, probs, save_path):
    fpr, tpr, _ = roc_curve(labels, probs)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, 'b-', linewidth=2)
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

def plot_precision_recall(labels, probs, save_path):
    precision, recall, _ = precision_recall_curve(labels, probs)
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, 'b-', linewidth=2)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

def plot_threshold_analysis(labels, probs, save_path):
    thresholds = np.arange(0.1, 0.95, 0.05)
    f1_scores, precisions, recalls = [], [], []
    
    for thresh in thresholds:
        preds = (probs > thresh).astype(float)
        f1_scores.append(f1_score(labels, preds))
        precisions.append(precision_score(labels, preds))
        recalls.append(recall_score(labels, preds))
    
    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, f1_scores, 'b-', label='F1 Score', linewidth=2)
    plt.plot(thresholds, precisions, 'g-', label='Precision', linewidth=2)
    plt.plot(thresholds, recalls, 'r-', label='Recall', linewidth=2)
    plt.xlabel('Threshold')
    plt.ylabel('Score')
    plt.title('Metrics vs Threshold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

def plot_score_distribution(labels, probs, save_path):
    plt.figure(figsize=(10, 6))
    plt.hist(probs[labels == 0], bins=50, alpha=0.5, label='Legitimate', color='green')
    plt.hist(probs[labels == 1], bins=50, alpha=0.5, label='Fraud', color='red')
    plt.xlabel('Predicted Probability')
    plt.ylabel('Count')
    plt.title('Score Distribution by Class')
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

def main():
    logger = setup_logger('visualization')
    os.makedirs('results', exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    _, _, test_loader = get_dataloaders(CONFIG)
    
    model = FraudDetector(250, CONFIG).to(device)
    model.load_state_dict(torch.load(f"{CONFIG['paths']['models']}/best_model.pt"))
    
    probs, labels = get_predictions(model, test_loader, device)
    preds = (probs > 0.7).astype(float)
    
    plot_confusion_matrix(labels, preds, 'results/confusion_matrix.png')
    logger.info("Saved: results/confusion_matrix.png")
    
    plot_roc_curve(labels, probs, 'results/roc_curve.png')
    logger.info("Saved: results/roc_curve.png")
    
    plot_precision_recall(labels, probs, 'results/pr_curve.png')
    logger.info("Saved: results/pr_curve.png")
    
    plot_threshold_analysis(labels, probs, 'results/threshold_analysis.png')
    logger.info("Saved: results/threshold_analysis.png")
    
    plot_score_distribution(labels, probs, 'results/score_distribution.png')
    logger.info("Saved: results/score_distribution.png")
    
    logger.info("All visualizations complete")

if __name__ == '__main__':
    main()