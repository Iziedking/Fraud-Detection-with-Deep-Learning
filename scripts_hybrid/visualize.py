import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (confusion_matrix, roc_curve, precision_recall_curve,
                             f1_score, precision_score, recall_score, roc_auc_score)
import seaborn as sns
import os

from config import CONFIG
from data_loader import get_dataloaders
from model import HybridFraudDetector

def get_predictions(model, loader, device):
    model.eval()
    all_probs, all_labels = [], []
    
    with torch.no_grad():
        for x_temp, x_behav, y in loader:
            x_temp, x_behav = x_temp.to(device), x_behav.to(device)
            probs = model(x_temp, x_behav)
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(y.numpy())
    
    return np.array(all_probs), np.array(all_labels)

def main():
    os.makedirs(CONFIG['paths']['results'], exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    _, _, test_loader, dims = get_dataloaders(CONFIG)
    
    model = HybridFraudDetector(dims['temporal'], dims['behavioral'], CONFIG).to(device)
    model.load_state_dict(torch.load(f"{CONFIG['paths']['models']}/hybrid_model.pt"))
    
    probs, labels = get_predictions(model, test_loader, device)
    preds = (probs > 0.7).astype(float)
    
    # Confusion Matrix
    cm = confusion_matrix(labels, preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Legitimate', 'Fraud'],
                yticklabels=['Legitimate', 'Fraud'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix (Hybrid Model)')
    plt.tight_layout()
    plt.savefig(f"{CONFIG['paths']['results']}/confusion_matrix.png", dpi=150)
    plt.close()
    
    # ROC Curve
    fpr, tpr, _ = roc_curve(labels, probs)
    auc_score = roc_auc_score(labels, probs)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, 'b-', linewidth=2, label=f'AUC = {auc_score:.4f}')
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve (Hybrid Model)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{CONFIG['paths']['results']}/roc_curve.png", dpi=150)
    plt.close()
    
    # Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(labels, probs)
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, 'b-', linewidth=2)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve (Hybrid Model)')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{CONFIG['paths']['results']}/pr_curve.png", dpi=150)
    plt.close()
    
    # Threshold Analysis
    thresholds = np.arange(0.1, 0.95, 0.05)
    f1_scores, precisions, recalls = [], [], []
    for thresh in thresholds:
        p = (probs > thresh).astype(float)
        f1_scores.append(f1_score(labels, p))
        precisions.append(precision_score(labels, p))
        recalls.append(recall_score(labels, p))
    
    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, f1_scores, 'b-', label='F1 Score', linewidth=2)
    plt.plot(thresholds, precisions, 'g-', label='Precision', linewidth=2)
    plt.plot(thresholds, recalls, 'r-', label='Recall', linewidth=2)
    plt.xlabel('Threshold')
    plt.ylabel('Score')
    plt.title('Metrics vs Threshold (Hybrid Model)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{CONFIG['paths']['results']}/threshold_analysis.png", dpi=150)
    plt.close()
    
    # Score Distribution
    plt.figure(figsize=(10, 6))
    plt.hist(probs[labels == 0], bins=50, alpha=0.5, label='Legitimate', color='green')
    plt.hist(probs[labels == 1], bins=50, alpha=0.5, label='Fraud', color='red')
    plt.xlabel('Predicted Probability')
    plt.ylabel('Count')
    plt.title('Score Distribution (Hybrid Model)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{CONFIG['paths']['results']}/score_distribution.png", dpi=150)
    plt.close()
    
    print(f"Saved all visualizations to {CONFIG['paths']['results']}/")
    print(f"\nFinal Metrics (threshold=0.7):")
    print(f"  F1: {f1_score(labels, preds):.4f}")
    print(f"  Precision: {precision_score(labels, preds):.4f}")
    print(f"  Recall: {recall_score(labels, preds):.4f}")
    print(f"  AUC: {auc_score:.4f}")

if __name__ == '__main__':
    main()