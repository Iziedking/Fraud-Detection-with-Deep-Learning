import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (confusion_matrix, roc_curve, precision_recall_curve,
                             f1_score, precision_score, recall_score, roc_auc_score)
import seaborn as sns
import os

from config import CONFIG
from data_loader import get_dataloaders
from model import FusionModel

def get_predictions(model, loader, device):
    model.eval()
    all_probs, all_labels = [], []
    
    with torch.no_grad():
        for x_temp, x_rel, x_beh, y in loader:
            x_temp = x_temp.to(device)
            x_rel = x_rel.to(device)
            x_beh = x_beh.to(device)
            probs = model(x_temp, x_rel, x_beh)
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(y.numpy())
    
    return np.array(all_probs), np.array(all_labels)

def main():
    os.makedirs(CONFIG['paths']['results'], exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    _, _, test_loader = get_dataloaders(CONFIG)
    
    model = FusionModel(CONFIG).to(device)
    model.load_state_dict(torch.load(f"{CONFIG['paths']['models']}/fusion_model.pt"))
    
    probs, labels = get_predictions(model, test_loader, device)
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # 1. Confusion Matrix (threshold 0.7)
    preds = (probs > 0.7).astype(float)
    cm = confusion_matrix(labels, preds)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 0],
                xticklabels=['Legitimate', 'Fraud'],
                yticklabels=['Legitimate', 'Fraud'])
    axes[0, 0].set_xlabel('Predicted')
    axes[0, 0].set_ylabel('Actual')
    axes[0, 0].set_title('Confusion Matrix (threshold=0.7)')
    
    # 2. ROC Curve
    fpr, tpr, _ = roc_curve(labels, probs)
    auc_score = roc_auc_score(labels, probs)
    axes[0, 1].plot(fpr, tpr, 'b-', linewidth=2, label=f'AUC = {auc_score:.4f}')
    axes[0, 1].plot([0, 1], [0, 1], 'k--', linewidth=1)
    axes[0, 1].set_xlabel('False Positive Rate')
    axes[0, 1].set_ylabel('True Positive Rate')
    axes[0, 1].set_title('ROC Curve')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(labels, probs)
    axes[0, 2].plot(recall, precision, 'b-', linewidth=2)
    axes[0, 2].set_xlabel('Recall')
    axes[0, 2].set_ylabel('Precision')
    axes[0, 2].set_title('Precision-Recall Curve')
    axes[0, 2].grid(True, alpha=0.3)
    
    # 4. Threshold Analysis
    thresholds = np.arange(0.1, 0.95, 0.05)
    f1_scores, precisions, recalls = [], [], []
    for thresh in thresholds:
        p = (probs > thresh).astype(float)
        f1_scores.append(f1_score(labels, p))
        precisions.append(precision_score(labels, p, zero_division=0))
        recalls.append(recall_score(labels, p))
    
    axes[1, 0].plot(thresholds, f1_scores, 'b-', label='F1 Score', linewidth=2)
    axes[1, 0].plot(thresholds, precisions, 'g-', label='Precision', linewidth=2)
    axes[1, 0].plot(thresholds, recalls, 'r-', label='Recall', linewidth=2)
    axes[1, 0].axvline(x=0.7, color='gray', linestyle='--', alpha=0.7)
    axes[1, 0].set_xlabel('Threshold')
    axes[1, 0].set_ylabel('Score')
    axes[1, 0].set_title('Metrics vs Threshold')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 5. Score Distribution
    axes[1, 1].hist(probs[labels == 0], bins=50, alpha=0.5, label='Legitimate', color='green')
    axes[1, 1].hist(probs[labels == 1], bins=50, alpha=0.5, label='Fraud', color='red')
    axes[1, 1].axvline(x=0.7, color='black', linestyle='--', label='Threshold=0.7')
    axes[1, 1].set_xlabel('Predicted Probability')
    axes[1, 1].set_ylabel('Count')
    axes[1, 1].set_title('Score Distribution')
    axes[1, 1].legend()
    
    # 6. Model Comparison
    models = ['Baseline\nDense', 'LSTM+Dense\nHybrid', 'Sequence\nLSTM', 'GNN', 'Fusion\n(Ours)']
    aucs = [0.9138, 0.9203, 0.7811, 0.7770, 0.9356]
    f1s = [0.6338, 0.6435, 0.2839, 0.1541, 0.6556]
    
    x = np.arange(len(models))
    width = 0.35
    axes[1, 2].bar(x - width/2, aucs, width, label='AUC', color='steelblue')
    axes[1, 2].bar(x + width/2, f1s, width, label='F1 (0.7)', color='coral')
    axes[1, 2].set_ylabel('Score')
    axes[1, 2].set_title('Model Comparison')
    axes[1, 2].set_xticks(x)
    axes[1, 2].set_xticklabels(models, fontsize=8)
    axes[1, 2].legend()
    axes[1, 2].set_ylim(0, 1.1)
    axes[1, 2].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(f"{CONFIG['paths']['results']}/fusion_analysis.png", dpi=150)
    plt.close()
    
    # Save individual plots
    # Confusion Matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Legitimate', 'Fraud'],
                yticklabels=['Legitimate', 'Fraud'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix - Fusion Model (threshold=0.7)')
    plt.tight_layout()
    plt.savefig(f"{CONFIG['paths']['results']}/confusion_matrix.png", dpi=150)
    plt.close()
    
    # ROC Curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, 'b-', linewidth=2, label=f'Fusion Model (AUC = {auc_score:.4f})')
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve - Fusion Model')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{CONFIG['paths']['results']}/roc_curve.png", dpi=150)
    plt.close()
    
    print(f"Saved visualizations to {CONFIG['paths']['results']}/")
    print(f"\nFinal Metrics (threshold=0.7):")
    print(f"  F1: {f1_score(labels, preds):.4f}")
    print(f"  Precision: {precision_score(labels, preds):.4f}")
    print(f"  Recall: {recall_score(labels, preds):.4f}")
    print(f"  AUC: {auc_score:.4f}")

if __name__ == '__main__':
    main()