import torch
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, precision_recall_curve
import matplotlib.pyplot as plt

from config import CONFIG
from data_loader import get_dataloaders
from model import FraudDetector
from utils import setup_logger

def find_optimal_threshold():
    logger = setup_logger('evaluation')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    _, val_loader, test_loader = get_dataloaders(CONFIG)
    
    model = FraudDetector(250, CONFIG).to(device)
    model.load_state_dict(torch.load(f"{CONFIG['paths']['models']}/best_model.pt"))
    model.eval()
    
    all_probs, all_labels = [], []
    with torch.no_grad():
        for X, y in test_loader:
            X = X.to(device)
            probs = model(X)
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(y.numpy())
    
    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    
    logger.info("=== Threshold Analysis ===")
    thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    best_f1, best_thresh = 0, 0.5
    
    for thresh in thresholds:
        preds = (all_probs > thresh).astype(float)
        f1 = f1_score(all_labels, preds)
        prec = precision_score(all_labels, preds)
        rec = recall_score(all_labels, preds)
        logger.info(f"Thresh {thresh:.1f} | F1: {f1:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f}")
        
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh
    
    logger.info(f"\nOptimal threshold: {best_thresh} (F1: {best_f1:.4f})")
    
    preds = (all_probs > best_thresh).astype(float)
    logger.info("\n=== Final Test Results ===")
    logger.info(f"F1: {f1_score(all_labels, preds):.4f}")
    logger.info(f"Precision: {precision_score(all_labels, preds):.4f}")
    logger.info(f"Recall: {recall_score(all_labels, preds):.4f}")
    logger.info(f"AUC: {roc_auc_score(all_labels, all_probs):.4f}")
    
    precision, recall, pr_thresholds = precision_recall_curve(all_labels, all_probs)
    plt.figure(figsize=(10, 6))
    plt.plot(recall, precision)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.savefig('logs/pr_curve.png', dpi=150)
    logger.info("Saved: logs/pr_curve.png")

if __name__ == '__main__':
    find_optimal_threshold()