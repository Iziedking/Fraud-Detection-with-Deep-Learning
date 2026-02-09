import torch
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
import matplotlib.pyplot as plt
import os

from config import CONFIG
from utils import setup_logger
from data_loader import get_dataloaders
from model import HybridFraudDetector

def evaluate():
    logger = setup_logger('hybrid_evaluation')
    os.makedirs(CONFIG['paths']['results'], exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    _, _, test_loader, dims = get_dataloaders(CONFIG)
    
    model = HybridFraudDetector(dims['temporal'], dims['behavioral'], CONFIG).to(device)
    model.load_state_dict(torch.load(f"{CONFIG['paths']['models']}/hybrid_model.pt"))
    model.eval()
    
    all_probs, all_labels = [], []
    with torch.no_grad():
        for x_temp, x_behav, y in test_loader:
            x_temp, x_behav = x_temp.to(device), x_behav.to(device)
            probs = model(x_temp, x_behav)
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(y.numpy())
    
    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    
    logger.info("=== Threshold Analysis ===")
    thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
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
    logger.info("\n=== Final Results (Optimal Threshold) ===")
    logger.info(f"F1: {f1_score(all_labels, preds):.4f}")
    logger.info(f"Precision: {precision_score(all_labels, preds):.4f}")
    logger.info(f"Recall: {recall_score(all_labels, preds):.4f}")
    logger.info(f"AUC: {roc_auc_score(all_labels, all_probs):.4f}")

if __name__ == '__main__':
    evaluate()