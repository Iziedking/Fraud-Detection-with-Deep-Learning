import torch
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
import os

from config import CONFIG
from utils import setup_logger
from data_loader import get_dataloaders
from model import FusionModel
from losses import FocalLoss

def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds, all_probs, all_labels = [], [], []
    
    with torch.no_grad():
        for x_temp, x_rel, x_beh, y in loader:
            x_temp = x_temp.to(device)
            x_rel = x_rel.to(device)
            x_beh = x_beh.to(device)
            y = y.to(device)
            
            outputs = model(x_temp, x_rel, x_beh)
            loss = criterion(outputs, y)
            total_loss += loss.item()
            
            all_probs.extend(outputs.cpu().numpy())
            preds = (outputs > 0.5).float()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
    
    metrics = {
        'loss': total_loss / len(loader),
        'f1': f1_score(all_labels, all_preds),
        'precision': precision_score(all_labels, all_preds),
        'recall': recall_score(all_labels, all_preds),
        'auc': roc_auc_score(all_labels, all_probs)
    }
    return metrics, all_probs, all_labels

def train():
    logger = setup_logger('fusion_training')
    logger.info("Starting LSTM + GNN + Dense Fusion Model Training")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Device: {device}")
    
    train_loader, val_loader, test_loader = get_dataloaders(CONFIG)
    logger.info(f"Train batches: {len(train_loader)}")
    
    model = FusionModel(CONFIG).to(device)
    params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {params:,}")
    
    logger.info(f"Temporal branch output: {model.temporal.output_dim}")
    logger.info(f"Relational branch output: {model.relational.output_dim}")
    logger.info(f"Behavioral branch output: {model.behavioral.output_dim}")
    
    criterion = FocalLoss(alpha=0.75, gamma=2.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG['training']['learning_rate'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    
    best_f1 = 0
    patience_counter = 0
    
    for epoch in range(CONFIG['training']['epochs']):
        model.train()
        train_loss = 0
        
        for batch_idx, (x_temp, x_rel, x_beh, y) in enumerate(train_loader):
            x_temp = x_temp.to(device)
            x_rel = x_rel.to(device)
            x_beh = x_beh.to(device)
            y = y.to(device)
            
            optimizer.zero_grad()
            outputs = model(x_temp, x_rel, x_beh)
            loss = criterion(outputs, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            train_loss += loss.item()
            
            if batch_idx % 100 == 0:
                logger.info(f"Epoch {epoch+1} | Batch {batch_idx}/{len(train_loader)} | Loss: {loss.item():.4f}")
        
        train_loss /= len(train_loader)
        val_metrics, _, _ = evaluate(model, val_loader, criterion, device)
        scheduler.step(val_metrics['loss'])
        
        logger.info(f"Epoch {epoch+1} | Train Loss: {train_loss:.4f} | Val Loss: {val_metrics['loss']:.4f}")
        logger.info(f"Val F1: {val_metrics['f1']:.4f} | Prec: {val_metrics['precision']:.4f} | Rec: {val_metrics['recall']:.4f} | AUC: {val_metrics['auc']:.4f}")
        
        if val_metrics['f1'] > best_f1:
            best_f1 = val_metrics['f1']
            patience_counter = 0
            os.makedirs(CONFIG['paths']['models'], exist_ok=True)
            torch.save(model.state_dict(), f"{CONFIG['paths']['models']}/fusion_model.pt")
            logger.info(f"New best model saved (F1: {best_f1:.4f})")
        else:
            patience_counter += 1
        
        if patience_counter >= CONFIG['training']['early_stop_patience']:
            logger.info(f"Early stopping at epoch {epoch+1}")
            break
    
    model.load_state_dict(torch.load(f"{CONFIG['paths']['models']}/fusion_model.pt"))
    test_metrics, probs, labels = evaluate(model, test_loader, criterion, device)
    
    logger.info("=" * 60)
    logger.info("TEST RESULTS (LSTM + GNN + Dense Fusion)")
    logger.info("=" * 60)
    logger.info(f"F1: {test_metrics['f1']:.4f}")
    logger.info(f"Precision: {test_metrics['precision']:.4f}")
    logger.info(f"Recall: {test_metrics['recall']:.4f}")
    logger.info(f"AUC: {test_metrics['auc']:.4f}")
    
    logger.info("\nThreshold Analysis:")
    import numpy as np
    probs = np.array(probs)
    labels = np.array(labels)
    for thresh in [0.3, 0.5, 0.7, 0.8, 0.9]:
        preds = (probs > thresh).astype(float)
        f1 = f1_score(labels, preds)
        prec = precision_score(labels, preds)
        rec = recall_score(labels, preds)
        logger.info(f"Threshold {thresh}: F1={f1:.4f} | Prec={prec:.4f} | Rec={rec:.4f}")

if __name__ == '__main__':
    train()