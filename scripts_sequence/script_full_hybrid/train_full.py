import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
import os
import logging
from datetime import datetime
import numpy as np
from torch.utils.data import Dataset, DataLoader

from config_full import CONFIG

def setup_logger(name, log_dir="logs"):
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"{name}_{timestamp}.log")
    
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers = []
    
    file_handler = logging.FileHandler(log_file)
    console_handler = logging.StreamHandler()
    
    formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.75, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, inputs, targets):
        bce = F.binary_cross_entropy(inputs, targets, reduction='none')
        pt = torch.where(targets == 1, inputs, 1 - inputs)
        alpha_t = torch.where(targets == 1, self.alpha, 1 - self.alpha)
        focal_weight = alpha_t * (1 - pt) ** self.gamma
        loss = focal_weight * bce
        return loss.mean()

class FusionDataset(Dataset):
    def __init__(self, temp_path, rel_path, beh_path, y_path):
        self.X_temp = np.load(temp_path)
        self.X_rel = np.load(rel_path)
        self.X_beh = np.load(beh_path)
        self.y = np.load(y_path)
    
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return (
            torch.tensor(self.X_temp[idx]),
            torch.tensor(self.X_rel[idx]),
            torch.tensor(self.X_beh[idx]),
            torch.tensor(self.y[idx])
        )

class TemporalBranch(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, dropout):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        self.layer_norm = nn.LayerNorm(hidden_dim * 2)
        self.output_dim = hidden_dim * 2
    
    def forward(self, x):
        x = x.unsqueeze(1)
        lstm_out, (h_n, _) = self.lstm(x)
        out = torch.cat([h_n[-2], h_n[-1]], dim=1)
        out = self.layer_norm(out)
        return out

class RelationalBranch(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_heads, dropout):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )
        self.output_dim = hidden_dim
    
    def forward(self, x):
        x = self.input_proj(x)
        x = x.unsqueeze(1)
        attn_out, _ = self.attention(x, x, x)
        attn_out = attn_out.squeeze(1)
        out = self.layer_norm(attn_out + self.ffn(attn_out))
        return out

class BehavioralBranch(nn.Module):
    def __init__(self, input_dim, hidden_dims, dropout):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = h_dim
        self.network = nn.Sequential(*layers)
        self.output_dim = hidden_dims[-1]
    
    def forward(self, x):
        return self.network(x)

class FusionModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        cfg = config['model']
        
        self.temporal = TemporalBranch(
            input_dim=cfg['temporal_dim'],
            hidden_dim=cfg['lstm_hidden'],
            num_layers=cfg['lstm_layers'],
            dropout=cfg['dropout']
        )
        
        self.relational = RelationalBranch(
            input_dim=cfg['relational_dim'],
            hidden_dim=cfg['gnn_hidden'],
            num_heads=cfg['gnn_heads'],
            dropout=cfg['dropout']
        )
        
        self.behavioral = BehavioralBranch(
            input_dim=cfg['behavioral_dim'],
            hidden_dims=cfg['dense_hidden'],
            dropout=cfg['dropout']
        )
        
        fusion_input = (
            self.temporal.output_dim +
            self.relational.output_dim +
            self.behavioral.output_dim
        )
        
        self.fusion = nn.Sequential(
            nn.Linear(fusion_input, cfg['fusion_hidden']),
            nn.BatchNorm1d(cfg['fusion_hidden']),
            nn.ReLU(),
            nn.Dropout(cfg['dropout']),
            nn.Linear(cfg['fusion_hidden'], cfg['fusion_hidden'] // 2),
            nn.ReLU(),
            nn.Dropout(cfg['dropout']),
            nn.Linear(cfg['fusion_hidden'] // 2, 1)
        )
    
    def forward(self, x_temp, x_rel, x_beh):
        temp_out = self.temporal(x_temp)
        rel_out = self.relational(x_rel)
        beh_out = self.behavioral(x_beh)
        
        fused = torch.cat([temp_out, rel_out, beh_out], dim=1)
        out = torch.sigmoid(self.fusion(fused))
        return out.squeeze()

def get_dataloaders(config):
    train_ds = FusionDataset(
        config['data']['train_temp'],
        config['data']['train_rel'],
        config['data']['train_beh'],
        config['data']['train_y']
    )
    val_ds = FusionDataset(
        config['data']['val_temp'],
        config['data']['val_rel'],
        config['data']['val_beh'],
        config['data']['val_y']
    )
    test_ds = FusionDataset(
        config['data']['test_temp'],
        config['data']['test_rel'],
        config['data']['test_beh'],
        config['data']['test_y']
    )
    
    train_loader = DataLoader(train_ds, batch_size=config['training']['batch_size'], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=config['training']['batch_size'], shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=config['training']['batch_size'], shuffle=False)
    
    return train_loader, val_loader, test_loader

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
    logger = setup_logger('fusion_full_training')
    logger.info("Starting FULL FEATURE Fusion Model Training")
    logger.info(f"Features: Temporal={CONFIG['model']['temporal_dim']}, Relational={CONFIG['model']['relational_dim']}, Behavioral={CONFIG['model']['behavioral_dim']}")
    logger.info(f"Total features: {CONFIG['model']['temporal_dim'] + CONFIG['model']['relational_dim'] + CONFIG['model']['behavioral_dim']}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Device: {device}")
    
    train_loader, val_loader, test_loader = get_dataloaders(CONFIG)
    logger.info(f"Train batches: {len(train_loader)}")
    
    model = FusionModel(CONFIG).to(device)
    params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {params:,}")
    
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
            torch.save(model.state_dict(), f"{CONFIG['paths']['models']}/fusion_model_full.pt")
            logger.info(f"New best model saved (F1: {best_f1:.4f})")
        else:
            patience_counter += 1
        
        if patience_counter >= CONFIG['training']['early_stop_patience']:
            logger.info(f"Early stopping at epoch {epoch+1}")
            break
    
    model.load_state_dict(torch.load(f"{CONFIG['paths']['models']}/fusion_model_full.pt"))
    test_metrics, probs, labels = evaluate(model, test_loader, criterion, device)
    
    logger.info("=" * 70)
    logger.info("TEST RESULTS (FULL FEATURE Fusion: LSTM + GNN + Dense)")
    logger.info("=" * 70)
    logger.info(f"F1: {test_metrics['f1']:.4f}")
    logger.info(f"Precision: {test_metrics['precision']:.4f}")
    logger.info(f"Recall: {test_metrics['recall']:.4f}")
    logger.info(f"AUC: {test_metrics['auc']:.4f}")
    
    logger.info("\nThreshold Analysis:")
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