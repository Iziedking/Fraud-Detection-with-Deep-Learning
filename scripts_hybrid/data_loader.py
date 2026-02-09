import pandas as pd
import json
import torch
from torch.utils.data import Dataset, DataLoader

class HybridFraudDataset(Dataset):
    def __init__(self, X_path, y_path, feature_groups_path):
        X = pd.read_parquet(X_path)
        self.y = pd.read_parquet(y_path).values.astype('float32').flatten()
        
        with open(feature_groups_path, 'r') as f:
            groups = json.load(f)
        
        temporal_cols = [c for c in groups['temporal'] if c in X.columns]
        behavioral_cols = [c for c in groups['behavioral'] if c in X.columns]
        
        self.X_temporal = X[temporal_cols].values.astype('float32')
        self.X_behavioral = X[behavioral_cols].values.astype('float32')
        
        self.temporal_dim = len(temporal_cols)
        self.behavioral_dim = len(behavioral_cols)
    
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return (
            torch.tensor(self.X_temporal[idx]),
            torch.tensor(self.X_behavioral[idx]),
            torch.tensor(self.y[idx])
        )

def get_dataloaders(config):
    fg_path = config['data']['feature_groups']
    
    train_ds = HybridFraudDataset(config['data']['train_X'], config['data']['train_y'], fg_path)
    val_ds = HybridFraudDataset(config['data']['val_X'], config['data']['val_y'], fg_path)
    test_ds = HybridFraudDataset(config['data']['test_X'], config['data']['test_y'], fg_path)
    
    train_loader = DataLoader(train_ds, batch_size=config['training']['batch_size'], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=config['training']['batch_size'], shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=config['training']['batch_size'], shuffle=False)
    
    dims = {'temporal': train_ds.temporal_dim, 'behavioral': train_ds.behavioral_dim}
    
    return train_loader, val_loader, test_loader, dims