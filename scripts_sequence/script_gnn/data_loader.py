import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class GNNDataset(Dataset):
    def __init__(self, X_path, y_path):
        self.X = np.load(X_path)
        self.y = np.load(y_path)
    
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return torch.tensor(self.X[idx]), torch.tensor(self.y[idx])

def get_dataloaders(config):
    train_ds = GNNDataset(config['data']['train_X'], config['data']['train_y'])
    val_ds = GNNDataset(config['data']['val_X'], config['data']['val_y'])
    test_ds = GNNDataset(config['data']['test_X'], config['data']['test_y'])
    
    train_loader = DataLoader(train_ds, batch_size=config['training']['batch_size'], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=config['training']['batch_size'], shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=config['training']['batch_size'], shuffle=False)
    
    return train_loader, val_loader, test_loader