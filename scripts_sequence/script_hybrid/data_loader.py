import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

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