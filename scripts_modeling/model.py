import torch
import torch.nn as nn

class FraudDetector(nn.Module):
    def __init__(self, input_dim, config):
        super().__init__()
        
        hidden_dims = config['model']['dense_hidden']
        dropout = config['model']['dropout']
        
        layers = []
        prev_dim = input_dim
        
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = h_dim
        
        self.encoder = nn.Sequential(*layers)
        self.classifier = nn.Linear(prev_dim, 1)
    
    def forward(self, x):
        features = self.encoder(x)
        output = torch.sigmoid(self.classifier(features))
        return output.squeeze()