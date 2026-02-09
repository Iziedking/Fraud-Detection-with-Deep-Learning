import torch
import torch.nn as nn

class HybridFraudDetector(nn.Module):
    def __init__(self, temporal_dim, behavioral_dim, config):
        super().__init__()
        
        lstm_hidden = config['model']['lstm_hidden']
        lstm_layers = config['model']['lstm_layers']
        dense_hidden = config['model']['dense_hidden']
        dropout = config['model']['dropout']
        
        self.lstm = nn.LSTM(
            input_size=temporal_dim,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0
        )
        self.lstm_norm = nn.LayerNorm(lstm_hidden)
        
        behavioral_layers = []
        prev_dim = behavioral_dim
        for h_dim in dense_hidden:
            behavioral_layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = h_dim
        self.behavioral_encoder = nn.Sequential(*behavioral_layers)
        
        fusion_input = lstm_hidden + dense_hidden[-1]
        self.fusion = nn.Sequential(
            nn.Linear(fusion_input, config['model']['fusion_dim']),
            nn.BatchNorm1d(config['model']['fusion_dim']),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(config['model']['fusion_dim'], 64),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.classifier = nn.Linear(64, 1)
    
    def forward(self, x_temporal, x_behavioral):
        x_temporal = x_temporal.unsqueeze(1)
        lstm_out, (h_n, _) = self.lstm(x_temporal)
        temporal_features = self.lstm_norm(h_n[-1])
        
        behavioral_features = self.behavioral_encoder(x_behavioral)
        
        fused = torch.cat([temporal_features, behavioral_features], dim=1)
        fused = self.fusion(fused)
        
        output = torch.sigmoid(self.classifier(fused))
        return output.squeeze()