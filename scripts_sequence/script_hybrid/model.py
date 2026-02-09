import torch
import torch.nn as nn
import torch.nn.functional as F

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
    
    def get_embeddings(self, x_temp, x_rel, x_beh):
        temp_out = self.temporal(x_temp)
        rel_out = self.relational(x_rel)
        beh_out = self.behavioral(x_beh)
        return temp_out, rel_out, beh_out