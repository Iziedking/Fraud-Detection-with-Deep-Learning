import torch
import torch.nn as nn
import torch.nn.functional as F

class GraphAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = out_features // num_heads
        
        self.W = nn.Linear(in_features, out_features)
        self.a = nn.Linear(2 * self.head_dim, 1)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(out_features)
    
    def forward(self, x):
        batch_size = x.size(0)
        h = self.W(x)
        h = h.view(batch_size, self.num_heads, self.head_dim)
        
        attn_input = torch.cat([
            h.unsqueeze(2).expand(-1, -1, self.num_heads, -1),
            h.unsqueeze(1).expand(-1, self.num_heads, -1, -1)
        ], dim=-1)
        
        attn_weights = F.leaky_relu(self.a(attn_input).squeeze(-1), negative_slope=0.2)
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        out = torch.bmm(attn_weights.view(batch_size, self.num_heads, self.num_heads), 
                        h.view(batch_size, self.num_heads, self.head_dim))
        out = out.view(batch_size, -1)
        out = self.layer_norm(out)
        
        return out

class GNNFraudDetector(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        input_dim = config['model']['input_dim']
        hidden_dim = config['model']['hidden_dim']
        num_heads = config['model']['num_heads']
        num_layers = config['model']['num_layers']
        dropout = config['model']['dropout']
        
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.gat_layers = nn.ModuleList([
            GraphAttentionLayer(hidden_dim, hidden_dim, dropout, num_heads)
            for _ in range(num_layers)
        ])
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1)
        )
    
    def forward(self, x):
        h = self.input_proj(x)
        
        for gat in self.gat_layers:
            h = h + gat(h)
        
        out = torch.sigmoid(self.classifier(h))
        return out.squeeze()