import pandas as pd
import numpy as np

df = pd.read_parquet("data_files/train_final.parquet")

target = 'isFraud'
features = [c for c in df.columns if c != target]

# Calculate correlation with fraud
correlations = df[features].corrwith(df[target]).abs().sort_values(ascending=False)

print("=" * 60)
print("TOP 30 FEATURES BY CORRELATION WITH FRAUD")
print("=" * 60)
for i, (feat, corr) in enumerate(correlations.head(30).items(), 1):
    print(f"{i:2}. {feat:20} : {corr:.4f}")

print("\n" + "=" * 60)
print("FEATURE GROUPS FOR HYBRID MODEL")
print("=" * 60)

# Group features by purpose (aligned with project methodology)
feature_mapping = {
    'TEMPORAL (for LSTM)': {
        'Transaction Time': ['TransactionDT'],
        'Time Deltas': [c for c in features if c.startswith('D')],
        'Velocity/Counts': [c for c in features if c.startswith('C')],
    },
    'RELATIONAL (for GNN)': {
        'Card Info': [c for c in features if c.startswith('card')],
        'Address': [c for c in features if c in ['addr1', 'dist1', 'dist2']],
        'Email': [c for c in features if 'email' in c.lower()],
        'Device': [c for c in features if 'device' in c.lower()],
        'Identity': [c for c in features if c.startswith('id_')],
    },
    'BEHAVIORAL (for Dense)': {
        'Amount': ['TransactionAmt'],
        'Product': ['ProductCD'],
        'Match Flags': [c for c in features if c.startswith('M')],
        'Vesta Features': [c for c in features if c.startswith('V')],
    }
}

for branch, groups in feature_mapping.items():
    print(f"\n{branch}")
    print("-" * 40)
    branch_features = []
    for group_name, cols in groups.items():
        existing = [c for c in cols if c in features]
        branch_features.extend(existing)
        if existing:
            # Get avg correlation for this group
            avg_corr = correlations[existing].mean()
            print(f"  {group_name}: {len(existing)} features (avg corr: {avg_corr:.4f})")
    
    # Top 5 in this branch
    branch_corrs = correlations[branch_features].sort_values(ascending=False)
    print(f"  >> Top 5: {branch_corrs.head(5).index.tolist()}")

print("\n" + "=" * 60)
print("SUMMARY FOR MODEL ARCHITECTURE")
print("=" * 60)

temporal_feats = ['TransactionDT'] + [c for c in features if c.startswith('D') or c.startswith('C')]
relational_feats = [c for c in features if c.startswith('card') or c.startswith('id_') or 'device' in c.lower() or 'email' in c.lower() or c in ['addr1', 'dist1', 'dist2']]
behavioral_feats = ['TransactionAmt', 'ProductCD'] + [c for c in features if c.startswith('M') or c.startswith('V')]

print(f"Temporal features (LSTM): {len(temporal_feats)}")
print(f"Relational features (GNN): {len(relational_feats)}")
print(f"Behavioral features (Dense): {len(behavioral_feats)}")
print(f"Total: {len(temporal_feats) + len(relational_feats) + len(behavioral_feats)}")