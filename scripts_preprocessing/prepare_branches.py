import pandas as pd
import json

X_train = pd.read_parquet("data_files/X_train_resampled.parquet")
features = X_train.columns.tolist()

temporal_features = ['TransactionDT'] + [c for c in features if c.startswith('D') or c.startswith('C')]

relational_features = [c for c in features if 
    c.startswith('card') or 
    c.startswith('id_') or 
    c in ['addr1', 'dist1', 'dist2', 'P_emaildomain', 'R_emaildomain', 'DeviceType', 'DeviceInfo']]

behavioral_features = ['TransactionAmt', 'ProductCD'] + [c for c in features if 
    c.startswith('M') or c.startswith('V')]

feature_groups = {
    'temporal': temporal_features,
    'relational': relational_features,
    'behavioral': behavioral_features
}

print("Feature Groups for Hybrid Model:")
print(f"  Temporal (LSTM):     {len(temporal_features)} features")
print(f"  Relational (GNN):    {len(relational_features)} features")
print(f"  Behavioral (Dense):  {len(behavioral_features)} features")

total = len(temporal_features) + len(relational_features) + len(behavioral_features)
print(f"  Total:               {total} features")

with open("data_files/feature_groups.json", "w") as f:
    json.dump(feature_groups, f, indent=2)

print("\nSaved: data_files/feature_groups.json")