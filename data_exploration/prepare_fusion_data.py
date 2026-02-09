import pandas as pd
import numpy as np
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

print("Loading data...")
df = pd.read_parquet("data_files/train_final.parquet")

with open("data_files/feature_groups.json", "r") as f:
    groups = json.load(f)

temporal_cols = [c for c in groups['temporal'] if c in df.columns]
relational_cols = [c for c in groups['relational'] if c in df.columns]
behavioral_cols = [c for c in groups['behavioral'] if c in df.columns]

print(f"Temporal: {len(temporal_cols)}")
print(f"Relational: {len(relational_cols)}")
print(f"Behavioral: {len(behavioral_cols)}")

X_temporal = df[temporal_cols].values
X_relational = df[relational_cols].values
X_behavioral = df[behavioral_cols].values
y = df['isFraud'].values

print(f"\nTotal samples: {len(y):,}")
print(f"Fraud rate: {y.mean()*100:.2f}%")

indices = np.arange(len(y))
idx_train, idx_temp, y_train, y_temp = train_test_split(
    indices, y, test_size=0.30, random_state=42, stratify=y
)
idx_val, idx_test, y_val, y_test = train_test_split(
    idx_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
)

X_temp_train, X_temp_val, X_temp_test = X_temporal[idx_train], X_temporal[idx_val], X_temporal[idx_test]
X_rel_train, X_rel_val, X_rel_test = X_relational[idx_train], X_relational[idx_val], X_relational[idx_test]
X_beh_train, X_beh_val, X_beh_test = X_behavioral[idx_train], X_behavioral[idx_val], X_behavioral[idx_test]

scaler_temp = StandardScaler()
scaler_rel = StandardScaler()
scaler_beh = StandardScaler()

X_temp_train = scaler_temp.fit_transform(X_temp_train)
X_temp_val = scaler_temp.transform(X_temp_val)
X_temp_test = scaler_temp.transform(X_temp_test)

X_rel_train = scaler_rel.fit_transform(X_rel_train)
X_rel_val = scaler_rel.transform(X_rel_val)
X_rel_test = scaler_rel.transform(X_rel_test)

X_beh_train = scaler_beh.fit_transform(X_beh_train)
X_beh_val = scaler_beh.transform(X_beh_val)
X_beh_test = scaler_beh.transform(X_beh_test)

print(f"\nBefore SMOTE - Train: {len(y_train):,} ({y_train.mean()*100:.2f}% fraud)")

X_combined_train = np.hstack([X_temp_train, X_rel_train, X_beh_train])

smote = SMOTE(sampling_strategy=0.5, random_state=42, k_neighbors=5)
X_combined_smote, y_train_smote = smote.fit_resample(X_combined_train, y_train)

n_temp = len(temporal_cols)
n_rel = len(relational_cols)
n_beh = len(behavioral_cols)

X_temp_train_smote = X_combined_smote[:, :n_temp]
X_rel_train_smote = X_combined_smote[:, n_temp:n_temp+n_rel]
X_beh_train_smote = X_combined_smote[:, n_temp+n_rel:]

print(f"After SMOTE - Train: {len(y_train_smote):,} ({y_train_smote.mean()*100:.2f}% fraud)")
print(f"Val: {len(y_val):,} ({y_val.mean()*100:.2f}% fraud)")
print(f"Test: {len(y_test):,} ({y_test.mean()*100:.2f}% fraud)")

np.save("data_files/X_temp_train_fusion.npy", X_temp_train_smote.astype('float32'))
np.save("data_files/X_rel_train_fusion.npy", X_rel_train_smote.astype('float32'))
np.save("data_files/X_beh_train_fusion.npy", X_beh_train_smote.astype('float32'))
np.save("data_files/y_train_fusion.npy", y_train_smote.astype('float32'))

np.save("data_files/X_temp_val_fusion.npy", X_temp_val.astype('float32'))
np.save("data_files/X_rel_val_fusion.npy", X_rel_val.astype('float32'))
np.save("data_files/X_beh_val_fusion.npy", X_beh_val.astype('float32'))
np.save("data_files/y_val_fusion.npy", y_val.astype('float32'))

np.save("data_files/X_temp_test_fusion.npy", X_temp_test.astype('float32'))
np.save("data_files/X_rel_test_fusion.npy", X_rel_test.astype('float32'))
np.save("data_files/X_beh_test_fusion.npy", X_beh_test.astype('float32'))
np.save("data_files/y_test_fusion.npy", y_test.astype('float32'))

dims = {
    'temporal': n_temp,
    'relational': n_rel,
    'behavioral': n_beh
}
with open("data_files/fusion_dims.json", "w") as f:
    json.dump(dims, f)

print(f"\nSaved fusion data:")
print(f"  Temporal dim: {n_temp}")
print(f"  Relational dim: {n_rel}")
print(f"  Behavioral dim: {n_beh}")