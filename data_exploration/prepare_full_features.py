import pandas as pd
import numpy as np
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

print("Loading original data with ALL features...")
df = pd.read_csv("data_files/train.csv")

print(f"Original shape: {df.shape}")
print(f"Fraud rate: {df['isFraud'].mean()*100:.2f}%")

y = df['isFraud'].values

exclude_cols = ['TransactionID', 'isFraud']
feature_cols = [c for c in df.columns if c not in exclude_cols]

print(f"\nTotal features: {len(feature_cols)}")

temporal_cols = ['TransactionDT'] + [c for c in feature_cols if c.startswith('D') and c[1:].isdigit()]
temporal_cols += [c for c in feature_cols if c.startswith('C') and c[1:].isdigit()]

relational_cols = [c for c in feature_cols if c.startswith('card')]
relational_cols += [c for c in feature_cols if c.startswith('addr')]
relational_cols += [c for c in feature_cols if c.startswith('dist')]
relational_cols += [c for c in feature_cols if c.startswith('P_emaildomain') or c.startswith('R_emaildomain')]
relational_cols += [c for c in feature_cols if c.startswith('id_')]
relational_cols += ['DeviceType', 'DeviceInfo'] if 'DeviceType' in feature_cols else []

behavioral_cols = ['TransactionAmt', 'ProductCD']
behavioral_cols += [c for c in feature_cols if c.startswith('M')]
behavioral_cols += [c for c in feature_cols if c.startswith('V')]

temporal_cols = [c for c in temporal_cols if c in feature_cols]
relational_cols = [c for c in relational_cols if c in feature_cols]
behavioral_cols = [c for c in behavioral_cols if c in feature_cols]

all_grouped = set(temporal_cols + relational_cols + behavioral_cols)
remaining = [c for c in feature_cols if c not in all_grouped]
behavioral_cols += remaining

print(f"\nFeature Groups:")
print(f"  Temporal: {len(temporal_cols)}")
print(f"  Relational: {len(relational_cols)}")
print(f"  Behavioral: {len(behavioral_cols)}")
print(f"  Total: {len(temporal_cols) + len(relational_cols) + len(behavioral_cols)}")

for col in feature_cols:
    if df[col].dtype == 'object':
        df[col] = pd.factorize(df[col])[0]

df[feature_cols] = df[feature_cols].fillna(-999)

X_temporal = df[temporal_cols].values.astype('float32')
X_relational = df[relational_cols].values.astype('float32')
X_behavioral = df[behavioral_cols].values.astype('float32')

print(f"\nSplitting data...")
indices = np.arange(len(y))
idx_train, idx_temp, y_train, y_temp = train_test_split(
    indices, y, test_size=0.30, random_state=42, stratify=y
)
idx_val, idx_test, y_val, y_test = train_test_split(
    idx_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
)

X_temp_train = X_temporal[idx_train]
X_temp_val = X_temporal[idx_val]
X_temp_test = X_temporal[idx_test]

X_rel_train = X_relational[idx_train]
X_rel_val = X_relational[idx_val]
X_rel_test = X_relational[idx_test]

X_beh_train = X_behavioral[idx_train]
X_beh_val = X_behavioral[idx_val]
X_beh_test = X_behavioral[idx_test]

print("Scaling features...")
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

print("Applying SMOTE (this may take a few minutes)...")
smote = SMOTE(sampling_strategy=0.5, random_state=42, k_neighbors=5)
X_combined_smote, y_train_smote = smote.fit_resample(X_combined_train, y_train)

n_temp = len(temporal_cols)
n_rel = len(relational_cols)
n_beh = len(behavioral_cols)

X_temp_train_smote = X_combined_smote[:, :n_temp].astype('float32')
X_rel_train_smote = X_combined_smote[:, n_temp:n_temp+n_rel].astype('float32')
X_beh_train_smote = X_combined_smote[:, n_temp+n_rel:].astype('float32')

print(f"After SMOTE - Train: {len(y_train_smote):,} ({y_train_smote.mean()*100:.2f}% fraud)")
print(f"Val: {len(y_val):,} ({y_val.mean()*100:.2f}% fraud)")
print(f"Test: {len(y_test):,} ({y_test.mean()*100:.2f}% fraud)")

np.save("data_files/X_temp_train_full.npy", X_temp_train_smote)
np.save("data_files/X_rel_train_full.npy", X_rel_train_smote)
np.save("data_files/X_beh_train_full.npy", X_beh_train_smote)
np.save("data_files/y_train_full.npy", y_train_smote.astype('float32'))

np.save("data_files/X_temp_val_full.npy", X_temp_val.astype('float32'))
np.save("data_files/X_rel_val_full.npy", X_rel_val.astype('float32'))
np.save("data_files/X_beh_val_full.npy", X_beh_val.astype('float32'))
np.save("data_files/y_val_full.npy", y_val.astype('float32'))

np.save("data_files/X_temp_test_full.npy", X_temp_test.astype('float32'))
np.save("data_files/X_rel_test_full.npy", X_rel_test.astype('float32'))
np.save("data_files/X_beh_test_full.npy", X_beh_test.astype('float32'))
np.save("data_files/y_test_full.npy", y_test.astype('float32'))

dims = {
    'temporal': n_temp,
    'relational': n_rel,
    'behavioral': n_beh,
    'temporal_cols': temporal_cols,
    'relational_cols': relational_cols,
    'behavioral_cols': behavioral_cols
}
with open("data_files/full_feature_dims.json", "w") as f:
    json.dump(dims, f, indent=2)

print(f"\nSaved full feature data:")
print(f"  Temporal dim: {n_temp}")
print(f"  Relational dim: {n_rel}")
print(f"  Behavioral dim: {n_beh}")
print(f"  Total: {n_temp + n_rel + n_beh}")