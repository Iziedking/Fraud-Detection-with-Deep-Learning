import pandas as pd
import numpy as np
import json

print("Loading data...")
df = pd.read_parquet("data_files/train_final.parquet")

with open("data_files/feature_groups.json", "r") as f:
    groups = json.load(f)

relational_cols = [c for c in groups['relational'] if c in df.columns]
print(f"Relational features: {len(relational_cols)}")
print(f"Columns: {relational_cols[:10]}...")

X_relational = df[relational_cols].values.astype('float32')
y = df['isFraud'].values.astype('float32')

print(f"\nRelational data shape: {X_relational.shape}")
print(f"Fraud rate: {y.mean()*100:.2f}%")

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

X_train, X_temp, y_train, y_temp = train_test_split(
    X_relational, y, test_size=0.30, random_state=42, stratify=y
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

print(f"\nTrain: {len(X_train):,} ({y_train.mean()*100:.2f}% fraud)")
print(f"Val: {len(X_val):,} ({y_val.mean()*100:.2f}% fraud)")
print(f"Test: {len(X_test):,} ({y_test.mean()*100:.2f}% fraud)")

from imblearn.over_sampling import SMOTE

print("\nApplying SMOTE to training data...")
smote = SMOTE(sampling_strategy=0.5, random_state=42, k_neighbors=5)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

print(f"Train after SMOTE: {len(X_train_smote):,} ({y_train_smote.mean()*100:.2f}% fraud)")

np.save("data_files/X_train_gnn.npy", X_train_smote.astype('float32'))
np.save("data_files/y_train_gnn.npy", y_train_smote.astype('float32'))
np.save("data_files/X_val_gnn.npy", X_val.astype('float32'))
np.save("data_files/y_val_gnn.npy", y_val.astype('float32'))
np.save("data_files/X_test_gnn.npy", X_test.astype('float32'))
np.save("data_files/y_test_gnn.npy", y_test.astype('float32'))

print("\nSaved GNN data files to data_files/")