import pandas as pd
from imblearn.over_sampling import SMOTE

X_train = pd.read_parquet("data_files/X_train_scaled.parquet")
y_train = pd.read_parquet("data_files/y_train.parquet")['isFraud']

print(f"Before SMOTE:")
print(f"  Total: {len(X_train):,}")
print(f"  Fraud: {y_train.sum():,} ({y_train.mean()*100:.2f}%)")
print(f"  Legit: {(y_train==0).sum():,}")

smote = SMOTE(sampling_strategy=0.5, random_state=42, k_neighbors=5)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

X_resampled = pd.DataFrame(X_resampled, columns=X_train.columns)
y_resampled = pd.Series(y_resampled, name='isFraud')

print(f"\nAfter SMOTE:")
print(f"  Total: {len(X_resampled):,}")
print(f"  Fraud: {y_resampled.sum():,} ({y_resampled.mean()*100:.2f}%)")
print(f"  Legit: {(y_resampled==0).sum():,}")

X_resampled.to_parquet("data_files/X_train_resampled.parquet", index=False)
y_resampled.to_frame().to_parquet("data_files/y_train_resampled.parquet", index=False)

print(f"\nSaved resampled training data")