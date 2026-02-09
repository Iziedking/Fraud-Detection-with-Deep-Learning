import numpy as np
from imblearn.over_sampling import SMOTE

print("Loading sequence data...")
X_train = np.load("data_files/X_train_seq.npy")
y_train = np.load("data_files/y_train_seq.npy")

n_samples, seq_len, n_features = X_train.shape
print(f"Original shape: {X_train.shape}")
print(f"Fraud rate: {y_train.mean()*100:.2f}% ({y_train.sum():.0f} fraud)")

X_flat = X_train.reshape(n_samples, seq_len * n_features)
print(f"Flattened shape: {X_flat.shape}")

print("\nApplying SMOTE (this may take a few minutes)...")
smote = SMOTE(sampling_strategy=0.5, random_state=42, k_neighbors=5)
X_resampled, y_resampled = smote.fit_resample(X_flat, y_train)

X_resampled = X_resampled.reshape(-1, seq_len, n_features).astype('float32')
y_resampled = y_resampled.astype('float32')

print(f"\nResampled shape: {X_resampled.shape}")
print(f"Fraud rate: {y_resampled.mean()*100:.2f}% ({y_resampled.sum():.0f} fraud)")
print(f"Legit count: {(y_resampled==0).sum():.0f}")

np.save("data_files/X_train_seq_smote.npy", X_resampled)
np.save("data_files/y_train_seq_smote.npy", y_resampled)

print("\nSaved:")
print("  data_files/X_train_seq_smote.npy")
print("  data_files/y_train_seq_smote.npy")