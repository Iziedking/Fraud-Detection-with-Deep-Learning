import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

print("Loading sequences...")
X = np.load("data_files/X_sequences.npy")
y = np.load("data_files/y_sequences.npy")

print(f"Original shape: {X.shape}")
print(f"Fraud rate: {y.mean()*100:.2f}%")

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
)

print(f"\nTrain: {len(X_train):,} ({y_train.mean()*100:.2f}% fraud)")
print(f"Val:   {len(X_val):,} ({y_val.mean()*100:.2f}% fraud)")
print(f"Test:  {len(X_test):,} ({y_test.mean()*100:.2f}% fraud)")

n_samples, seq_len, n_features = X_train.shape
X_train_flat = X_train.reshape(-1, n_features)

scaler = StandardScaler()
X_train_flat = scaler.fit_transform(X_train_flat)
X_train = X_train_flat.reshape(n_samples, seq_len, n_features)

X_val_flat = X_val.reshape(-1, n_features)
X_val_flat = scaler.transform(X_val_flat)
X_val = X_val_flat.reshape(len(X_val), seq_len, n_features)

X_test_flat = X_test.reshape(-1, n_features)
X_test_flat = scaler.transform(X_test_flat)
X_test = X_test_flat.reshape(len(X_test), seq_len, n_features)

print(f"\nScaled shape: {X_train.shape}")
print(f"Train sample mean: {X_train[0, 0, :5].round(3)}")

np.save("data_files/X_train_seq.npy", X_train.astype('float32'))
np.save("data_files/X_val_seq.npy", X_val.astype('float32'))
np.save("data_files/X_test_seq.npy", X_test.astype('float32'))
np.save("data_files/y_train_seq.npy", y_train.astype('float32'))
np.save("data_files/y_val_seq.npy", y_val.astype('float32'))
np.save("data_files/y_test_seq.npy", y_test.astype('float32'))

print("\nSaved sequence datasets to data_files/")