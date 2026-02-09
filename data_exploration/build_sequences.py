import pandas as pd
import numpy as np

SEQ_LENGTH = 5
MIN_HISTORY = 3

print("Loading data...")
df = pd.read_parquet("data_files/train_final.parquet")
df = df.sort_values(['card1', 'TransactionDT']).reset_index(drop=True)

feature_cols = [c for c in df.columns if c not in ['isFraud', 'card1']]
n_features = len(feature_cols)

print(f"Features per transaction: {n_features}")
print(f"Sequence length: {SEQ_LENGTH}")
print(f"Building sequences...")

sequences = []
labels = []
card_groups = df.groupby('card1')

for card_id, group in card_groups:
    if len(group) < MIN_HISTORY + 1:
        continue
    
    group = group.sort_values('TransactionDT')
    values = group[feature_cols].values
    targets = group['isFraud'].values
    
    for i in range(MIN_HISTORY, len(group)):
        start_idx = max(0, i - SEQ_LENGTH)
        seq = values[start_idx:i]
        
        if len(seq) < SEQ_LENGTH:
            pad_size = SEQ_LENGTH - len(seq)
            padding = np.zeros((pad_size, n_features))
            seq = np.vstack([padding, seq])
        
        sequences.append(seq)
        labels.append(targets[i])

sequences = np.array(sequences, dtype='float32')
labels = np.array(labels, dtype='float32')

print(f"\n=== Sequence Dataset ===")
print(f"Total sequences: {len(sequences):,}")
print(f"Sequence shape: {sequences.shape}")
print(f"Fraud rate: {labels.mean()*100:.2f}%")
print(f"Fraud count: {labels.sum():,.0f}")

np.save("data_files/X_sequences.npy", sequences)
np.save("data_files/y_sequences.npy", labels)
print("\nSaved: data_files/X_sequences.npy")
print("Saved: data_files/y_sequences.npy")