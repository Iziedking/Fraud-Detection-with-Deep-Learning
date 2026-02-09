import pandas as pd

# Load and clean data
df = pd.read_parquet("data_files/train.parquet")
cols_to_drop = ['Unnamed: 0', 'TransactionID_x', 'TransactionID_y']
df_clean = df.drop(columns=cols_to_drop)

target = 'isFraud'
features = [col for col in df_clean.columns if col != target]

print(f"Total features: {len(features)}")
print(f"Total rows: {len(df_clean)}")

# 1. Near-constant features (>99% same value)
print("\n=== Near-constant features (>99% same value) ===")
near_constant = []
for col in features:
    top_freq = df_clean[col].value_counts(normalize=True).iloc[0]
    if top_freq > 0.99:
        near_constant.append(col)

print(f"Found: {len(near_constant)} features")
print(f"Examples: {near_constant[:10]}")

# 2. Zero correlation with target
print("\n=== Features with ZERO correlation to fraud ===")
correlations = df_clean[features].corrwith(df_clean[target]).abs()
zero_corr = correlations[correlations < 0.001].index.tolist()

print(f"Found: {len(zero_corr)} features")
print(f"Examples: {zero_corr[:10]}")

# 3. Summary
print("\n=== Summary ===")
print(f"Near-constant: {len(near_constant)}")
print(f"Zero correlation: {len(zero_corr)}")

# 4. Combine features to drop (remove duplicates)
features_to_drop = list(set(near_constant + zero_corr))
print(f"\nTotal unique features to drop: {len(features_to_drop)}")

# 5. Create final clean dataframe
df_final = df_clean.drop(columns=features_to_drop)
print(f"\nFinal dataset:")
print(f"  Rows: {len(df_final)}")
print(f"  Columns: {df_final.shape[1]}")
print(f"  Features: {df_final.shape[1] - 1}")

# 6. Save to parquet
df_final.to_parquet("data_files/train_clean.parquet", index=False)
print(f"\nSaved to: data_files/train_clean.parquet")

# 7. Show remaining feature groups
remaining_features = [c for c in df_final.columns if c != 'isFraud']
print(f"\n=== Remaining feature breakdown ===")
groups = {
    'Transaction': ['TransactionDT', 'TransactionAmt'],
    'Card': [c for c in remaining_features if c.startswith('card')],
    'Address': [c for c in remaining_features if c in ['addr1', 'addr2', 'dist1', 'dist2']],
    'Email': [c for c in remaining_features if 'email' in c.lower()],
    'Count (C)': [c for c in remaining_features if c.startswith('C') and c[1:].isdigit()],
    'Time Delta (D)': [c for c in remaining_features if c.startswith('D') and c[1:].isdigit()],
    'Match (M)': [c for c in remaining_features if c.startswith('M') and c[1:].isdigit()],
    'Vesta (V)': [c for c in remaining_features if c.startswith('V') and c[1:].isdigit()],
    'Identity (id)': [c for c in remaining_features if c.startswith('id_')],
    'Device': [c for c in remaining_features if 'device' in c.lower()]
}

for name, cols in groups.items():
    if cols:
        print(f"  {name}: {len(cols)}")