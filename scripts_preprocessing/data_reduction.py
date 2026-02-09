import pandas as pd

df = pd.read_parquet("data_files/train_clean.parquet")
print(f"Starting: {df.shape[1]} columns, {len(df)} rows")

target = 'isFraud'

# Get V features
v_cols = [c for c in df.columns if c.startswith('V')]
print(f"V features before: {len(v_cols)}")

# Calculate correlation with fraud
correlations = df[v_cols].corrwith(df[target]).abs()

# Keep only V features with correlation > 0.02
v_to_keep = correlations[correlations > 0.02].index.tolist()
v_to_drop = correlations[correlations <= 0.02].index.tolist()

print(f"V features to keep: {len(v_to_keep)}")
print(f"V features to drop: {len(v_to_drop)}")

# Drop low-correlation V features
df_final = df.drop(columns=v_to_drop)

# summary
features_final = [c for c in df_final.columns if c != target]
print(f"\n=== Final Dataset ===")
print(f"Total columns: {df_final.shape[1]}")
print(f"Total features: {len(features_final)}")

# Breakdown
print(f"\n=== Feature Breakdown ===")
groups = {
    'Transaction': [c for c in features_final if c.startswith('Transaction')],
    'Card': [c for c in features_final if c.startswith('card')],
    'Address': [c for c in features_final if c in ['addr1', 'dist1', 'dist2']],
    'Email': [c for c in features_final if 'email' in c.lower()],
    'Count (C)': [c for c in features_final if c.startswith('C') and c[1:].isdigit()],
    'Time Delta (D)': [c for c in features_final if c.startswith('D') and c[1:].isdigit()],
    'Match (M)': [c for c in features_final if c.startswith('M') and c[1:].isdigit()],
    'Vesta (V)': [c for c in features_final if c.startswith('V')],
    'Identity (id)': [c for c in features_final if c.startswith('id_')],
    'Device': [c for c in features_final if 'device' in c.lower()]
}

total_grouped = 0
for name, cols in groups.items():
    if cols:
        print(f"  {name}: {len(cols)}")
        total_grouped += len(cols)

# Save
df_final.to_parquet("data_files/train_final.parquet", index=False)
print(f"\nSaved to: data_files/train_final.parquet")