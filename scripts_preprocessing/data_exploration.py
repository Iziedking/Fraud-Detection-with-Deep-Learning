import pandas as pd

df = pd.read_parquet("data_files/train_final.parquet")

print("=== Dataset Overview ===")
print(f"Rows: {len(df):,}")
print(f"Columns: {df.shape[1]}")
print(f"Memory: {df.memory_usage(deep=True).sum() / 1e6:.1f} MB")

print("\n=== Target Distribution ===")
print(f"Fraud (1): {df['isFraud'].sum():,} ({df['isFraud'].mean()*100:.2f}%)")
print(f"Legit (0): {(df['isFraud']==0).sum():,} ({(df['isFraud']==0).mean()*100:.2f}%)")

print("\n=== Data Types ===")
print(df.dtypes.value_counts())

print("\n=== Column Names ===")
print(df.columns.tolist())

print("\n=== First 5 Rows (key columns) ===")
key_cols = ['isFraud', 'TransactionDT', 'TransactionAmt', 'ProductCD', 'card1', 'card4', 'addr1', 'P_emaildomain']
print(df[key_cols].head())

print("\n=== Basic Stats (numeric) ===")
print(df[['TransactionAmt', 'TransactionDT', 'card1', 'C1', 'D1']].describe().round(2))