import pandas as pd
import numpy as np

df = pd.read_parquet("data_files/train_final.parquet")

print("=== Checking User/Card Grouping Potential ===\n")

grouping_candidates = ['card1', 'card2', 'addr1', 'P_emaildomain']

for col in grouping_candidates:
    if col in df.columns:
        unique_count = df[col].nunique()
        total_rows = len(df)
        avg_per_group = total_rows / unique_count
        
        group_sizes = df.groupby(col).size()
        
        print(f"{col}:")
        print(f"  Unique values: {unique_count:,}")
        print(f"  Avg transactions per group: {avg_per_group:.1f}")
        print(f"  Groups with 2+ transactions: {(group_sizes >= 2).sum():,}")
        print(f"  Groups with 5+ transactions: {(group_sizes >= 5).sum():,}")
        print(f"  Groups with 10+ transactions: {(group_sizes >= 10).sum():,}")
        print(f"  Max group size: {group_sizes.max()}")
        print()

print("=== Transaction Time Analysis ===\n")
df_sorted = df.sort_values('TransactionDT')
print(f"Time range: {df['TransactionDT'].min()} to {df['TransactionDT'].max()}")
print(f"Time span (seconds): {df['TransactionDT'].max() - df['TransactionDT'].min():,}")
print(f"Time span (days): {(df['TransactionDT'].max() - df['TransactionDT'].min()) / 86400:.1f}")

print("\n=== Sample Card1 Sequence ===\n")
top_card = df['card1'].value_counts().index[0]
card_txns = df[df['card1'] == top_card].sort_values('TransactionDT')
print(f"Card1 = {top_card} has {len(card_txns)} transactions")
print(card_txns[['TransactionDT', 'TransactionAmt', 'isFraud']].head(10))

print("\n=== Fraud Rate by Sequence Length ===\n")
card_stats = df.groupby('card1').agg({
    'isFraud': ['sum', 'count', 'mean']
}).reset_index()
card_stats.columns = ['card1', 'fraud_count', 'txn_count', 'fraud_rate']

for min_txns in [1, 2, 5, 10, 20]:
    subset = card_stats[card_stats['txn_count'] >= min_txns]
    total_txns = subset['txn_count'].sum()
    total_fraud = subset['fraud_count'].sum()
    print(f"Cards with {min_txns}+ txns: {len(subset):,} cards, {total_txns:,} txns, {total_fraud/total_txns*100:.2f}% fraud")