import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_parquet("data_files/train_final.parquet")

X = df.drop(columns=['isFraud'])
y = df['isFraud']

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
)

print(f"Train: {len(X_train):,} ({y_train.mean()*100:.2f}% fraud)")
print(f"Val:   {len(X_val):,} ({y_val.mean()*100:.2f}% fraud)")
print(f"Test:  {len(X_test):,} ({y_test.mean()*100:.2f}% fraud)")

X_train.to_parquet("data_files/X_train.parquet", index=False)
X_val.to_parquet("data_files/X_val.parquet", index=False)
X_test.to_parquet("data_files/X_test.parquet", index=False)
y_train.to_frame().to_parquet("data_files/y_train.parquet", index=False)
y_val.to_frame().to_parquet("data_files/y_val.parquet", index=False)
y_test.to_frame().to_parquet("data_files/y_test.parquet", index=False)

print("Saved to data_files/")