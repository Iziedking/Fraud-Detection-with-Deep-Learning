import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler

X_train = pd.read_parquet("data_files/X_train.parquet")
X_val = pd.read_parquet("data_files/X_val.parquet")
X_test = pd.read_parquet("data_files/X_test.parquet")

feature_names = X_train.columns.tolist()

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

X_train_scaled = pd.DataFrame(X_train_scaled, columns=feature_names)
X_val_scaled = pd.DataFrame(X_val_scaled, columns=feature_names)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=feature_names)

X_train_scaled.to_parquet("data_files/X_train_scaled.parquet", index=False)
X_val_scaled.to_parquet("data_files/X_val_scaled.parquet", index=False)
X_test_scaled.to_parquet("data_files/X_test_scaled.parquet", index=False)
joblib.dump(scaler, "data_files/scaler.joblib")

print(f"Scaled: {X_train_scaled.shape[1]} features")
print(f"Train mean sample: {X_train_scaled.iloc[0, :5].round(3).tolist()}")
print(f"Scaler saved: data_files/scaler.joblib")