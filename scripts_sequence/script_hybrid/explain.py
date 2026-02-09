import torch
import numpy as np
import matplotlib.pyplot as plt
import shap
import os

from config import CONFIG
from data_loader import get_dataloaders
from model import FusionModel

print("Loading model and data...")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

_, _, test_loader = get_dataloaders(CONFIG)
model = FusionModel(CONFIG).to(device)
model.load_state_dict(torch.load(f"{CONFIG['paths']['models']}/fusion_model.pt"))
model.eval()

X_temp_test = np.load(CONFIG['data']['test_temp'])
X_rel_test = np.load(CONFIG['data']['test_rel'])
X_beh_test = np.load(CONFIG['data']['test_beh'])
y_test = np.load(CONFIG['data']['test_y'])

X_combined = np.hstack([X_temp_test, X_rel_test, X_beh_test])

temporal_names = [f'temp_{i}' for i in range(CONFIG['model']['temporal_dim'])]
relational_names = [f'rel_{i}' for i in range(CONFIG['model']['relational_dim'])]
behavioral_names = [f'beh_{i}' for i in range(CONFIG['model']['behavioral_dim'])]
feature_names = temporal_names + relational_names + behavioral_names

n_temp = CONFIG['model']['temporal_dim']
n_rel = CONFIG['model']['relational_dim']

def model_predict(X):
    if X.ndim == 1:
        X = X.reshape(1, -1)
    
    X = torch.tensor(X, dtype=torch.float32).to(device)
    x_temp = X[:, :n_temp]
    x_rel = X[:, n_temp:n_temp+n_rel]
    x_beh = X[:, n_temp+n_rel:]
    
    with torch.no_grad():
        preds = model(x_temp, x_rel, x_beh)
    
    result = preds.cpu().numpy()
    if result.ndim == 0:
        result = result.reshape(1)
    return result

print("Selecting samples for SHAP analysis...")
n_background = 100
n_explain = 100

fraud_idx = np.where(y_test == 1)[0]
legit_idx = np.where(y_test == 0)[0]

np.random.seed(42)
bg_fraud = np.random.choice(fraud_idx, size=min(20, len(fraud_idx)), replace=False)
bg_legit = np.random.choice(legit_idx, size=80, replace=False)
background_idx = np.concatenate([bg_fraud, bg_legit])
background = X_combined[background_idx]

exp_fraud = np.random.choice(fraud_idx, size=min(50, len(fraud_idx)), replace=False)
exp_legit = np.random.choice(legit_idx, size=50, replace=False)
explain_idx = np.concatenate([exp_fraud, exp_legit])
X_explain = X_combined[explain_idx]
y_explain = y_test[explain_idx]

print(f"Background samples: {len(background)}")
print(f"Samples to explain: {len(X_explain)}")
print(f"  Fraud: {(y_explain == 1).sum()}")
print(f"  Legit: {(y_explain == 0).sum()}")

print("\nUsing kmeans to summarize background...")
background_summary = shap.kmeans(background, 50)

print("Computing SHAP values (this may take several minutes)...")
explainer = shap.KernelExplainer(model_predict, background_summary)
shap_values = explainer.shap_values(X_explain, nsamples=50)

os.makedirs(CONFIG['paths']['results'], exist_ok=True)

print("Generating SHAP visualizations...")

mean_shap = np.abs(shap_values).mean(axis=0)
top_idx = np.argsort(mean_shap)[-20:][::-1]

plt.figure(figsize=(10, 8))
plt.barh(range(20), mean_shap[top_idx][::-1])
plt.yticks(range(20), [feature_names[i] for i in top_idx][::-1])
plt.xlabel('Mean |SHAP Value|')
plt.title('Feature Importance - Top 20')
plt.tight_layout()
plt.savefig(f"{CONFIG['paths']['results']}/shap_importance.png", dpi=150)
plt.close()

temp_importance = mean_shap[:n_temp].sum()
rel_importance = mean_shap[n_temp:n_temp+n_rel].sum()
beh_importance = mean_shap[n_temp+n_rel:].sum()
total = temp_importance + rel_importance + beh_importance

plt.figure(figsize=(8, 6))
branches = ['Temporal\n(LSTM)', 'Relational\n(GNN)', 'Behavioral\n(Dense)']
importances = [temp_importance/total*100, rel_importance/total*100, beh_importance/total*100]
colors = ['#3498db', '#e74c3c', '#2ecc71']
plt.bar(branches, importances, color=colors)
plt.ylabel('Contribution (%)')
plt.title('Branch Contribution to Predictions')
for i, v in enumerate(importances):
    plt.text(i, v + 1, f'{v:.1f}%', ha='center', fontweight='bold')
plt.ylim(0, max(importances) + 10)
plt.tight_layout()
plt.savefig(f"{CONFIG['paths']['results']}/branch_importance.png", dpi=150)
plt.close()

fraud_mask = y_explain == 1
legit_mask = y_explain == 0

fraud_shap_mean = np.abs(shap_values[fraud_mask]).mean(axis=0)
legit_shap_mean = np.abs(shap_values[legit_mask]).mean(axis=0)

top_fraud_idx = np.argsort(fraud_shap_mean)[-10:][::-1]
top_legit_idx = np.argsort(legit_shap_mean)[-10:][::-1]

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

axes[0].barh(range(10), fraud_shap_mean[top_fraud_idx][::-1], color='red', alpha=0.7)
axes[0].set_yticks(range(10))
axes[0].set_yticklabels([feature_names[i] for i in top_fraud_idx][::-1])
axes[0].set_xlabel('Mean |SHAP Value|')
axes[0].set_title('Top Features for Fraud Detection')

axes[1].barh(range(10), legit_shap_mean[top_legit_idx][::-1], color='green', alpha=0.7)
axes[1].set_yticks(range(10))
axes[1].set_yticklabels([feature_names[i] for i in top_legit_idx][::-1])
axes[1].set_xlabel('Mean |SHAP Value|')
axes[1].set_title('Top Features for Legitimate Transactions')

plt.tight_layout()
plt.savefig(f"{CONFIG['paths']['results']}/shap_fraud_vs_legit.png", dpi=150)
plt.close()

try:
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, X_explain, feature_names=feature_names, show=False, max_display=20)
    plt.tight_layout()
    plt.savefig(f"{CONFIG['paths']['results']}/shap_summary.png", dpi=150, bbox_inches='tight')
    plt.close()
except Exception as e:
    print(f"Could not generate summary plot: {e}")

print("\n" + "="*60)
print("SHAP ANALYSIS RESULTS")
print("="*60)

print(f"\nBranch Contributions:")
print(f"  Temporal (LSTM):    {temp_importance/total*100:.1f}%")
print(f"  Relational (GNN):   {rel_importance/total*100:.1f}%")
print(f"  Behavioral (Dense): {beh_importance/total*100:.1f}%")

print(f"\nTop 10 Most Important Features:")
for i, idx in enumerate(top_idx[:10]):
    branch = 'Temporal' if idx < n_temp else ('Relational' if idx < n_temp + n_rel else 'Behavioral')
    print(f"  {i+1}. {feature_names[idx]} ({branch}): {mean_shap[idx]:.4f}")

print(f"\nSaved visualizations to {CONFIG['paths']['results']}/")