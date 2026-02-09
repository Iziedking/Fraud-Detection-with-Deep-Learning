import matplotlib.pyplot as plt
import numpy as np
import os

results = {
    'Baseline Dense': {
        'f1': 0.6338, 'precision': 0.7674, 'recall': 0.5399, 'auc': 0.9138,
        'description': 'Simple MLP on all 250 features'
    },
    'LSTM+Dense Hybrid': {
        'f1': 0.6435, 'precision': 0.7433, 'recall': 0.5673, 'auc': 0.9203,
        'description': 'LSTM (temporal) + Dense (behavioral) branches'
    },
    'Sequence LSTM': {
        'f1': 0.2839, 'precision': 0.3381, 'recall': 0.2447, 'auc': 0.7811,
        'description': 'LSTM on card-grouped transaction sequences'
    },
    'GNN': {
        'f1': 0.1541, 'precision': 0.0867, 'recall': 0.6909, 'auc': 0.7770,
        'description': 'Graph attention on relational features'
    },
    'Fusion (LSTM+GNN+Dense)': {
        'f1': 0.6556, 'precision': 0.6946, 'recall': 0.6208, 'auc': 0.9356,
        'description': 'Full hybrid: temporal + relational + behavioral'
    }
}

os.makedirs('results_sequence', exist_ok=True)

models = list(results.keys())
metrics = ['f1', 'precision', 'recall', 'auc']
colors = ['#3498db', '#2ecc71', '#e74c3c', '#9b59b6']

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

for idx, metric in enumerate(metrics):
    values = [results[m][metric] for m in models]
    bars = axes[idx].barh(models, values, color=colors[idx], alpha=0.8)
    axes[idx].set_xlabel('Score')
    axes[idx].set_title(f'{metric.upper()} Comparison')
    axes[idx].set_xlim(0, 1.1)
    axes[idx].axvline(x=0.85 if metric == 'f1' else (0.95 if metric == 'precision' else 0.83), 
                      color='red', linestyle='--', alpha=0.5, label='Target')
    
    for bar, val in zip(bars, values):
        axes[idx].text(val + 0.02, bar.get_y() + bar.get_height()/2, 
                      f'{val:.3f}', va='center', fontsize=9)

plt.tight_layout()
plt.savefig('results_sequence/model_comparison_metrics.png', dpi=150)
plt.close()

fig, ax = plt.subplots(figsize=(12, 6))
x = np.arange(len(models))
width = 0.2

for i, metric in enumerate(metrics):
    values = [results[m][metric] for m in models]
    bars = ax.bar(x + i*width, values, width, label=metric.upper(), alpha=0.8)

ax.set_ylabel('Score')
ax.set_title('Model Performance Comparison (All Metrics)')
ax.set_xticks(x + width * 1.5)
ax.set_xticklabels([m.replace(' ', '\n') for m in models], fontsize=9)
ax.legend()
ax.set_ylim(0, 1.1)
ax.axhline(y=0.85, color='gray', linestyle='--', alpha=0.3)
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('results_sequence/model_comparison_grouped.png', dpi=150)
plt.close()

fig, ax = plt.subplots(figsize=(10, 8))

for model in models:
    ax.scatter(results[model]['precision'], results[model]['recall'], 
              s=results[model]['auc']*500, alpha=0.6, label=model)
    ax.annotate(model.split()[0], 
               (results[model]['precision'], results[model]['recall']),
               textcoords="offset points", xytext=(5,5), fontsize=8)

ax.set_xlabel('Precision')
ax.set_ylabel('Recall')
ax.set_title('Precision vs Recall (bubble size = AUC)')
ax.legend(loc='lower left', fontsize=8)
ax.set_xlim(0, 1.05)
ax.set_ylim(0, 1.05)
ax.grid(True, alpha=0.3)

ax.axvline(x=0.95, color='red', linestyle='--', alpha=0.3, label='Precision Target')
ax.axhline(y=0.83, color='blue', linestyle='--', alpha=0.3, label='Recall Target')

plt.tight_layout()
plt.savefig('results_sequence/precision_recall_scatter.png', dpi=150)
plt.close()

print("="*70)
print("FINAL MODEL COMPARISON RESULTS")
print("="*70)
print(f"\n{'Model':<25} {'F1':>8} {'Precision':>10} {'Recall':>8} {'AUC':>8}")
print("-"*70)
for model, metrics in results.items():
    print(f"{model:<25} {metrics['f1']:>8.4f} {metrics['precision']:>10.4f} {metrics['recall']:>8.4f} {metrics['auc']:>8.4f}")

print("\n" + "="*70)
print("KEY FINDINGS")
print("="*70)

best_auc = max(results.items(), key=lambda x: x[1]['auc'])
best_f1 = max(results.items(), key=lambda x: x[1]['f1'])
best_precision = max(results.items(), key=lambda x: x[1]['precision'])
best_recall = max(results.items(), key=lambda x: x[1]['recall'])

print(f"\nBest AUC:       {best_auc[0]} ({best_auc[1]['auc']:.4f})")
print(f"Best F1:        {best_f1[0]} ({best_f1[1]['f1']:.4f})")
print(f"Best Precision: {best_precision[0]} ({best_precision[1]['precision']:.4f})")
print(f"Best Recall:    {best_recall[0]} ({best_recall[1]['recall']:.4f})")

print("\n" + "="*70)
print("TARGET ACHIEVEMENT ANALYSIS")
print("="*70)
print("\nTargets: F1 > 0.85, Precision > 0.95, Recall > 0.83")
print("\nNo model achieved all targets. Best performance:")
print(f"  - Fusion model: F1=0.6556, Precision=0.6946, Recall=0.6208, AUC=0.9356")
print(f"  - At threshold 0.8: Precision=0.8746 (close to 0.95 target)")
print(f"  - At threshold 0.9: Precision=0.9652 (exceeds target but low recall)")

print("\n" + "="*70)
print("BRANCH CONTRIBUTION (SHAP Analysis)")
print("="*70)
print("\n  Behavioral (Dense):  61.0% - Vesta V-features most predictive")
print("  Temporal (LSTM):     23.8% - Time-based patterns")
print("  Relational (GNN):    15.2% - Card/device relationships")

print("\n" + "="*70)
print("CONCLUSIONS")
print("="*70)
print("""
1. The Fusion model (LSTM+GNN+Dense) achieved the best overall performance
   with AUC=0.9356, demonstrating effective multi-branch integration.

2. Behavioral features (Vesta V-features) contribute 61% of predictive power,
   indicating the pre-engineered features already capture fraud patterns well.

3. Sequence LSTM and standalone GNN underperformed because:
   - The dataset lacks explicit graph structure (edges between transactions)
   - Vesta features already encode temporal/relational patterns
   - Card1 may not represent unique users consistently

4. The precision-recall tradeoff is fundamental:
   - High precision (0.87+) achievable at threshold 0.8 but sacrifices recall
   - Threshold selection depends on business priorities

5. For production deployment, the Fusion model with threshold 0.7-0.8 
   provides the best balance of fraud detection and customer experience.
""")

print("\nSaved comparison visualizations to results_sequence/")