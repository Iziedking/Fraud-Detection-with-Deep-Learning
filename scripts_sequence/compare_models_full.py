import numpy as np
import matplotlib.pyplot as plt
import os

results = {
    'Baseline Dense': {
        'f1': 0.6338, 'precision': 0.7674, 'recall': 0.5399, 'auc': 0.9138,
        'features': 251
    },
    'LSTM+Dense Hybrid': {
        'f1': 0.6435, 'precision': 0.7433, 'recall': 0.5673, 'auc': 0.9203,
        'features': 251
    },
    'Sequence LSTM': {
        'f1': 0.2839, 'precision': 0.3381, 'recall': 0.2447, 'auc': 0.7811,
        'features': 251
    },
    'GNN': {
        'f1': 0.1541, 'precision': 0.0867, 'recall': 0.6909, 'auc': 0.7770,
        'features': 251
    },
    'Fusion (Reduced)': {
        'f1': 0.6556, 'precision': 0.6946, 'recall': 0.6208, 'auc': 0.9356,
        'features': 252
    },
    'Fusion (Full)': {
        'f1': 0.6860, 'precision': 0.7512, 'recall': 0.6312, 'auc': 0.9379,
        'features': 435
    }
}

# Threshold analysis for full features model
threshold_results_full = {
    0.3: {'f1': 0.2775, 'precision': 0.1649, 'recall': 0.8738},
    0.5: {'f1': 0.5043, 'precision': 0.3723, 'recall': 0.7812},
    0.7: {'f1': 0.6860, 'precision': 0.7512, 'recall': 0.6312},
    0.8: {'f1': 0.6684, 'precision': 0.9128, 'recall': 0.5273},
    0.9: {'f1': 0.5386, 'precision': 0.9771, 'recall': 0.3717}
}

threshold_results_reduced = {
    0.3: {'f1': 0.2304, 'precision': 0.1322, 'recall': 0.8974},
    0.5: {'f1': 0.4419, 'precision': 0.3048, 'recall': 0.8032},
    0.7: {'f1': 0.6556, 'precision': 0.6946, 'recall': 0.6208},
    0.8: {'f1': 0.6322, 'precision': 0.8746, 'recall': 0.4950},
    0.9: {'f1': 0.4930, 'precision': 0.9652, 'recall': 0.3311}
}

def generate_visualizations():
    os.makedirs('results_sequence/result_full_comparison', exist_ok=True)
    
    models = list(results.keys())
    metrics = ['f1', 'precision', 'recall', 'auc']
    
    # 1. Grouped bar chart - All metrics comparison
    fig, ax = plt.subplots(figsize=(16, 8))
    x = np.arange(len(models))
    width = 0.2
    
    colors = ['#2ecc71', '#3498db', '#e74c3c', '#9b59b6']
    for i, metric in enumerate(metrics):
        values = [results[m][metric] for m in models]
        bars = ax.bar(x + i*width, values, width, label=metric.upper(), alpha=0.85, color=colors[i])
    
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Model Performance Comparison (Including Full Features)', fontsize=14, fontweight='bold')
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels([m.replace(' ', '\n') for m in models], fontsize=10)
    ax.legend(fontsize=10)
    ax.set_ylim(0, 1.1)
    ax.axhline(y=0.85, color='green', linestyle='--', alpha=0.5, label='F1 Target')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('results_sequence/result_full_comparison/model_comparison_all.png', dpi=150)
    plt.close()
    print("Saved: model_comparison_all.png")
    
    # 2. Reduced vs Full Features comparison
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    ax = axes[0]
    x = np.arange(len(metrics))
    width = 0.35
    
    reduced_vals = [results['Fusion (Reduced)'][m] for m in metrics]
    full_vals = [results['Fusion (Full)'][m] for m in metrics]
    
    bars1 = ax.bar(x - width/2, reduced_vals, width, label='Reduced (252 features)', color='#3498db', alpha=0.8)
    bars2 = ax.bar(x + width/2, full_vals, width, label='Full (435 features)', color='#e74c3c', alpha=0.8)
    
    ax.set_ylabel('Score')
    ax.set_title('Reduced vs Full Features Comparison', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([m.upper() for m in metrics])
    ax.legend()
    ax.set_ylim(0, 1.1)
    ax.grid(True, alpha=0.3, axis='y')
    
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.4f}',
                       xy=(bar.get_x() + bar.get_width()/2, height),
                       xytext=(0, 3), textcoords="offset points",
                       ha='center', va='bottom', fontsize=9)
    
    # Improvement analysis
    ax = axes[1]
    improvements = [(full_vals[i] - reduced_vals[i]) / reduced_vals[i] * 100 
                   for i in range(len(metrics))]
    
    colors = ['green' if imp > 0 else 'red' for imp in improvements]
    bars = ax.bar([m.upper() for m in metrics], improvements, color=colors, alpha=0.7)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.set_ylabel('Improvement (%)')
    ax.set_title('Performance Improvement: Full vs Reduced', fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    for bar, imp in zip(bars, improvements):
        height = bar.get_height()
        ax.annotate(f'{imp:+.1f}%',
                   xy=(bar.get_x() + bar.get_width()/2, height),
                   xytext=(0, 3 if height >= 0 else -12), textcoords="offset points",
                   ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('results_sequence/result_full_comparison/reduced_vs_full.png', dpi=150)
    plt.close()
    print("Saved: reduced_vs_full.png")
    
    # 3. AUC comparison chart
    fig, ax = plt.subplots(figsize=(12, 6))
    
    aucs = [results[m]['auc'] for m in models]
    colors = ['#95a5a6'] * 4 + ['#3498db', '#e74c3c']
    bars = ax.barh(models, aucs, color=colors, alpha=0.8)
    
    ax.set_xlabel('AUC-ROC Score')
    ax.set_title('AUC-ROC Comparison Across All Models', fontsize=14, fontweight='bold')
    ax.set_xlim(0.7, 1.0)
    ax.axvline(x=0.9, color='green', linestyle='--', alpha=0.5)
    
    for bar, auc in zip(bars, aucs):
        ax.annotate(f'{auc:.4f}',
                   xy=(auc, bar.get_y() + bar.get_height()/2),
                   xytext=(5, 0), textcoords="offset points",
                   ha='left', va='center', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('results_sequence/result_full_comparison/auc_comparison.png', dpi=150)
    plt.close()
    print("Saved: auc_comparison.png")
    
    # 4. Threshold analysis comparison
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    thresholds = [0.3, 0.5, 0.7, 0.8, 0.9]
    
    for idx, metric in enumerate(['f1', 'precision', 'recall']):
        ax = axes[idx]
        reduced_vals = [threshold_results_reduced[t][metric] for t in thresholds]
        full_vals = [threshold_results_full[t][metric] for t in thresholds]
        
        ax.plot(thresholds, reduced_vals, 'o-', label='Reduced (252)', color='#3498db', linewidth=2, markersize=8)
        ax.plot(thresholds, full_vals, 's-', label='Full (435)', color='#e74c3c', linewidth=2, markersize=8)
        
        ax.set_xlabel('Threshold')
        ax.set_ylabel(metric.upper())
        ax.set_title(f'{metric.upper()} vs Threshold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.05)
    
    plt.tight_layout()
    plt.savefig('results_sequence/result_full_comparison/threshold_comparison.png', dpi=150)
    plt.close()
    print("Saved: threshold_comparison.png")
    
    # 5. Feature count vs performance scatter
    fig, ax = plt.subplots(figsize=(10, 8))
    
    feature_counts = [results[m]['features'] for m in models]
    f1_scores = [results[m]['f1'] for m in models]
    auc_scores = [results[m]['auc'] for m in models]
    
    scatter = ax.scatter(feature_counts, f1_scores, s=[a*500 for a in auc_scores], 
                        c=auc_scores, cmap='RdYlGn', alpha=0.7, edgecolors='black', linewidth=2)
    
    for i, model in enumerate(models):
        ax.annotate(model, 
                   (feature_counts[i], f1_scores[i]),
                   xytext=(10, 5), textcoords='offset points', fontsize=9)
    
    ax.set_xlabel('Number of Features', fontsize=12)
    ax.set_ylabel('F1 Score (threshold 0.7)', fontsize=12)
    ax.set_title('Feature Count vs F1 Score (bubble size & color = AUC)', fontsize=14, fontweight='bold')
    plt.colorbar(scatter, label='AUC')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results_sequence/result_full_comparison/features_vs_performance.png', dpi=150)
    plt.close()
    print("Saved: features_vs_performance.png")
    
    # 6. Best model summary
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.axis('off')
    
    summary_text = """
    ╔══════════════════════════════════════════════════════════════════════════════╗
    ║                    FINAL MODEL COMPARISON SUMMARY                            ║
    ╠══════════════════════════════════════════════════════════════════════════════╣
    ║                                                                              ║
    ║  BEST MODEL: Fusion (Full Features) - 435 features                           ║
    ║                                                                              ║
    ║  ┌─────────────────────────────────────────────────────────────────────────┐ ║
    ║  │  Threshold 0.7 (Balanced):                                              │ ║
    ║  │    • F1 Score:   0.6860  (Best among all models)                        │ ║
    ║  │    • Precision:  0.7512                                                 │ ║
    ║  │    • Recall:     0.6312                                                 │ ║
    ║  │    • AUC-ROC:    0.9379  (Best among all models)                        │ ║
    ║  └─────────────────────────────────────────────────────────────────────────┘ ║
    ║                                                                              ║
    ║  ┌─────────────────────────────────────────────────────────────────────────┐ ║
    ║  │  Threshold 0.8 (High Precision):                                        │ ║
    ║  │    • F1 Score:   0.6684                                                 │ ║
    ║  │    • Precision:  0.9128  (91.3% of flagged transactions are fraud)      │ ║
    ║  │    • Recall:     0.5273  (52.7% of fraud detected)                      │ ║
    ║  └─────────────────────────────────────────────────────────────────────────┘ ║
    ║                                                                              ║
    ║  ┌─────────────────────────────────────────────────────────────────────────┐ ║
    ║  │  Threshold 0.9 (Very High Precision):                                   │ ║
    ║  │    • F1 Score:   0.5386                                                 │ ║
    ║  │    • Precision:  0.9771  (97.7% accuracy on flagged)                    │ ║
    ║  │    • Recall:     0.3717  (37.2% of fraud detected)                      │ ║
    ║  └─────────────────────────────────────────────────────────────────────────┘ ║
    ║                                                                              ║
    ║  KEY FINDING: Feature reduction hurt performance!                            ║
    ║    • F1 improved by 4.6% with full features                                  ║
    ║    • Precision improved by 8.1%                                              ║
    ║    • Recall improved by 1.7%                                                 ║
    ║    • AUC improved by 0.2%                                                    ║
    ║                                                                              ║
    ╚══════════════════════════════════════════════════════════════════════════════╝
    """
    
    ax.text(0.5, 0.5, summary_text, transform=ax.transAxes, fontsize=10,
           verticalalignment='center', horizontalalignment='center',
           fontfamily='monospace', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.savefig('results_sequence/result_full_comparison/final_summary.png', dpi=150, 
               bbox_inches='tight', facecolor='white')
    plt.close()
    print("Saved: final_summary.png")
    
    # Print console summary
    print("\n" + "=" * 80)
    print("FINAL MODEL COMPARISON RESULTS")
    print("=" * 80)
    print(f"{'Model':<25} {'F1':>8} {'Precision':>10} {'Recall':>8} {'AUC':>8} {'Features':>10}")
    print("-" * 80)
    for model in models:
        r = results[model]
        print(f"{model:<25} {r['f1']:>8.4f} {r['precision']:>10.4f} {r['recall']:>8.4f} {r['auc']:>8.4f} {r['features']:>10}")
    print("=" * 80)
    
    print("\n" + "=" * 80)
    print("REDUCED vs FULL FEATURES IMPACT")
    print("=" * 80)
    reduced = results['Fusion (Reduced)']
    full = results['Fusion (Full)']
    
    for metric in metrics:
        diff = full[metric] - reduced[metric]
        pct = (diff / reduced[metric] * 100)
        symbol = "↑" if diff > 0 else "↓" if diff < 0 else "="
        print(f"{metric.upper():>10}: {reduced[metric]:.4f} → {full[metric]:.4f} ({symbol} {abs(pct):.1f}%)")
    
    print("\n" + "=" * 80)
    print("THRESHOLD ANALYSIS - FULL FEATURES MODEL")
    print("=" * 80)
    print(f"{'Threshold':<12} {'F1':>8} {'Precision':>10} {'Recall':>8}")
    print("-" * 40)
    for thresh, vals in threshold_results_full.items():
        print(f"{thresh:<12} {vals['f1']:>8.4f} {vals['precision']:>10.4f} {vals['recall']:>8.4f}")
    
    print("\n" + "=" * 80)
    print("KEY CONCLUSIONS")
    print("=" * 80)
    print("1. FULL FEATURES MODEL IS THE BEST PERFORMER")
    print(f"   - Highest AUC: 0.9379")
    print(f"   - Highest F1:  0.6860 (at threshold 0.7)")
    print()
    print("2. FEATURE REDUCTION NEGATIVELY IMPACTED PERFORMANCE")
    print(f"   - Removing 48% of V-features cost us ~4.6% F1 improvement")
    print(f"   - Deep learning benefits from more features")
    print()
    print("3. PRECISION-RECALL TRADEOFF")
    print(f"   - Threshold 0.7: Balanced (F1=0.6860, Prec=0.7512)")
    print(f"   - Threshold 0.8: High precision (Prec=0.9128, Rec=0.5273)")
    print(f"   - Threshold 0.9: Very high precision (Prec=0.9771, Rec=0.3717)")
    print()
    print("4. PRODUCTION RECOMMENDATION")
    print(f"   - Use Full Features Fusion model with threshold 0.7-0.8")
    print(f"   - For fraud-critical systems, use threshold 0.8+ for high precision")
    print("=" * 80)

if __name__ == '__main__':
    generate_visualizations()