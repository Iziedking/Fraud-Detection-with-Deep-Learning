import torch
import numpy as np
import shap
import matplotlib.pyplot as plt
import json
import os
from train_full import FusionModel
from config_full import CONFIG

def run_shap_analysis():
    print("=" * 70)
    print("SHAP ANALYSIS - FULL FEATURE FUSION MODEL")
    print("=" * 70)
    
    # Use GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Load model
    model = FusionModel(CONFIG).to(device)
    model.load_state_dict(torch.load(f"{CONFIG['paths']['models']}/fusion_model_full.pt", map_location=device))
    model.eval()
    print("Model loaded successfully")
    
    # Load test data
    X_temp = np.load(CONFIG['data']['test_temp'])
    X_rel = np.load(CONFIG['data']['test_rel'])
    X_beh = np.load(CONFIG['data']['test_beh'])
    y = np.load(CONFIG['data']['test_y'])
    
    print(f"Test data shapes: Temp={X_temp.shape}, Rel={X_rel.shape}, Beh={X_beh.shape}")
    
    # Load feature names
    with open('data_files/full_feature_dims.json', 'r') as f:
        dims = json.load(f)
    
    temp_cols = dims['temporal_cols']
    rel_cols = dims['relational_cols']
    beh_cols = dims['behavioral_cols']
    
    # Create combined feature names
    all_feature_names = (
        [f"temp_{c}" for c in temp_cols] +
        [f"rel_{c}" for c in rel_cols] +
        [f"beh_{c}" for c in beh_cols]
    )
    
    print(f"Total features: {len(all_feature_names)}")
    
    # Select samples for SHAP
    fraud_idx = np.where(y == 1)[0]
    legit_idx = np.where(y == 0)[0]
    
    np.random.seed(42)
    n_fraud = min(50, len(fraud_idx))
    n_legit = min(50, len(legit_idx))
    
    selected_fraud = np.random.choice(fraud_idx, n_fraud, replace=False)
    selected_legit = np.random.choice(legit_idx, n_legit, replace=False)
    
    # Background samples (for SHAP)
    bg_fraud = np.random.choice(fraud_idx, 20, replace=False)
    bg_legit = np.random.choice(legit_idx, 80, replace=False)
    bg_idx = np.concatenate([bg_fraud, bg_legit])
    
    # Combine features for SHAP
    X_combined_bg = np.hstack([X_temp[bg_idx], X_rel[bg_idx], X_beh[bg_idx]])
    
    explain_idx = np.concatenate([selected_fraud, selected_legit])
    X_combined_explain = np.hstack([X_temp[explain_idx], X_rel[explain_idx], X_beh[explain_idx]])
    y_explain = y[explain_idx]
    
    print(f"Background samples: {len(bg_idx)}")
    print(f"Samples to explain: {len(explain_idx)} ({n_fraud} fraud, {n_legit} legit)")
    
    # Feature dimensions
    n_temp = len(temp_cols)
    n_rel = len(rel_cols)
    
    # Create prediction function for SHAP - FIXED VERSION
    def predict_fn(X):
        """Prediction function that returns proper array shape for SHAP"""
        X = np.atleast_2d(X)  # Ensure 2D
        batch_size = X.shape[0]
        
        x_temp = torch.tensor(X[:, :n_temp], dtype=torch.float32).to(device)
        x_rel = torch.tensor(X[:, n_temp:n_temp+n_rel], dtype=torch.float32).to(device)
        x_beh = torch.tensor(X[:, n_temp+n_rel:], dtype=torch.float32).to(device)
        
        with torch.no_grad():
            preds = model(x_temp, x_rel, x_beh)
        
        # Ensure output is 1D array with shape (batch_size,)
        result = preds.cpu().numpy()
        if result.ndim == 0:
            result = np.array([result.item()])
        result = result.flatten()
        return result
    
    # Test prediction function
    print("\nTesting prediction function...")
    test_pred = predict_fn(X_combined_bg[:5])
    print(f"Test prediction shape: {test_pred.shape}, values: {test_pred[:3]}")
    
    # Use k-means to summarize background data
    print("\nSummarizing background data with k-means...")
    background_summary = shap.kmeans(X_combined_bg, 50)
    
    # Create SHAP explainer
    print("Creating SHAP KernelExplainer...")
    explainer = shap.KernelExplainer(predict_fn, background_summary)
    
    # Calculate SHAP values with progress
    print("Calculating SHAP values (this may take 10-20 minutes with GPU)...")
    shap_values = explainer.shap_values(X_combined_explain, nsamples=100, silent=False)
    
    # Create output directory
    os.makedirs('results_sequence/result_hybrid_full', exist_ok=True)
    
    # Convert to numpy array if needed
    shap_values = np.array(shap_values)
    print(f"SHAP values shape: {shap_values.shape}")
    
    # 1. Summary plot
    print("\nGenerating SHAP summary plot...")
    plt.figure(figsize=(12, 10))
    shap.summary_plot(shap_values, X_combined_explain, feature_names=all_feature_names, 
                     show=False, max_display=30)
    plt.title('SHAP Feature Importance (Full Features - Top 30)', fontsize=14)
    plt.tight_layout()
    plt.savefig('results_sequence/result_hybrid_full/shap_summary_full.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: shap_summary_full.png")
    
    # 2. Feature importance bar chart
    print("Generating feature importance bar chart...")
    mean_shap = np.abs(shap_values).mean(axis=0)
    importance_df = list(zip(all_feature_names, mean_shap))
    importance_df.sort(key=lambda x: x[1], reverse=True)
    
    top_n = 30
    top_features = importance_df[:top_n]
    
    plt.figure(figsize=(12, 10))
    features, importances = zip(*top_features)
    colors = []
    for f in features:
        if f.startswith('temp_'):
            colors.append('#3498db')
        elif f.startswith('rel_'):
            colors.append('#e74c3c')
        else:
            colors.append('#2ecc71')
    
    plt.barh(range(len(features)), importances, color=colors, alpha=0.8)
    plt.yticks(range(len(features)), features)
    plt.xlabel('Mean |SHAP Value|')
    plt.title(f'Top {top_n} Most Important Features (Full Model - 435 Features)')
    plt.gca().invert_yaxis()
    
    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#3498db', label='Temporal (30)'),
        Patch(facecolor='#e74c3c', label='Relational (52)'),
        Patch(facecolor='#2ecc71', label='Behavioral (353)')
    ]
    plt.legend(handles=legend_elements, loc='lower right')
    
    plt.tight_layout()
    plt.savefig('results_sequence/result_hybrid_full/shap_importance_full.png', dpi=150)
    plt.close()
    print("Saved: shap_importance_full.png")
    
    # 3. Branch contribution analysis
    print("Calculating branch contributions...")
    temp_importance = sum(imp for feat, imp in importance_df if feat.startswith('temp_'))
    rel_importance = sum(imp for feat, imp in importance_df if feat.startswith('rel_'))
    beh_importance = sum(imp for feat, imp in importance_df if feat.startswith('beh_'))
    
    total = temp_importance + rel_importance + beh_importance
    
    branch_contrib = {
        'Temporal (LSTM)': temp_importance / total * 100,
        'Relational (GNN)': rel_importance / total * 100,
        'Behavioral (Dense)': beh_importance / total * 100
    }
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Pie chart
    ax = axes[0]
    colors_pie = ['#3498db', '#e74c3c', '#2ecc71']
    wedges, texts, autotexts = ax.pie(
        branch_contrib.values(), 
        labels=branch_contrib.keys(), 
        autopct='%1.1f%%',
        colors=colors_pie, 
        startangle=90,
        explode=(0.02, 0.02, 0.02)
    )
    ax.set_title('Branch Contribution (Full Features - 435)', fontweight='bold')
    
    # Bar comparison with reduced model
    ax = axes[1]
    branches = ['Temporal', 'Relational', 'Behavioral']
    reduced_contrib = [23.8, 15.2, 61.0]  # From previous analysis
    full_contrib = [branch_contrib['Temporal (LSTM)'], 
                   branch_contrib['Relational (GNN)'],
                   branch_contrib['Behavioral (Dense)']]
    
    x = np.arange(len(branches))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, reduced_contrib, width, label='Reduced (252 feat)', color='#3498db', alpha=0.7)
    bars2 = ax.bar(x + width/2, full_contrib, width, label='Full (435 feat)', color='#e74c3c', alpha=0.7)
    
    ax.set_ylabel('Contribution (%)')
    ax.set_title('Branch Contribution: Reduced vs Full', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(branches)
    ax.legend()
    ax.set_ylim(0, 80)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.1f}%',
                       xy=(bar.get_x() + bar.get_width()/2, height),
                       xytext=(0, 3), textcoords="offset points",
                       ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('results_sequence/result_hybrid_full/branch_comparison_full.png', dpi=150)
    plt.close()
    print("Saved: branch_comparison_full.png")
    
    # 4. Fraud vs Legit comparison
    print("Generating fraud vs legitimate comparison...")
    fraud_shap = np.abs(shap_values[:n_fraud]).mean(axis=0)
    legit_shap = np.abs(shap_values[n_fraud:]).mean(axis=0)
    
    # Get top features that differ most
    diff = np.abs(fraud_shap - legit_shap)
    top_diff_idx = np.argsort(diff)[-20:][::-1]
    
    plt.figure(figsize=(14, 8))
    x = np.arange(len(top_diff_idx))
    width = 0.35
    
    plt.bar(x - width/2, fraud_shap[top_diff_idx], width, label='Fraud', color='#e74c3c', alpha=0.8)
    plt.bar(x + width/2, legit_shap[top_diff_idx], width, label='Legitimate', color='#2ecc71', alpha=0.8)
    
    plt.xlabel('Feature')
    plt.ylabel('Mean |SHAP Value|')
    plt.title('Top 20 Features with Largest Fraud vs Legitimate Difference')
    plt.xticks(x, [all_feature_names[i] for i in top_diff_idx], rotation=45, ha='right')
    plt.legend()
    plt.tight_layout()
    plt.savefig('results_sequence/result_hybrid_full/shap_fraud_vs_legit_full.png', dpi=150)
    plt.close()
    print("Saved: shap_fraud_vs_legit_full.png")
    
    # 5. V-features analysis (since we added them back)
    print("Analyzing V-features importance...")
    v_features = [(f, imp) for f, imp in importance_df if 'V' in f]
    v_features.sort(key=lambda x: x[1], reverse=True)
    
    plt.figure(figsize=(14, 8))
    top_v = v_features[:25]
    if top_v:
        v_names, v_imps = zip(*top_v)
        plt.barh(range(len(v_names)), v_imps, color='#9b59b6', alpha=0.8)
        plt.yticks(range(len(v_names)), v_names)
        plt.xlabel('Mean |SHAP Value|')
        plt.title('Top 25 V-Features by Importance (Previously Removed Features Highlighted)')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig('results_sequence/result_hybrid_full/v_features_importance.png', dpi=150)
        plt.close()
        print("Saved: v_features_importance.png")
    
    # Print summary
    print("\n" + "=" * 70)
    print("SHAP ANALYSIS COMPLETE - FULL FEATURE MODEL (435 features)")
    print("=" * 70)
    
    print("\nBranch Contributions:")
    print("-" * 40)
    for branch, contrib in branch_contrib.items():
        print(f"  {branch}: {contrib:.1f}%")
    
    print("\nComparison with Reduced Features Model:")
    print("-" * 40)
    print(f"  {'Branch':<20} {'Reduced':>10} {'Full':>10} {'Change':>10}")
    print(f"  {'Temporal':<20} {'23.8%':>10} {branch_contrib['Temporal (LSTM)']:>9.1f}% {branch_contrib['Temporal (LSTM)']-23.8:>+9.1f}%")
    print(f"  {'Relational':<20} {'15.2%':>10} {branch_contrib['Relational (GNN)']:>9.1f}% {branch_contrib['Relational (GNN)']-15.2:>+9.1f}%")
    print(f"  {'Behavioral':<20} {'61.0%':>10} {branch_contrib['Behavioral (Dense)']:>9.1f}% {branch_contrib['Behavioral (Dense)']-61.0:>+9.1f}%")
    
    print(f"\nTop 15 Most Important Features:")
    print("-" * 50)
    for i, (feat, imp) in enumerate(importance_df[:15], 1):
        branch = "Temporal" if feat.startswith("temp_") else "Relational" if feat.startswith("rel_") else "Behavioral"
        print(f"  {i:2}. {feat:<25} ({branch:<10}): {imp:.4f}")
    
    if v_features:
        print