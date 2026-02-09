# Save as scripts_sequence/generate_documentation.py

from datetime import datetime

def generate_documentation():
    doc = '''# Deep Learning for E-Commerce Fraud Detection
## Comprehensive Implementation Report

**Generated:** {date}

---

## Executive Summary

This project developed and evaluated multiple deep learning architectures for detecting fraudulent e-commerce transactions using the IEEE-CIS Fraud Detection dataset.

### Key Finding: Feature Reduction Impact
**Critical discovery:** Correlation-based feature reduction removed 48% of V-features (164 out of 339), which negatively impacted model performance. Restoring all features improved results across all metrics.

### Final Best Model: Fusion (Full Features)
| Metric | Value |
|--------|-------|
| **AUC-ROC** | **0.9379** |
| **F1 (threshold 0.7)** | **0.6860** |
| **Precision (threshold 0.7)** | **0.7512** |
| **Recall (threshold 0.7)** | **0.6312** |

### Model Comparison Summary

| Model | Features | F1 | Precision | Recall | AUC |
|-------|----------|-----|-----------|--------|-----|
| Baseline Dense | 251 | 0.6338 | 0.7674 | 0.5399 | 0.9138 |
| LSTM+Dense Hybrid | 251 | 0.6435 | 0.7433 | 0.5673 | 0.9203 |
| Sequence LSTM | 251 | 0.2839 | 0.3381 | 0.2447 | 0.7811 |
| GNN | 251 | 0.1541 | 0.0867 | 0.6909 | 0.7770 |
| Fusion (Reduced) | 252 | 0.6556 | 0.6946 | 0.6208 | 0.9356 |
| **Fusion (Full)** | **435** | **0.6860** | **0.7512** | **0.6312** | **0.9379** |

---

## 1. Project Overview

### 1.1 Objective
Develop and compare deep learning models for e-commerce fraud detection, analyzing the impact of feature engineering decisions on model performance.

### 1.2 Dataset
- **Source:** IEEE-CIS Fraud Detection (Kaggle/Vesta Corporation)
- **Transactions:** 590,540
- **Fraud Rate:** 3.50% (20,663 fraudulent)
- **Original Features:** 436
- **V-features:** 339 (Vesta proprietary features)

### 1.3 Research Questions
1. Which deep learning architecture performs best for fraud detection?
2. Does correlation-based feature reduction hurt model performance?
3. Which feature groups contribute most to predictions?

---

## 2. Feature Engineering Analysis

### 2.1 Critical Finding: Feature Reduction Impact

Our preprocessing pipeline removed features based on correlation analysis:

| Feature Group | Original | After Reduction | Removed |
|--------------|----------|-----------------|---------|
| V-features | 339 | 175 | **164 (48%)** |
| Other features | 97 | 76 | 21 (22%) |
| **Total** | **436** | **251** | **185 (42%)** |

**Impact on Performance:**

| Metric | Reduced (252) | Full (435) | Improvement |
|--------|---------------|------------|-------------|
| F1 | 0.6556 | 0.6860 | **+4.6%** |
| Precision | 0.6946 | 0.7512 | **+8.1%** |
| Recall | 0.6208 | 0.6312 | **+1.7%** |
| AUC | 0.9356 | 0.9379 | **+0.2%** |

**Conclusion:** Correlation-based feature reduction is inappropriate for deep learning models. Neural networks can extract unique patterns from correlated features that traditional methods might consider redundant.

### 2.2 Feature Groups

**Full Dataset (435 features):**

| Group | Count | Description |
|-------|-------|-------------|
| Temporal | 30 | TransactionDT, D1-D15, C1-C14 |
| Relational | 52 | card1-6, addr1-2, email domains, device info, id_features |
| Behavioral | 353 | TransactionAmt, ALL V-features (V1-V339), M-features |

---

## 3. Model Architectures

### 3.1 Baseline Dense Network
Simple fully-connected network for baseline comparison.
```
Input (features) → Dense(256) → BatchNorm → ReLU → Dropout(0.3)
                 → Dense(128) → BatchNorm → ReLU → Dropout(0.3)
                 → Dense(64)  → BatchNorm → ReLU → Dropout(0.3)
                 → Dense(1)   → Sigmoid
```

**Results:** F1=0.6338, AUC=0.9138

### 3.2 LSTM + Dense Hybrid
Separates temporal and behavioral feature processing.
```
Temporal Features (30)  → Bidirectional LSTM(128) → LayerNorm
                                                         ↓
                                                  Concatenate → Dense → Output
                                                         ↑
Behavioral Features → Dense(256) → Dense(128) → Dense(64)
```

**Results:** F1=0.6435, AUC=0.9203 (+7.1% AUC improvement over baseline)

### 3.3 Sequence LSTM
Attempts to model transaction history per card.

- **Approach:** Group transactions by card1, create 5-transaction sequences
- **Results:** F1=0.2839, AUC=0.7811 (POOR)
- **Why it failed:**
  - V-features already encode temporal velocity patterns
  - card1 represents BIN ranges, not individual cardholders
  - Transaction-level fraud often doesn't depend on card history

### 3.4 Graph Neural Network (GNN)
Uses multi-head attention on relational features.

- **Approach:** Apply graph attention mechanism to card/device/email features
- **Results:** F1=0.1541, AUC=0.7770 (WORST)
- **Why it failed:**
  - Dataset lacks explicit graph structure (no transaction-to-transaction edges)
  - Relational features are hashed categorical IDs, not connected nodes
  - Without real graph topology, GNN degrades to attention over features

### 3.5 Fusion Model (LSTM + GNN + Dense) - BEST
Combines all three branches for comprehensive fraud detection.
```
Temporal (30 features)    → Bidirectional LSTM(64, 2 layers) → LayerNorm    → 128-dim
Relational (52 features)  → Multi-head Attention(64, 4 heads) → FFN         → 64-dim  
Behavioral (353 features) → Dense(256) → Dense(128)                         → 128-dim
                                                                                ↓
                                                                      Concatenate (320-dim)
                                                                                ↓
                                                              Dense(256) → Dense(128) → Output
```

**Results (Full Features):**
- **AUC: 0.9379** (Best)
- **F1: 0.6860** (Best)
- Precision: 0.7512
- Recall: 0.6312

---

## 4. Training Configuration

### 4.1 Loss Function: Focal Loss
Addresses extreme class imbalance (3.5% fraud):
```
FL(pt) = -α(1-pt)^γ × log(pt)
```

- α = 0.75 (weight for fraud class)
- γ = 2.0 (focusing parameter to down-weight easy examples)

### 4.2 Class Balancing: SMOTE
- Strategy: 0.5 (create synthetic samples until 33% fraud)
- Before SMOTE: 413,378 samples (3.5% fraud)
- After SMOTE: 598,371 samples (33.3% fraud)

### 4.3 Training Settings
```
Optimizer: Adam (lr=0.001)
Scheduler: ReduceLROnPlateau (patience=5, factor=0.5)
Early Stopping: patience=10 epochs
Batch Size: 512
Gradient Clipping: max_norm=1.0
```

### 4.4 Data Split
```
Train: 70% (413,378 → 598,371 after SMOTE)
Val:   15% (88,581)
Test:  15% (88,581)
```

---

## 5. Results

### 5.1 Complete Model Comparison

| Model | Features | F1 | Precision | Recall | AUC |
|-------|----------|-----|-----------|--------|-----|
| Baseline Dense | 251 | 0.6338 | 0.7674 | 0.5399 | 0.9138 |
| LSTM+Dense Hybrid | 251 | 0.6435 | 0.7433 | 0.5673 | 0.9203 |
| Sequence LSTM | 251 | 0.2839 | 0.3381 | 0.2447 | 0.7811 |
| GNN | 251 | 0.1541 | 0.0867 | 0.6909 | 0.7770 |
| Fusion (Reduced) | 252 | 0.6556 | 0.6946 | 0.6208 | 0.9356 |
| **Fusion (Full)** | **435** | **0.6860** | **0.7512** | **0.6312** | **0.9379** |

### 5.2 Threshold Analysis (Fusion - Full Features)

| Threshold | F1 | Precision | Recall | Use Case |
|-----------|-----|-----------|--------|----------|
| 0.3 | 0.2775 | 0.1649 | 0.8738 | Maximum fraud catch |
| 0.5 | 0.5043 | 0.3723 | 0.7812 | High recall |
| **0.7** | **0.6860** | **0.7512** | **0.6312** | **Balanced (recommended)** |
| 0.8 | 0.6684 | 0.9128 | 0.5273 | High precision |
| 0.9 | 0.5386 | 0.9771 | 0.3717 | Very high precision |

### 5.3 Reduced vs Full Features Impact

| Metric | Reduced | Full | Change |
|--------|---------|------|--------|
| F1 | 0.6556 | 0.6860 | **+4.6%** |
| Precision | 0.6946 | 0.7512 | **+8.1%** |
| Recall | 0.6208 | 0.6312 | **+1.7%** |
| AUC | 0.9356 | 0.9379 | **+0.2%** |

---

## 6. Explainability Analysis (SHAP)

### 6.1 Branch Contributions

| Branch | Reduced (252 feat) | Full (435 feat) | Change |
|--------|-------------------|-----------------|--------|
| Behavioral (Dense) | 61.0% | **71.0%** | +10.0% |
| Temporal (LSTM) | 23.8% | 16.9% | -6.9% |
| Relational (GNN) | 15.2% | 12.2% | -3.0% |

**Insight:** Adding back 164 V-features increased Behavioral branch dominance, confirming V-features are highly predictive for fraud detection.

### 6.2 Top 15 Most Important Features (Full Model)

| Rank | Feature | Branch | SHAP Value |
|------|---------|--------|------------|
| 1 | C14 | Temporal | 0.0147 |
| 2 | C1 | Temporal | 0.0116 |
| 3 | card6 | Relational | 0.0089 |
| 4 | **V135** | Behavioral | 0.0051 |
| 5 | **V147** | Behavioral | 0.0050 |
| 6 | C13 | Temporal | 0.0047 |
| 7 | **V77** | Behavioral | 0.0045 |
| 8 | TransactionAmt | Behavioral | 0.0044 |
| 9 | card2 | Relational | 0.0042 |
| 10 | **V281** | Behavioral | 0.0041 |
| 11 | **V283** | Behavioral | 0.0040 |
| 12 | C9 | Temporal | 0.0040 |
| 13 | D15 | Temporal | 0.0039 |
| 14 | card1 | Relational | 0.0035 |
| 15 | **V26** | Behavioral | 0.0035 |

**Key Finding:** V-features that were previously removed by correlation analysis (V135, V147, V77, V281, V283, V26) appear in the top 15 most important features. This validates our hypothesis that feature reduction hurt model performance.

### 6.3 Feature Category Insights

1. **C-features (C1, C9, C13, C14):** Count-based features are highly predictive
2. **Card features (card1, card2, card6):** Card type and BIN information matters
3. **V-features:** Vesta's proprietary features capture fraud patterns effectively
4. **TransactionAmt:** Transaction amount remains important

---

## 7. Conclusions

### 7.1 Key Findings

1. **Fusion architecture achieves best performance**
   - Combining LSTM + Attention + Dense captures complementary information
   - AUC=0.9379, F1=0.6860 (best among all models)

2. **Feature reduction hurts deep learning models**
   - Removing 48% of V-features cost 4.6% F1 improvement
   - Correlation-based reduction is inappropriate for neural networks
   - Deep learning can extract unique patterns from correlated features

3. **Vesta V-features are highly predictive**
   - Behavioral branch contributes 71% of predictive power
   - Previously removed V-features (V135, V147, V77) are in top 15

4. **Standalone sequence/graph models failed**
   - V-features already encode temporal patterns
   - Dataset lacks explicit graph structure for GNN

5. **Precision-recall tradeoff is fundamental**
   - Threshold 0.7: Balanced (F1=0.6860, Precision=0.7512)
   - Threshold 0.8: High precision (Precision=0.9128, Recall=0.5273)
   - Threshold 0.9: Very high precision (Precision=0.9771, Recall=0.3717)

### 7.2 Target Achievement

| Target | Best Achieved | Status |
|--------|---------------|--------|
| F1 > 0.85 | 0.6860 | Not met (-0.16) |
| Precision > 0.95 | 0.9771 (at 0.9) | **MET** |
| Recall > 0.83 | 0.6312 | Not met (-0.20) |

**Note:** Achieving high precision (>95%) requires sacrificing recall. At threshold 0.9, we achieve 97.7% precision but only detect 37% of fraud.

### 7.3 Production Recommendations

1. **Model:** Deploy Fusion (Full Features) model
2. **Threshold selection by use case:**
   - **Balanced:** 0.7 (F1=0.6860, catch 63% of fraud, 75% accuracy on flags)
   - **Low false positives:** 0.8 (91% accuracy on flags, catch 53% of fraud)
   - **Critical fraud prevention:** 0.9 (98% accuracy on flags, catch 37% of fraud)
3. **Feature engineering:** Retain ALL features for deep learning models
4. **Monitoring:** Track precision/recall in production, adjust threshold as needed

---

## 8. Project Structure
```
fraud_detection_DL/
├── data_files/
│   ├── train.csv                    # Original data (436 features)
│   ├── train_final.parquet          # Reduced features (251)
│   ├── X_*_full.npy                 # Full feature arrays (435)
│   └── full_feature_dims.json       # Feature dimensions
├── scripts_preprocessing/
├── scripts_modeling/                 # Baseline dense model
├── scripts_hybrid/                   # LSTM+Dense hybrid
├── scripts_sequence/
│   ├── script_lstm/                  # Sequence LSTM (failed)
│   ├── script_gnn/                   # GNN model (failed)
│   ├── script_hybrid/                # Fusion (reduced features)
│   ├── script_full_hybrid/           # Fusion (full features) - BEST
│   ├── compare_models.py
│   └── compare_models_full.py
├── models/
│   ├── best_model.pt                 # Baseline
│   ├── hybrid_model.pt               # LSTM+Dense
│   ├── sequence_model.pt             # Sequence LSTM
│   ├── gnn_model.pt                  # GNN
│   ├── fusion_model.pt               # Fusion (reduced)
│   └── fusion_model_full.pt          # Fusion (full) - BEST
├── results_sequence/
│   ├── result_lstm/
│   ├── result_gnn/
│   ├── result_hybrid/
│   ├── result_hybrid_full/           # Full features SHAP analysis
│   └── result_full_comparison/       # Final comparison charts
├── logs/
└── DOCUMENTATION.md                   # This file
```

---

## 9. Future Work

1. **Explicit graph construction:** Build transaction graphs based on shared attributes (device fingerprint, shipping address, IP) to leverage GNN capabilities

2. **Ensemble methods:** Combine Fusion model with gradient boosting (XGBoost, LightGBM) for potential improvement

3. **Online learning:** Implement incremental learning to adapt to evolving fraud patterns

4. **Feature interaction modeling:** Use attention mechanisms to capture feature interactions within branches

5. **Cost-sensitive learning:** Incorporate business costs (fraud loss vs customer friction) into the loss function

---

## 10. References

1. IEEE-CIS Fraud Detection Dataset - Kaggle Competition
2. Lin, T.Y. et al. (2017). Focal Loss for Dense Object Detection
3. Chawla, N.V. et al. (2002). SMOTE: Synthetic Minority Over-sampling Technique
4. Lundberg, S.M. & Lee, S.I. (2017). A Unified Approach to Interpreting Model Predictions (SHAP)
5. Vaswani, A. et al. (2017). Attention Is All You Need

---

*Report generated: {date}*
*Author: Deep Learning Fraud Detection Project*
'''.format(date=datetime.now().strftime("%B %d, %Y"))
    
    with open('DOCUMENTATION.md', 'w', encoding='utf-8') as f:
        f.write(doc)
    
    print("=" * 70)
    print("DOCUMENTATION GENERATED SUCCESSFULLY")
    print("=" * 70)
    print("\nSaved to: DOCUMENTATION.md")
    print("\nKey sections:")
    print("  1. Executive Summary")
    print("  2. Feature Engineering Analysis (with reduction impact)")
    print("  3. Model Architectures")
    print("  4. Training Configuration")
    print("  5. Results (all models + threshold analysis)")
    print("  6. Explainability Analysis (SHAP)")
    print("  7. Conclusions")
    print("  8. Project Structure")
    print("  9. Future Work")
    print("  10. References")

if __name__ == '__main__':
    generate_documentation()