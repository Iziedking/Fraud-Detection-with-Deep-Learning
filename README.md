# Deep Learning for E-Commerce Fraud Detection

A hybrid deep learning system for detecting fraudulent e-commerce transactions using the IEEE-CIS dataset. This project implements and compares multiple neural network architectures including LSTM, Graph Neural Networks, and a multi-branch fusion model.

## Background

E-commerce fraud detection is challenging due to severe class imbalance (typically <4% fraud) and evolving fraud patterns. This project explores whether combining different neural network architectures that specialize in different aspects of transaction data can improve detection performance.

## Results Summary

| Model | F1 | Precision | Recall | AUC |
|-------|-----|-----------|--------|-----|
| Baseline Dense | 0.6338 | 0.7674 | 0.5399 | 0.9138 |
| LSTM and Dense Hybrid | 0.6435 | 0.7433 | 0.5673 | 0.9203 |
| Sequence LSTM | 0.2839 | 0.3381 | 0.2447 | 0.7811 |
| GNN | 0.1541 | 0.0867 | 0.6909 | 0.7770 |
| Fusion (Reduced Features) | 0.6556 | 0.6946 | 0.6208 | 0.9356 |
| Fusion (Full Features) | 0.6860 | 0.7512 | 0.6312 | 0.9379 |

The fusion model with full features achieved the best overall performance.

## Notable Finding

During development, we discovered that correlation-based feature reduction (a common preprocessing step) actually hurt model performance. The original preprocessing removed 164 V-features (48%) deemed redundant. Restoring these features improved F1 score by 4.6%. This suggests that for deep learning models, retaining all features and letting the network learn feature importance is preferable to manual feature selection.

## Architecture

The fusion model combines three specialized branches:

**Temporal Branch (LSTM)**: Processes time-related features (TransactionDT, C1-C14, D1-D15) using a bidirectional LSTM to capture sequential patterns.

**Relational Branch (Attention)**: Applies multi-head attention to card, device, and identity features to model relationships between transaction attributes.

**Behavioral Branch (Dense)**: Processes transaction amount and Vesta's proprietary V-features through fully connected layers.

The outputs from all three branches are concatenated and passed through fusion layers for final classification.
```
Temporal (30 features)   -> Bidirectional LSTM (64 units) -> 128-dim
Relational (52 features) -> Multi-head Attention (4 heads) -> 64-dim
Behavioral (353 features) -> Dense (256 -> 128) -> 128-dim
                                    |
                              Concatenate (320-dim)
                                    |
                              Dense (256 -> 128 -> 1)
                                    |
                              Fraud Probability
```

## Setup

### Requirements
- Python 3.10+
- PyTorch 2.0+
- 8GB+ RAM (16GB recommended)
- GPU optional but recommended for training

### Installation
```bash
git clone https://github.com/YOUR_USERNAME/fraud_detection_DL.git
cd fraud_detection_DL

python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows

pip install -r requirements.txt
```

### Data

Download the IEEE-CIS Fraud Detection dataset from Kaggle:
https://www.kaggle.com/c/ieee-fraud-detection/data

Place `train_transaction.csv` in `data_files/` and rename to `train.csv`.

### Training
```bash
# Prepare data with all features
python data_exploration/prepare_full_features.py

# Train fusion model
python scripts_sequence/script_full_hybrid/train_full.py

# Run SHAP analysis
python scripts_sequence/script_full_hybrid/explain_full.py

# Generate comparison visualizations
python scripts_sequence/compare_models_full.py
```

## Methods

### Class Imbalance Handling
- SMOTE oversampling (strategy=0.5) on training data only
- Focal Loss (alpha=0.75, gamma=2.0) to prioritize hard examples

### Training Configuration
- Optimizer: Adam (lr=0.001)
- Early stopping with patience=10
- Gradient clipping (max_norm=1.0)
- ReduceLROnPlateau scheduler

### Evaluation
- Stratified train/validation/test split (70/15/15)
- Metrics: F1, Precision, Recall, AUC-ROC
- Threshold analysis from 0.3 to 0.9
- SHAP explainability analysis

## What Worked and What Didn't

**Worked:**
- Multi-branch fusion architecture outperformed single-model approaches
- Focal loss helped with class imbalance
- Keeping all features improved results

**Didn't work:**
- Sequence LSTM (card1 doesn't represent unique users reliably)
- Standalone GNN (dataset lacks explicit graph structure between transactions)
- Correlation-based feature reduction

## SHAP Analysis Results

Branch contributions to predictions:
- Behavioral (V-features): 71%
- Temporal: 17%
- Relational: 12%

The V-features provided by Vesta dominate the predictions, suggesting they already encode significant fraud signals.

## Threshold Selection

The model outputs a fraud probability. Choosing where to set the decision threshold depends on business requirements:

| Threshold | Precision | Recall | Use Case |
|-----------|-----------|--------|----------|
| 0.7 | 75% | 63% | Balanced |
| 0.8 | 91% | 53% | Minimize false alarms |
| 0.9 | 98% | 37% | Very high confidence only |

## Dependencies
```
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
imbalanced-learn>=0.11.0
torch>=2.0.0
pyarrow>=14.0.0
joblib>=1.3.0
matplotlib>=3.7.0
seaborn>=0.12.0
shap>=0.42.0
```

## Documentation

See DOCUMENTATION.md for the full technical report including detailed methodology, all experimental results, and analysis.

## Dataset

IEEE-CIS Fraud Detection Dataset provided by Vesta Corporation through Kaggle.

## License

MIT
