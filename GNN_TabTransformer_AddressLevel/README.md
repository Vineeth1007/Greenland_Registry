# 🧠 Ethereum Address-Level Fraud Detection (GNN • TabTransformer • LightGBM)

This repository implements a **secure, leakage-free, and reproducible AI pipeline** for Ethereum address-level fraud detection.  
It combines **graph-based**, **transformer-based**, and **gradient boosting** models with verified data integrity, based on the paper *“GreenLand: A Secure Land Registration Scheme for Blockchain and AI-Enabled Agriculture Industry 5.0”*.

## 📘 Overview
The goal is to detect fraudulent blockchain addresses using a hybrid of **graph learning (GNN)**, **tabular deep learning (TabTransformer)**, and **gradient boosting (LightGBM)** on Ethereum transaction-level features.  
Each model is evaluated on **exactly the same canonical test set**, ensuring fair comparison and zero data leakage.

## 🚀 Key Highlights
✅ Group-aware splitting (no address overlap).  
✅ Train-only preprocessing (no leakage).  
✅ Rare-value sanitization for categorical features.  
✅ Unified output naming for automatic evaluation.  
✅ Verified reproducibility and fairness.

## 🧩 Pipeline Overview
### 1️⃣ Data Preprocessing
- Deduplicated addresses (numeric=mean, categorical=mode).  
- Used GroupShuffleSplit: 80/20, then 15% of train → validation (≈68/12/20).  
- Train-only fitting for LabelEncoders + StandardScaler.  
- Saved canonical splits:
  - X_train.npy / X_val.npy / X_test.npy
  - y_train.npy / y_val.npy / y_test.npy
  - addresses_train.npy / addresses_val.npy / addresses_test.npy
  - preproc_train_encoders.pkl

### 2️⃣ Model Training
- **LightGBM:** GBDT with early stopping via callbacks.  
- **GraphSAGE (GNN):** Aligned node indices via address mapping.  
- **TabTransformer:** Reused canonical splits; no new random split.  
- **Baselines:** Logistic Regression & SVM on same data.

### 3️⃣ Model Comparison
- Robust auto-discovery cell to load correct *_test_probs.npy per model.  
- Computes ROC-AUC, PR-AUC, Precision, Recall, F1, Accuracy.  
- Generates metrics table, ROC, PR, and confusion matrix plots.

## 📊 Final Results (Canonical Test Set, n=1964)
| Model | ROC-AUC | PR-AUC | Precision | Recall | F1 | Accuracy |
|--------|----------|--------|------------|---------|----|-----------|
| LightGBM | **0.9992** | **0.9981** | 1.000 | 0.972 | 0.986 | 0.994 |
| SVM | 0.9952 | 0.9868 | 0.988 | 0.919 | 0.952 | 0.980 |
| TabTransformer | 0.9949 | 0.9869 | 0.997 | 0.907 | 0.950 | 0.979 |
| GNN (GraphSAGE) | 0.9896 | 0.9816 | 0.926 | 0.937 | 0.932 | 0.970 |
| LogReg | 0.9180 | 0.8226 | 0.778 | 0.870 | 0.821 | 0.917 |

## 🧠 Findings
- LightGBM outperforms deep models on structured Ethereum data.  
- Deep models improve after alignment with canonical splits.  
- No leakage detected (validated via ablation & permutation tests).

## 📁 Repository Structure
```
├── artifacts/
│   ├── X_train.npy, X_val.npy, X_test.npy
│   ├── y_train.npy, y_val.npy, y_test.npy
│   ├── addresses_train.npy, addresses_val.npy, addresses_test.npy
│   ├── preproc_train_encoders.pkl
│   ├── *_test_probs.npy / *_test_preds.npy / *_test_true.npy
│   ├── models_comparison_metrics.csv
│   └── lightgbm_booster.txt
├── preprocessing_group_aware.py
├── build_knn_graph.py
├── train_lightgbm.py
├── train_gnn.py
├── train_tabtransformer.py
├── compare_models.py
├── requirements.txt
└── README.md
```

## 🧾 Citation
> Vineeth C S, *Ethereum Address-Level Fraud Detection via Graph and Transformer Models*, 2025.

## 🧪 Environment
- Python ≥ 3.10  
- scikit-learn ≥ 1.3  
- LightGBM ≥ 4.3  
- PyTorch ≥ 2.1  
- torch-geometric ≥ 2.4  
- pandas, numpy, matplotlib, seaborn

## 🏁 Next Steps
- Temporal hold-out evaluation.  
- Cross-validation (5-fold AUC).  
- SHAP interpretability analysis.  
- Publish sanitized dataset for reproducibility.

---
**Maintainer:** C S Vineeth  
AI / Blockchain Researcher  
