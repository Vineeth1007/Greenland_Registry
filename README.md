# **Ethereum Address-Level Models: Hybrid Architectures and Deep Learning Approaches**

This repository presents multiple approaches for **Ethereum address-level classification**, combining both **deep learning** and **tree-based** methods to capture tabular and relational patterns effectively.  
Recent experiments introduce **hybrid architectures** — integrating **TabTransformer + LightGBM** and **GNN + LightGBM** — along with a **fine-tuned Graph Neural Network (GNN)** that significantly improves upon the earlier baseline.

---

## **Repository Contents**

- **`GNN_TabTransformer_AddressLevel.ipynb`** — Main Jupyter notebook that:
  - Loads and preprocesses `/mnt/data/ethereum_raw.csv`
  - Performs address-level feature engineering and builds a k-NN graph for GNNs
  - Implements and evaluates the following models:
    - **Hybrid (TabTransformer + LightGBM)**
    - **Hybrid (GNN + LightGBM)**
    - **Fine-tuned GNN**
    - **LightGBM**
    - **TabTransformer**
    - **SVM**
    - **Logistic Regression**
  - Saves trained artifacts, metrics, and plots in `./artifacts/`

- **`requirements.txt`** — Python dependencies required to reproduce all experiments  
- **`run.sh`** — Script to create a virtual environment and install dependencies automatically  

---

## **Highlights of the Updated Version**

- ✅ Added **two hybrid ensemble models**:
  - **Hybrid (TabTransformer + LightGBM)** — Combines transformer-based feature embeddings with gradient-boosted decision trees for improved accuracy.  
  - **Hybrid (GNN + LightGBM)** — Integrates structural node embeddings from GNN with tabular boosting for enhanced relational learning.

- ✅ Implemented a **fine-tuned GNN**, optimized through:
  - Larger hidden dimensions  
  - Improved learning rate scheduling and dropout  
  - Better neighborhood aggregation for more expressive graph embeddings  

- ✅ Updated evaluation pipeline:
  - Computed **ROC-AUC**, **PR-AUC**, **Precision**, **Recall**, and **F1-score** for all models  
  - Added **comparison plots**, **confusion matrices**, and a **performance heatmap**

---

## **Performance Summary**

| Model | ROC-AUC | PR-AUC | Precision | Recall | F1-Score |
|:------|:-------:|:-------:|:----------:|:--------:|:---------:|
| **Hybrid (T+LGBM)** | **0.999498** | **0.998422** | **0.995261** | **0.976744** | **0.985915** |
| **Hybrid (LightGBM + GNN)** | **0.999263** | **0.997710** | **0.992891** | **0.974419** | **0.983568** |
| **LightGBM** | **0.999219** | **0.998079** | **1.000000** | **0.972093** | **0.985849** |
| **Fine-tuned GNN** | **0.998866** | **0.996544** | **0.993464** | **0.965116** | **0.979088** |
| **TabTransformer** | **0.998654** | **0.996321** | **0.993464** | **0.962791** | **0.977951** |
| **SVM** | **0.995806** | **0.992718** | **0.986607** | **0.953488** | **0.969767** |
| **Logistic Regression** | **0.994964** | **0.989233** | **0.986486** | **0.941860** | **0.963636** |

---

## **Key Observations**

- The **Hybrid (TabTransformer + LightGBM)** model achieved the best overall performance, offering near-perfect classification with a ROC-AUC of **0.9995**.  
- The **Hybrid (GNN + LightGBM)** closely followed, confirming that combining learned graph embeddings with tree-based models yields strong generalization.  
- The **Fine-tuned GNN** outperformed the earlier baseline by improving aggregation depth, showing that deeper message-passing significantly enhances structural learning.  
- **LightGBM** remains a powerful baseline due to its strong inductive bias on tabular data.  
- Traditional models like **SVM** and **Logistic Regression** performed respectably but could not match the representational depth of the hybrid and deep learning models.

---

## **Why Hybrid Models Excel on Address-Level Data**

1. **Complementary Learning**: Transformers and GNNs encode contextual or relational information, while LightGBM efficiently captures non-linear decision boundaries.
2. **Improved Feature Utilization**: TabTransformer provides richer feature embeddings, and GNN captures graph-structural dependencies between addresses.
3. **Reduced Overfitting**: Combining dense embeddings with gradient boosting mitigates overfitting seen in standalone deep networks.
4. **Balanced Precision–Recall Tradeoff**: Both hybrids maintain high recall without compromising precision — ideal for imbalanced blockchain datasets.

---

## **How to Run**

```bash
bash run.sh        # Creates a virtual environment and installs dependencies
jupyter lab        # or jupyter notebook
# Open GNN_TabTransformer_AddressLevel.ipynb and execute cells in order
