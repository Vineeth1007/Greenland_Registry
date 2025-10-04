# Ethereum Address-Level Models: LightGBM, GNN, TabTransformer

This bundle contains:
- `GNN_TabTransformer_AddressLevel.ipynb` — Jupyter notebook that:
  - Loads `/mnt/data/ethereum_raw.csv`
  - Performs minimal cleanup (drops index columns)
  - Builds tabular features and a k-NN graph (for GNN)
  - Trains a LightGBM baseline, a GraphSAGE node-classifier, and a Tabular Transformer
  - Evaluates and saves artifacts in `./artifacts/`
- `requirements.txt` — Python dependencies for the notebook
- `run.sh` — example script to create venv and install requirements

## Why tree models often beat deep models on tabular data

Short answer: **feature engineering + inductive bias**. Gradient-boosted decision trees (GBDT, e.g., LightGBM/XGBoost/CatBoost) are extremely effective on engineered tabular features for several reasons:
- They handle heterogeneous data (numeric + categorical) and missing values naturally.
- They capture non-linear feature interactions with low risk of overfitting when features are informative.
- For relatively small/medium-sized datasets (thousands to low millions of rows), GBDTs often require less data to generalize well compared to deep nets.
- Feature engineering (aggregations, time-based stats, domain-specific ratios) gives trees very direct splits to exploit.

In your pipeline:
- The dataset already contains ~48 well-crafted aggregated features (total sent/received, avg times, ERC20 summaries). This gives LightGBM a big advantage out-of-the-box.
- The GNN used a **k-NN graph** built from feature similarity (synthetic), which may not provide meaningful relational signal beyond the features themselves.
- The TabTransformer in the notebook treats the entire row as a single projected vector — not full per-feature attention — so it isn't fully exploiting transformer strengths.

## How to make deep models outperform trees (practical checklist)
1. **Use real graph edges** (transaction links) for GNN message passing instead of k-NN.
2. **Feature-tokenize for TabTransformer**: treat each column as a token, use embeddings for categorical columns, and let the transformer attend across features.
3. **Increase model capacity**: larger d_model, more transformer layers, wider GNN layers.
4. **Better handling of categorical high-cardinality**: use embeddings for tokens (don’t label-scale them), or hash bucket into manageable cardinality.
5. **Regularization & tuning**: LR schedules, early stopping, weight decay, dropout, and hyperparameter search (Optuna).
6. **Data scale**: deep models often need more data to shine — getting more labeled addresses or augmenting data can help.
7. **Ensemble**: average GBDT + deep model predictions — often yields small but useful gains.

## Files included
- `GNN_TabTransformer_AddressLevel.ipynb`
- `README.md`
- `requirements.txt`
- `run.sh`

## How to run (quick)
```bash
bash run.sh      # creates venv and installs requirements
jupyter lab      # or jupyter notebook
# open GNN_TabTransformer_AddressLevel.ipynb and run cells in order
```
