# GreenLand — AI Baseline & Feature Engineering (Address-level Ethereum Fraud)

**Project**: Reproducing & extending the AI portion of *GreenLand: A Secure Land Registration Scheme* (Ethereum fraud detection).  
**Dataset**: `ethereum_raw.csv` (address-level aggregated dataset).  
**Focus**: baseline ML models (LR, RF, XGBoost, LightGBM), preprocessing pipeline, evaluation, and visualizations.  
_Note: this repository contains code and notebooks for the AI experiments only — blockchain/smart-contract parts are out of scope here._

---

## Repo contents
- `Preprocessing.ipynb` — main Jupyter notebook (EDA, preprocessing, baseline models, evaluation, CV, plots).  
- `feature_engineering.py` — (optional) script form of pipeline used in notebook.  
- `requirements.txt` — Python dependencies.  
- `.gitignore` — ignore rules (includes `data/` by default).  
- `colname_mapping.json` — mapping of sanitized column names (if generated).  
- `results/` — (optional) folder for model output, saved metrics (not tracked).

> **Important**: Do **not** commit large raw data files to this repo. Keep `ethereum_raw.csv` in a local `data/` folder and add it to `.gitignore`. If you need to share data, use external storage or GitHub Releases / Git LFS.

---

## Quick start (Windows PowerShell)

1. Create virtual environment (Python 3.10 recommended), install dependencies:
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

2. Open the notebook:
```powershell
jupyter notebook
```
Then open `Preprocessing.ipynb` in the browser and run cells sequentially.

---

## Reproducing baseline results
- Ensure `ethereum_raw.csv` (address-level dataset) is available in a `data/` folder or update the path in the notebook.
- Run all notebook cells sequentially.
- Results are saved (optional) to `/mnt/data/ethereum_baseline_features.parquet` or CSV fallback.

---

## Notes & recommendations
- The dataset used here is **address-level aggregated** (one row per address). Results can be very high because these features are strong predictors of fraud. Removing ID-like leakage columns (`Unnamed:*`, `Index`) is required.
- If you want transaction-level experiments (more realistic & harder), you need raw transaction-edge data (not included).
- For interpretability use SHAP on the LightGBM model (not included by default).

---

## License & citation
This repository is for educational and research purposes. If you use this work for academic or research publications, please cite the original paper:

> F. Naguji et al., “GreenLand: A Secure Land Registration Scheme for Blockchain and AI-Enabled Agriculture Industry 5.0”, IEEE Access, 2024.

---

## Contact
Your Name — your.email@example.com
