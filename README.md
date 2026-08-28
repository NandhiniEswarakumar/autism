# ASD Detection Using TabM & SHAP

This repository contains an end-to-end implementation for screening Autism Spectrum Disorder (ASD) using a Tabular Mini-batch Ensemble model (TabM) and SHAP for model explainability.

---

## Project Overview

This project implements a complete end-to-end pipeline for ASD detection using the **NSCH 2023** (National Survey of Children's Health) dataset.

### Highlights
- Implementation of the TabM (Tabular Mini-batch Ensemble) architecture in PyTorch
- Model-agnostic SHAP explainability (KernelExplainer) for global and local explanations
- Optuna-based hyperparameter optimization (TPE sampler and pruning)
- SMOTE oversampling to address class imbalance in the dataset
- Ablation experiments and robustness tests (noise, missing values, cross-validation)
- A basic Streamlit web application for interactive prediction and explanation

---

## Workflow

The repository is organized as a sequence of reproducible steps starting from data preparation and exploratory analysis, followed by preprocessing (including imputation and SMOTE), feature selection, model definition and training with Optuna, evaluation and explainability with SHAP. A Streamlit application demonstrates how to make predictions and inspect explanations.

---

## Project Structure

```
asd project/
├── dataset/                     # Data files
│   ├── nsch_2023e_topical.sas7bdat   # Raw NSCH 2023 dataset
│   ├── asd_selected_dataset.csv      # Selected columns
│   ├── cleaned_asd_dataset.csv       # Cleaned dataset
│   ├── X_train_final.npy             # Final scaled + SMOTE train
│   ├── y_train_final.npy
│   ├── X_val_final.npy               # Validation set
│   ├── y_val_final.npy
│   ├── X_test_final.npy              # Test set
│   ├── y_test_final.npy
│   └── selected_features.npy         # Chosen feature names
├── src/
│   ├── 01_eda.py                # Exploratory Data Analysis
│   ├── 02_preprocessing.py      # Full preprocessing + SMOTE
│   ├── 03_feature_selection.py  # MI + F-score feature selection
│   ├── 04_tabm_model.py         # TabM PyTorch architecture
│   ├── tabm_model.py            # Alias module for imports
│   ├── 05_train.py              # Optuna HPO + training
│   ├── 06_evaluate.py           # Full performance evaluation
│   ├── 07_shap_explain.py       # SHAP global + local explanations
│   ├── 08_compare_models.py     # Baseline model comparison
│   ├── 09_ablation.py           # Ablation study
│   └── 10_robustness.py         # Robustness testing
├── models/
│   ├── tabm_best.pth            # Best TabM model weights
│   ├── preprocessor.pkl         # StandardScaler + metadata
│   └── selected_preprocessor.pkl
├── results/
│   ├── plots/                   # All generated figures (29 plots)
│   ├── metrics.json             # Performance metrics
│   ├── model_config.json        # Best hyperparameters + config
│   ├── best_hyperparams.json    # Optuna best params
│   ├── comparison_table.csv     # Model comparison results
│   ├── ablation_results.csv     # Ablation study results
│   ├── robustness_results.json  # CV + noise + missing tests
│   └── shap_values.npy          # Saved SHAP values
├── app/
│   └── streamlit_app.py         # Streamlit web application
├── main.py                      # Full pipeline orchestrator
└── requirements.txt             # Dependencies
```

---

## Setup & Installation

### 1. Create a virtual environment
```powershell
python -m venv .venv
.venv\Scripts\activate     # Windows PowerShell
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the full pipeline
```powershell
python main.py
```

### 4. Run individual steps
```bash
python main.py --step 1          # EDA only
python main.py --step 5 6 7      # Train + Evaluate + SHAP
python main.py --skip 7 8        # Skip SHAP + comparison
```

### 5. Launch the Streamlit app
```powershell
streamlit run app/streamlit_app.py
```

---

## TabM Architecture

TabM (Tabular Mini-batch Ensemble) uses **BatchEnsemble** for efficient ensembling:

```
Input (n_features)
      ↓
[Linear → BatchNorm → GELU → Dropout] × n_layers
      ↓  ← Each layer uses K rank-1 perturbation vectors
[Head Linear]  → K predictions
      ↓
[Mean over K heads]  → Final logit
      ↓
σ(logit)  → P(ASD)
```

**Key equation per head k:**
```
h_k(x) = (x ⊙ r_k) @ W ⊙ s_k + b
ŷ = σ(mean_k[head_k(h_k)])
```

---

## Performance (Example)

| Metric      | TabM   | MLP    | XGBoost |
|-------------|--------|--------|---------|
| ROC-AUC     | 0.88+  | 0.83   | 0.85    |
| F1-Score    | 0.72+  | 0.65   | 0.68    |
| Accuracy    | 0.91+  | 0.87   | 0.89    |

*Results depend on dataset and hyperparameters — run pipeline for actual values*

---

## Dataset

- **Source**: National Survey of Children's Health (NSCH) 2023
- **Features**: 7 clinical features (age, sex, comorbidities)
- **Target**: K2Q35A (ASD diagnosis: 1=Yes, 0=No)
- **Class Imbalance**: ~2-3% ASD prevalence → handled by SMOTE

---

## Dependencies

| Package | Purpose |
|---------|---------|
| `torch` | TabM model implementation |
| `shap` | SHAP explainability |
| `optuna` | Hyperparameter optimization |
| `imbalanced-learn` | SMOTE oversampling |
| `streamlit` | Web application |
| `xgboost/lightgbm/catboost` | Baseline models |
| `pytorch-tabnet` | TabNet baseline |

---

## Citation

If you use this work, please cite:

```bibtex
@article{tabm_asd_2024,
  title={Autism Spectrum Disorder Detection Using TabM and SHAP: 
         A Novel Explainable Deep Learning Framework},
  author={},
  journal={},
  year={2026}
}
```

> **Reference**: Gorishniy et al., "TabM: Advancing Tabular Deep Learning with 
> Parameter-Efficient Ensembling", NeurIPS 2024.
