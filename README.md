# Stability-Guided Ensemble Pruning for Improved Generalization

A university research project that proposes and evaluates a **stability-guided pruning strategy** for machine learning ensembles applied to binary medical classification tasks.

The core idea: instead of using all available classifiers in an ensemble, first evaluate each model's stability (consistency across repeated cross-validation folds), prune unstable or weak models, then build the ensemble from only the most reliable ones.

---

## Datasets

| Dataset | Source | Samples | Features | Task |
|---|---|---|---|---|
| Heart Disease | UCI ML Repository (id=45) | 303 | 13 | Binary (disease / no disease) |
| Pima Indians Diabetes | Brownlee GitHub CSV | 768 | 8 | Binary (diabetic / non-diabetic) |

Both datasets are downloaded programmatically — no manual downloads required.

---

## Pipeline Overview

### Stage 1 — Stability Evaluation
Five base classifiers (LR, SVM, kNN, RF, XGBoost) are evaluated using `RepeatedStratifiedKFold(n_splits=5, n_repeats=5)` — 25 folds per model. Each model is scored on 8 metrics per fold. A **Stability Score** is computed as:

```
Stability Score = Mean ROC-AUC / (Std ROC-AUC + 1e-9)
```

### Stage 2 — Pruning
Models are selected where **both** conditions hold:
- Mean ROC-AUC ≥ average across all models
- Std ROC-AUC ≤ average across all models

If fewer than 2 models qualify, thresholds are relaxed incrementally until at least 2 are selected.

### Stage 3 — Method Comparison
Five methods are evaluated and compared:
1. Individual classifiers (default params) — reused from Stage 1
2. Full Soft Voting Ensemble (all 5 models)
3. Full Stacking Ensemble (all 5 models, LR meta-learner)
4. Pruned Soft Voting Ensemble (selected models, default params)

### Stage 4 — Hyperparameter Tuning
`GridSearchCV(cv=3)` is run **only on the pruned models** to avoid bias. A final tuned pruned voting ensemble is then built and evaluated as method 5.

---

## Evaluation Metrics

All methods are scored on 8 metrics across all 25 folds:

| Metric | Description |
|---|---|
| Accuracy | Overall correct predictions |
| Recall (Sensitivity) | True positive rate |
| Precision | Positive predictive value |
| Specificity | True negative rate |
| F1 (weighted) | Harmonic mean of precision & recall |
| ROC-AUC | Area under the ROC curve |
| PR-AUC | Area under the Precision-Recall curve |
| MCC | Matthews Correlation Coefficient |

---

## Preprocessing

| Dataset | Imputation | Scaling | Imbalance handling |
|---|---|---|---|
| Heart Disease | `SimpleImputer(median)` | `StandardScaler` | None (balanced) |
| Pima Diabetes | `IterativeImputer(max_iter=10)` | `StandardScaler` | `SMOTE` |

All preprocessing is done **inside each CV fold** via scikit-learn / imblearn `Pipeline` objects — no data leakage.

---

## Setup

**Requirements:** Python 3.10+

```bash
# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate        # macOS/Linux
# .venv\Scripts\activate         # Windows

# Install dependencies
pip install -r requirements.txt

# macOS only — required by XGBoost
brew install libomp
```

---

## Running the Notebook

```bash
# Option 1: Execute all cells non-interactively (recommended for full pipeline)
jupyter nbconvert --to notebook --execute ensemble_pruning.ipynb --output ensemble_pruning.ipynb

# Option 2: Open interactive session
jupyter notebook ensemble_pruning.ipynb
```

---

## Output Files

After a full run, the following files are generated in the project directory:

**Clean datasets:**
- `heart_disease_clean.csv`
- `pima_diabetes_clean.csv`

**Stability results (Stage 1):**
- `stability_results_heart.csv`
- `stability_results_pima.csv`

**Final comparison results (all methods + all metrics):**
- `final_results_heart.csv`
- `final_results_pima.csv`

**Plots:**
- `plot_auc_comparison_heart.png` — ROC-AUC bar chart with error bars
- `plot_auc_comparison_pima.png`
- `plot_stability_heatmap.png` — Stability scores heatmap across both datasets
- `plot_boxplot_heart.png` — AUC distribution across folds (box plots)
- `plot_boxplot_pima.png`
- `plot_recall_spec_heart.png` — Recall vs Specificity grouped bar chart
- `plot_recall_spec_pima.png`
- `plot_mcc_heart.png` — MCC comparison bar chart
- `plot_mcc_pima.png`

---

## Project Structure

```
MLProject/
├── ensemble_pruning.ipynb   # Main notebook (all pipeline logic)
├── requirements.txt         # Pinned dependencies
├── README.md
├── *.csv                    # Generated result files
└── *.png                    # Generated plot files
```

`sandbox.ipynb` (gitignored) is available for scratch work.

---

## Key Design Decisions

- **Pruning before tuning:** Stability comparison uses default hyperparameters to ensure a fair, unbiased evaluation. Tuning is applied only after pruning, solely to enhance the final ensemble.
- **No leakage:** Imputation, scaling, and SMOTE are all applied inside CV folds via Pipeline.
- **SMOTE only for Pima:** Applied because Pima has a moderate class imbalance (268/500 ≈ 0.54 minority ratio) and zero-encoded missing values require careful imputation.
