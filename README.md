# Clinical Mortality Prediction Pipeline (HOSP_ADMIT)

## Overview
This repository contains an end-to-end Machine Learning Operations (MLOps) pipeline designed to predict in-hospital mortality for patients admitted with myocardial infarction (MI). 

Unlike standard academic exercises, this pipeline is engineered to handle real-world clinical data challenges, including severe class imbalance (84% survival baseline), high-dimensional sparse categorical features (ECG leads), physiological multicollinearity, and Missing-Not-At-Random (MNAR) clinical data.

## Architectural Highlights
* **Leak-Proof Preprocessing:** Utilizes stateful `scikit-learn` ColumnTransformers. Missing clinical onset times (`TIME_B_S`) are handled via median imputation combined with missingness indicators (`add_indicator=True`) to preserve the biological signal of technical dropouts.
* **Precision-Recall Evaluation:** Due to the 84/16 class imbalance, standard accuracy and standard K-Fold CV are avoided. The pipeline utilizes `StratifiedKFold` and is optimized/evaluated exclusively on **PR-AUC** and **Brier Score Calibration**.
* **Dimensionality Management:** Employs L1 (LASSO) and L2 (Ridge) regularization to safely manage Events-Per-Variable (EPV) limit breaches and quasi-complete separation common in sparse clinical datasets.
* **Non-Linear Biological Modeling:** Integrates Generalized Additive Models (GAMs) via `pygam` to mathematically prove non-linear physiological thresholds (e.g., U-shaped blood pressure risks and exponential sepsis spikes).
* **Interaction Mapping:** Utilizes Random Forests natively to map compounding multiplicative risks (e.g., Age × Leukocytes × Hypotension).

## Project Structure
```text
hosp-admit-mlops/
├── data/
│   ├── raw/                 # Original HOSP_ADMIT.csv (Never modified)
│   └── processed/           # Stateful engineered features (X_train, y_train, etc.)
├── src/
│   ├── preprocess.py        # Pipeline definition, imputation, and scaling
│   └── train.py             # Orchestrator for Unregularized, Regularized, GAM, and RF models
├── models/                  # Exported .pkl artifacts for deployment
├── reports/
│   ├── figures/             # Partial Dependence Plots (GAM) & Feature Importance (RF)
│   └── evaluation_results.txt # Persistent logging of all metrics
└── README.md
