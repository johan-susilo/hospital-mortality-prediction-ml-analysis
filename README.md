# Clinical Mortality Prediction Pipeline (HOSP_ADMIT)

## Overview
This repository contains an end-to-end Machine Learning pipeline designed to predict in-hospital mortality for patients admitted with myocardial infarction (MI).

This pipeline is engineered to handle real-world clinical data challenges present in the `HOSP_ADMIT` dataset, including severe class imbalance (84% survival baseline), high-dimensional sparse categorical features (ECG leads), physiological multicollinearity, and Missing-Not-At-Random (MNAR) clinical data.

**Note:** If you are looking for the evaluation report covering the assignment questions, please refer to the [`report.md`](./report.md) file.

## Architectural Highlights
* **Leak-Proof Preprocessing:** Utilizes stateful `scikit-learn` ColumnTransformers. Missing clinical onset times (`TIME_B_S`) are handled via median imputation combined with missingness indicators (`add_indicator=True`) to preserve the biological signal of technical dropouts.
* **Precision-Recall Evaluation:** Due to the 84/16 class imbalance, standard accuracy and standard K-Fold CV are avoided. The pipeline utilizes `StratifiedKFold` and is optimized/evaluated exclusively on **PR-AUC** and **Brier Score Calibration**.
* **Dimensionality Management:** Employs L1 (LASSO) and L2 (Ridge) regularization to safely manage Events-Per-Variable (EPV) limit breaches and quasi-complete separation common in sparse clinical datasets.
* **Non-Linear Biological Modeling:** Uses Generalized Additive Models (GAMs) via `pygam` to map U-shaped physiological risk profiles, and evaluates non-linear thresholds in full models utilizing `SplineTransformer`.
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
├── README.md                # Project architecture and instructions
└── report.md                # Detailed answers to the assignment questions
```

## Setup & Running the Pipeline

**1. Install Requirements:**
Ensure you have the appropriate dependencies installed:
```bash
pip install pandas numpy scikit-learn matplotlib pygam joblib
```

**2. Process the Data:**
Run the preprocessing script to clean, impute, encode, and split the raw data into train and test sets.
```bash
python src/preprocess.py
```
This will output `X_train.csv`, `X_test.csv`, `y_train.csv`, `y_test.csv`, and `preprocessor.pkl` to the `data/processed` folder.

**3. Train the Models:**
Run the training script to fit all models, evaluate them, extract coefficients/importances, and generate plots.
```bash
python src/train.py
```
This will:
- Print metrics to the console.
- Save the trained models in the `./models/` directory.
- Output PNG plots and a full `evaluation_results.txt` to the `./reports/figures/` directory.