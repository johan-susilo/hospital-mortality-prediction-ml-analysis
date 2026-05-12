# Statistical Methods for Bioinformatics - Assignment Report

This report evaluates and answers the 5 assignment questions regarding predicting in-hospital mortality using the `HOSP_ADMIT` dataset. The analysis uses models built with an MLOps pipeline designed to handle extreme class imbalances and high-dimensional, sparse data typical in clinical settings.

## 1. Study and describe the predictor variables.

**Variable Analysis & Dimensionality:**
The dataset contains 1700 patient entries and 84 predictor variables (plus the target variable `OUTCOME`). The predictors include demographic information (e.g., age, sex), comprehensive cardiac history (e.g., angina, prior MIs), presence of arrhythmias and conduction blocks, clinical measurements at admission (e.g., systolic and diastolic blood pressure, liver enzymes, leukocyte count), and administration of fibrinolytic therapies.

**Challenges for Predictions:**
1. **Dimensionality & Sparsity:** Many features are derived from categorical history or highly specific ECG indicators (e.g., `np01`, `n_p_ecg_p_01`). When applying one-hot encoding or using these sparse indicators, the feature space grows, breaching the Events-Per-Variable (EPV) rule for traditional models. This leads to quasi-complete separation and overfitting in standard unregularized regressions.
2. **Class Imbalance:** The baseline survival rate is 84%, meaning we have severe class imbalance (16% mortality). Accuracy is a flawed metric here, necessitating precision-recall evaluation.
3. **Missingness:** Missing clinical measurements or missing onset times (`TIME_B_S`) often occur Not-At-Random (MNAR). Missing indicators are critical to handle structural dropouts.

## 2. Fit and compare an appropriate unregularized regression model, LASSO and ridge regression, optimizing each model as well as you can.

To properly evaluate these models given the 84/16 class imbalance, standard baseline accuracy is discarded. We use Precision-Recall AUC (PR-AUC) and the Brier Score.
* **Baseline Performance:** A naive dummy classifier (predicting survival based purely on the prior distribution) yields a Test PR-AUC of **0.159**.

**Model Comparison (Linear versions):**
* **Unregularized Logistic Regression:** Test PR-AUC: 0.541 (Drop from train: 0.243)
* **LASSO (L1) Logistic (Best C=1.0):** Test PR-AUC: 0.540 (Drop from train: 0.215)
* **Ridge (L2) Logistic (Best C=1.0):** Test PR-AUC: 0.545 (Drop from train: 0.212)

**Evaluation & Over-learning Evidence:**
* All models significantly outperform the baseline of 0.159.
* We see clear evidence of over-learning, especially in the Unregularized model, which drops 0.243 points in PR-AUC from Training to Test sets.
* Regularization (LASSO and Ridge) successfully reduces this generalization gap (drop ~0.21), with Ridge performing slightly better on the test set.
* As hypothesized, not all 84+ variables carry sufficient independent information. The unregularized model is essentially memorizing noise in sparse ECG/history indicators, whereas LASSO enforces sparsity and Ridge controls magnitude, mitigating the curse of dimensionality.

## 3. Is there evidence for non-linear effects among the continuous predictors?

Yes. We utilized Generalized Additive Models (GAMs) and Partial Dependence Plots to map the non-linear risks. We systematically evaluated this by expanding continuous variables using a SplineTransformer (degree 3, 4 knots) across our full regression models.

**Comparing Performance (Linear vs Non-Linear Splines):**
* **Unregularized (Linear):** Test PR-AUC = 0.541  |  **Unregularized (Splines):** Test PR-AUC = 0.558
* **LASSO (Linear):** Test PR-AUC = 0.540  |  **LASSO (Splines):** Test PR-AUC = 0.560
* **Ridge (Linear):** Test PR-AUC = 0.545  |  **Ridge (Splines):** Test PR-AUC = 0.575

**Findings:**
Adding non-linear spline terms **improved prediction performance across all three models**. Ridge Regression with Splines achieved the best PR-AUC of 0.575. The physiological reason is clear: biological metrics like blood pressure often possess a "Goldilocks zone" (U-shaped risk profile), where both extremely high and extremely low values increase mortality risk. Linear coefficients fail to capture these non-monotonic relationships.

## 4. Fit a random forest model and compare its performance to your best model.

We fit a Random Forest model optimized via RandomizedSearchCV.
* **Random Forest Test PR-AUC:** 0.649
* **Best Linear/Spline Model (Ridge with Splines):** Test PR-AUC: 0.575

The Random Forest substantially outperforms the best regularized regression model. This implies that complex, multiplicative interaction effects (e.g., Age × Leukocytes × Hypotension) are vital to accurate mortality prediction, which linear and simple additive models fail to capture.

**Variable Importance Comparison (Blood Pressure):**
* **Regularized Coefficients:**
  * `S_AD_ORIT` (Systolic): LASSO = -0.3800, Ridge = -0.3419
  * `D_AD_ORIT` (Diastolic): LASSO = -0.1431, Ridge = -0.2208
* **Random Forest Feature Importance:**
  * Systolic BP (`S_AD_ORIT`) dominates and is the #1 most important variable.
  * Diastolic BP (`D_AD_ORIT`) is also highly ranked in the top 15.

**Discrepancy Explained:**
Both approaches agree that blood pressure is critical. However, in the linear models, they both appear as moderate negative coefficients, implying "higher blood pressure is linearly better". In reality, blood pressure is profoundly non-linear. The Random Forest captures the severe risk at the lower bounds (cardiogenic shock) and upper bounds (hypertensive crisis) without assuming a straight line, leading it to prioritize `S_AD_ORIT` as the undisputed most critical feature, whereas linear models dilute its importance because a straight line poorly fits a U-shaped clinical risk.

## 5. Summarize your findings.

**Most Important Predictors:**
Based on the Random Forest Gini Importance, the most critical predictors of in-hospital mortality are:
1. **Systolic Blood Pressure (S_AD_ORIT)**
2. **Leukocytes / WBC Count (L_BLOOD)**
3. **Patient Age (AGE)**
4. **Time to Admission (TIME_B_S)**
5. **Diastolic Blood Pressure (D_AD_ORIT)**
6. Liver Enzymes (ALT_BLOOD, AST_BLOOD) and Inflammation (ROE).

**Limitations and Caveats:**
1. **Data Missingness Mechanism:** While median imputation and missingness indicators are effective, if data is missing for socio-economic reasons or unrecorded due to extreme medical emergencies (e.g., patient died before blood was drawn), the models might be learning "administrative dropouts" rather than direct biological signals.
2. **Interpretability vs. Performance:** The Random Forest heavily outperforms standard regressions but acts as a black box. In a clinical setting, deploying an opaque model is dangerous. While GAMs provide a middle ground (interpretable non-linear effects), they do not capture the deep interaction terms mapped by the Forest.
3. **Imbalance Metrics:** Although PR-AUC metrics correctly evaluate the minority class (death), 0.649 still leaves considerable room for false positives/negatives in real-world triage, necessitating conservative probability thresholds.
