import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from sklearn.dummy import DummyClassifier

from pygam import LogisticGAM, s

def evaluate_and_print(name, model, X_train, y_train, X_test, y_test, log_func):
    """Helper function to calculate metrics and log them both to console and file."""
    try:
        train_probs = model.predict_proba(X_train)[:, 1]
        test_probs = model.predict_proba(X_test)[:, 1]
    except:
        # for pygam or models that use predict_proba differently
        train_probs = model.predict_proba(X_train)
        test_probs = model.predict_proba(X_test)
        
    train_pr = average_precision_score(y_train, train_probs)
    test_pr = average_precision_score(y_test, test_probs)
    
    log_func(f"[{name}]")
    log_func(f"  PR-AUC     | Train: {train_pr:.3f} | Test: {test_pr:.3f} | Drop: {(train_pr - test_pr):.3f}")
    log_func(f"  ROC-AUC    | Train: {roc_auc_score(y_train, train_probs):.3f} | Test: {roc_auc_score(y_test, test_probs):.3f}")
    log_func(f"  Brier Loss | Train: {brier_score_loss(y_train, train_probs):.3f} | Test: {brier_score_loss(y_test, test_probs):.3f}\n")

def run_full_pipeline(data_dir: str, output_dir: str) -> None:
    data_path = Path(data_dir)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    # setup the text file to save all printed information
    log_file_path = out_path / "evaluation_results.txt"
    
    # overwrite the file fresh on each run
    with open(log_file_path, "w", encoding="utf-8") as f:
        f.write("=== PIPELINE EVALUATION REPORT ===\n\n")

    def log(message=""):
        """Prints to console AND saves to the text file."""
        print(message)
        with open(log_file_path, "a", encoding="utf-8") as f:
            f.write(str(message) + "\n")
            
    # 1. load the pristine data
    X_train = pd.read_csv(data_path / "X_train.csv")
    y_train = pd.read_csv(data_path / "y_train.csv").values.ravel()
    X_test = pd.read_csv(data_path / "X_test.csv")
    y_test = pd.read_csv(data_path / "y_test.csv").values.ravel()

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    log("="*50)
    log(" QUESTION 2: REGULARIZED VS UNREGULARIZED MODELS")
    log("="*50)
    
    # baseline
    dummy = DummyClassifier(strategy="prior").fit(X_train, y_train)
    log(f"[Baseline (Predict All Survive)] Test PR-AUC: {average_precision_score(y_test, dummy.predict_proba(X_test)[:, 1]):.3f}\n")

    # unregularized
    unreg_model = LogisticRegression(penalty=None, solver='lbfgs', max_iter=5000)
    unreg_model.fit(X_train, y_train)
    evaluate_and_print("Unregularized Logistic", unreg_model, X_train, y_train, X_test, y_test, log)

    # LASSO
    lasso_base = LogisticRegression(penalty='l1', solver='saga', max_iter=5000, random_state=42)
    lasso_grid = GridSearchCV(lasso_base, {'C': [0.001, 0.01, 0.1, 1.0, 10.0]}, cv=skf, scoring='average_precision', n_jobs=-1)
    lasso_grid.fit(X_train, y_train)
    log(f"Best LASSO Parameter: {lasso_grid.best_params_}")
    evaluate_and_print("LASSO (L1) Logistic", lasso_grid.best_estimator_, X_train, y_train, X_test, y_test, log)

    # Ridge
    ridge_base = LogisticRegression(penalty='l2', solver='lbfgs', max_iter=5000, random_state=42)
    ridge_grid = GridSearchCV(ridge_base, {'C': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]}, cv=skf, scoring='average_precision', n_jobs=-1)
    ridge_grid.fit(X_train, y_train)
    log(f"Best Ridge Parameter: {ridge_grid.best_params_}")
    evaluate_and_print("Ridge (L2) Logistic", ridge_grid.best_estimator_, X_train, y_train, X_test, y_test, log)


    log("="*50)
    log(" QUESTION 3: NON-LINEAR EFFECTS (GAMs)")
    log("="*50)
    
    continuous_cols = ['AGE', 'S_AD_ORIT', 'D_AD_ORIT', 'ALT_BLOOD', 'AST_BLOOD', 'L_BLOOD', 'ROE', 'TIME_B_S']
    X_cont = X_train.iloc[:, :8].values 
    
    gam = LogisticGAM(s(0) + s(1) + s(2) + s(3) + s(4) + s(5) + s(6) + s(7))
    gam.gridsearch(X_cont, y_train, progress=False)
    
    fig, axs = plt.subplots(2, 4, figsize=(20, 10))
    axs = axs.flatten()
    for i, feature in enumerate(continuous_cols):
        XX = gam.generate_X_grid(term=i)
        pdep, confi = gam.partial_dependence(term=i, X=XX, width=0.95)
        axs[i].plot(XX[:, i], pdep, color='darkred')
        axs[i].plot(XX[:, i], confi, c='r', ls='--', alpha=0.5)
        axs[i].set_title(f'Non-Linear Effect: {feature}')
        axs[i].set_xlabel(f'Scaled {feature}')
        axs[i].set_ylabel('Log Odds of Mortality')
        axs[i].axhline(0, color='black', linestyle=':', alpha=0.5)

    plt.tight_layout()
    gam_plot_path = out_path / 'gam_partial_dependence_plots.png'
    plt.savefig(gam_plot_path, dpi=300)
    log(f"-> GAM Non-Linearity Plots saved to: {gam_plot_path}\n")


    log("="*50)
    log(" QUESTION 4: TREE-BASED MODELS & INTERACTIONS")
    log("="*50)

    rf_base = RandomForestClassifier(class_weight='balanced', random_state=42)
    param_dist = {
        'n_estimators': [200, 500],
        'max_depth': [5, 10, 15, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2']
    }

    rf_search = RandomizedSearchCV(rf_base, param_distributions=param_dist, n_iter=20, cv=skf, scoring='average_precision', n_jobs=-1, random_state=42)
    rf_search.fit(X_train, y_train)
    best_rf = rf_search.best_estimator_
    
    log(f"Best RF Parameters: {rf_search.best_params_}")
    evaluate_and_print("Random Forest Classifier", best_rf, X_train, y_train, X_test, y_test, log)

    # plot translated feature importance
    CLINICAL_MAP = {
        'S_AD_ORIT': 'Systolic Blood Pressure', 'D_AD_ORIT': 'Diastolic Blood Pressure',
        'L_BLOOD': 'Leukocytes (WBC Count)', 'ALT_BLOOD': 'ALT (Liver Enzyme)',
        'AST_BLOOD': 'AST (Liver Enzyme)', 'TIME_B_S': 'Time to Admission (Hours)',
        'ROE': 'ESR (Inflammation)', 'AGE': 'Patient Age',
        'n_r_ecg_p_04': 'Frequent PVCs', 'n_r_ecg_p_05': 'Paroxysmal Atrial Fib.',
        'n_r_ecg_p_06': 'Persistent Atrial Fib.', 'n_r_ecg_p_08': 'Paroxysmal SVT',
        'n_r_ecg_p_09': 'Paroxysmal Ventricular Tachy.', 'n_r_ecg_p_10': 'Ventricular Fibrillation',
        'n_p_ecg_p_01': 'Sinoatrial Block', 'n_p_ecg_p_03': '1st-Degree AV Block',
        'n_p_ecg_p_06': '3rd-Degree AV Block', 'ant_im': 'Anterior MI',
        'lat_im': 'Lateral MI', 'inf_im': 'Inferior MI', 'post_im': 'Posterior MI'
    }
    
    importances = best_rf.feature_importances_
    indices = np.argsort(importances)[::-1][:15] 
    
    def translate_feature(name):
        for key in CLINICAL_MAP.keys():
            if name.startswith(key):
                return CLINICAL_MAP[key]
        return name

    feature_names = X_train.columns
    readable_labels = [translate_feature(feature_names[i]) for i in indices]
    
    plt.figure(figsize=(12, 7))
    plt.title("Random Forest: Top 15 Predictive Features", fontsize=14, pad=15)
    plt.bar(range(15), importances[indices], align="center", color='#1f77b4', edgecolor='black')
    plt.xticks(range(15), readable_labels, rotation=45, ha='right', fontsize=10)
    plt.xlim([-1, 15])
    plt.ylabel("Gini Importance Score", fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    rf_plot_path = out_path / 'rf_feature_importance.png'
    plt.savefig(rf_plot_path, dpi=300, bbox_inches='tight')
    log(f"-> Feature Importance Plot saved to: {rf_plot_path}")

    # save all models
    model_path = Path("./models")
    model_path.mkdir(exist_ok=True)
    joblib.dump(unreg_model, model_path / "model_unreg.pkl")
    joblib.dump(lasso_grid.best_estimator_, model_path / "model_lasso.pkl")
    joblib.dump(ridge_grid.best_estimator_, model_path / "model_ridge.pkl")
    joblib.dump(best_rf, model_path / "model_rf.pkl")
    
    log("\n✅ All models saved successfully to ./models")
    log(f"✅ Text summary saved to: {log_file_path}")

if __name__ == "__main__":
    run_full_pipeline(data_dir="./data/processed", output_dir="./reports/figures")