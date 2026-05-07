import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from typing import Tuple

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split

# feature groups
TARGET_COL = 'OUTCOME'

# continuous variables (Need median imputation + scaling + missingness indicator)
NUMERIC_COLS = ['AGE', 'S_AD_ORIT', 'D_AD_ORIT', 'ALT_BLOOD', 'AST_BLOOD', 'L_BLOOD', 'ROE', 'TIME_B_S']

# categorical/nominal variables (Need frequent imputation + One-Hot Encoding)
CATEGORICAL_COLS = ['INF_ANAM', 'STENOK_AN', 'FK_STENOK', 'ZSN_A', 'IM_PG_P']

# binary/pass-through variables (Already 0/1, just need missing values handled)
BINARY_COLS = [
    # core
    "SEX", "SIM_GIPERT", "NITR_S",

    # clinical history
    "IBS_POST", "GB", "DLIT_AG",

    # ECG rhythm / abnormalities
    "nr11","nr01","nr02","nr03","nr04","nr07","nr08",
    "np01","np04","np05","np07","np08","np09","np10",

    # endocrine / comorbidities
    "endocr_01","endocr_02","endocr_03",
    "zab_leg_01","zab_leg_02","zab_leg_03",
    "zab_leg_04","zab_leg_06",

    # post-event complications
    "O_L_POST","K_SH_POST","MP_TP_POST",
    "SVT_POST","GT_POST","FIB_G_POST",

    # infarct localization
    "ant_im","lat_im","inf_im","post_im",

    # ECG rhythm (extended)
    "ritm_ecg_p_01","ritm_ecg_p_02","ritm_ecg_p_04",
    "ritm_ecg_p_06","ritm_ecg_p_07","ritm_ecg_p_08",

    # ECG negative/positive response groups
    "n_r_ecg_p_01","n_r_ecg_p_02","n_r_ecg_p_03",
    "n_r_ecg_p_04","n_r_ecg_p_05","n_r_ecg_p_06",
    "n_r_ecg_p_08","n_r_ecg_p_09","n_r_ecg_p_10",

    "n_p_ecg_p_01","n_p_ecg_p_03","n_p_ecg_p_04",
    "n_p_ecg_p_05","n_p_ecg_p_06","n_p_ecg_p_07",
    "n_p_ecg_p_08","n_p_ecg_p_09","n_p_ecg_p_10",
    "n_p_ecg_p_11","n_p_ecg_p_12",

    # therapy
    "fibr_ter_01","fibr_ter_02","fibr_ter_03",
    "fibr_ter_05","fibr_ter_06","fibr_ter_07","fibr_ter_08"
]

def build_preprocessor() -> ColumnTransformer:
  """
  constructs the preprocessing pipeline for clinical data.
  """
  
  numerical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
    ("scaler", StandardScaler())
  ])
  
  categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore", drop="first"))
  ])
  
  binary_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent"))
  ])
  
  preprocessor = ColumnTransformer([
    ("num", numerical_pipeline, NUMERIC_COLS),
    ("cat", categorical_pipeline, CATEGORICAL_COLS),
    ("bin", binary_pipeline, BINARY_COLS)
  ], verbose_feature_names_out=False)
  
  return preprocessor

def process_data(input_path: str, output_dir: str) -> None:
    """
    Loads raw data, splits, fits transformations, and saves artifacts.
    """
    # 1. Load data
    df = pd.read_csv(input_path)
    
    # 2. Separate X and y
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]
    
    # 3. Train/Test Split (YOUR TASK: Implement Stratified splitting to handle the 84/16 class imbalance)
    X_train, X_test, y_train, y_test = train_test_split(
      X, 
      y,
      test_size=0.2,
      random_state=42,
      stratify=y
      # split the data so that the train and test sets have the same 
      # class proportions as the original target variable y.
      # because the category 0 and 1 is imbalanced
      ) 
    
    # 4. Build and Fit Preprocessor
    preprocessor = build_preprocessor()
    
    # YOUR TASK:
    
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    # 5. Fit and transform X_train
    # 6. Transform (DO NOT FIT) X_test
    
    # 7. Save outputs
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    # Save transformed dataframes and the pipeline artifact (.pkl)
    joblib.dump(preprocessor, out_path/"preprocessor.pkl")
    
    # Grab the feature names from the pipeline
    feature_names = preprocessor.get_feature_names_out()
    
    # Save transformed dataframes with the correct column names
    joblib.dump(preprocessor, out_path/"preprocessor.pkl")
    
    pd.DataFrame(X_train_processed, columns=feature_names).to_csv(out_path / "X_train.csv", index=False)
    pd.DataFrame(X_test_processed, columns=feature_names).to_csv(out_path / "X_test.csv", index=False)
    y_train.to_csv(out_path / "y_train.csv", index=False)
    y_test.to_csv(out_path / "y_test.csv", index=False)
    
if __name__ == "__main__":
  process_data(
    input_path="./data/raw/HOSP_ADMIT.csv",
    output_dir="./data/processed"
  )