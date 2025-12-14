import os
import json
import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

# =====================================================
# CONFIGURATION
# =====================================================
DATA_PATH = "data/data.csv"
MODEL_PATH = "models/model.pkl"
ARTIFACT_DIR = "mlflow_artifacts"

MLFLOW_TRACKING_URI = "http://136.115.41.149:8100"
EXPERIMENT_NAME = "heart_disease_logistic_regression"

RANDOM_STATE = 42
TEST_SIZE = 0.2

# =====================================================
# SETUP
# =====================================================
os.makedirs("models", exist_ok=True)
os.makedirs(ARTIFACT_DIR, exist_ok=True)

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(EXPERIMENT_NAME)

# =====================================================
# LOAD DATA
# =====================================================
df = pd.read_csv(DATA_PATH)

# =====================================================
# HANDLE TARGET LABEL (yes/no → 1/0)
# =====================================================
if "target" not in df.columns:
    raise ValueError("Expected column 'target' not found")

df["target"] = df["target"].astype(str).str.lower().str.strip()

allowed_targets = {"yes", "no"}
invalid_targets = set(df["target"].dropna().unique()) - allowed_targets
if invalid_targets:
    raise ValueError(f"Invalid target values found: {invalid_targets}")

df["target"] = df["target"].map({"no": 0, "yes": 1})

# =====================================================
# HANDLE GENDER (male/female → 1/0)
# =====================================================
if "gender" not in df.columns:
    raise ValueError("Expected column 'gender' not found")

df["gender"] = df["gender"].astype(str).str.lower().str.strip()

allowed_gender = {"male", "female"}
invalid_gender = set(df["gender"].dropna().unique()) - allowed_gender
if invalid_gender:
    raise ValueError(f"Invalid gender values found: {invalid_gender}")

df["gender"] = df["gender"].map({"male": 1, "female": 0})

# =====================================================
# FEATURE / TARGET SPLIT
# =====================================================
X = df.drop(columns=["target"])
y = df["target"]

x_train, x_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y
)

# =====================================================
# PIPELINE (IMPUTER + LOGISTIC REGRESSION)
# =====================================================
pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("classifier", LogisticRegression(max_iter=1000))
    ]
)

# =====================================================
# RANDOMIZED SEARCH SPACE
# =====================================================
log_reg_grid = {
    "classifier__C": np.logspace(-4, 4, 20),
    "classifier__solver": ["liblinear"]
}

rs_log_reg = RandomizedSearchCV(
    estimator=pipeline,
    param_distributions=log_reg_grid,
    cv=5,
    n_iter=20,
    scoring="roc_auc",
    verbose=True,
    n_jobs=-1,
    random_state=RANDOM_STATE,
    error_score="raise"
)

# =====================================================
# TRAINING + MLFLOW LOGGING
# =====================================================
with mlflow.start_run(run_name="log_reg_random_search"):

    # -----------------------------
    # LOG PREPROCESSING DETAILS
    # -----------------------------
    mlflow.log_param("test_size", TEST_SIZE)
    mlflow.log_param("random_state", RANDOM_STATE)
    mlflow.log_param("num_samples", X.shape[0])
    mlflow.log_param("num_features", X.shape[1])
    mlflow.log_param("gender_encoding", "male=1,female=0")
    mlflow.log_param("target_encoding", "no=0,yes=1")
    mlflow.log_param("imputation_strategy", "median")

    # -----------------------------
    # TRAIN
    # -----------------------------
    rs_log_reg.fit(x_train, y_train)

    best_model = rs_log_reg.best_estimator_

    # -----------------------------
    # EVALUATION
    # -----------------------------
    y_pred = best_model.predict(x_test)
    y_prob = best_model.predict_proba(x_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_prob)
    }

    for k, v in metrics.items():
        mlflow.log_metric(k, v)

    # -----------------------------
    # LOG BEST PARAMETERS
    # -----------------------------
    for param, value in rs_log_reg.best_params_.items():
        mlflow.log_param(f"best_{param}", value)

    # -----------------------------
    # SAVE CV RESULTS
    # -----------------------------
    cv_results_path = os.path.join(ARTIFACT_DIR, "cv_results.csv")
    pd.DataFrame(rs_log_reg.cv_results_).to_csv(cv_results_path, index=False)
    mlflow.log_artifact(cv_results_path, artifact_path="cv_results")

    # -----------------------------
    # SAVE CONFUSION MATRIX
    # -----------------------------
    cm = confusion_matrix(y_test, y_pred)
    cm_path = os.path.join(ARTIFACT_DIR, "confusion_matrix.json")
    with open(cm_path, "w") as f:
        json.dump(cm.tolist(), f)
    mlflow.log_artifact(cm_path, artifact_path="evaluation")

    # -----------------------------
    # SAVE CLASSIFICATION REPORT
    # -----------------------------
    report = classification_report(y_test, y_pred, output_dict=True)
    report_path = os.path.join(ARTIFACT_DIR, "classification_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    mlflow.log_artifact(report_path, artifact_path="evaluation")

    # -----------------------------
    # SAVE MODEL LOCALLY
    # -----------------------------
    joblib.dump(best_model, MODEL_PATH)

    # -----------------------------
    # LOG MODEL TO MLFLOW
    # -----------------------------
    mlflow.sklearn.log_model(
        best_model,
        artifact_path="model",
        registered_model_name="heart_disease_log_reg"
    )

    print("\n✅ Training completed successfully")
    print("Best Parameters:", rs_log_reg.best_params_)
    print("Metrics:", metrics)
    print(f"Model saved to: {MODEL_PATH}")

