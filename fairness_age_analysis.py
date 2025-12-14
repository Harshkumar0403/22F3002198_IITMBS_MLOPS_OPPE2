import json
import joblib
import pandas as pd

from fairlearn.metrics import (
    MetricFrame,
    selection_rate,
    demographic_parity_difference,
    demographic_parity_ratio
)
from sklearn.metrics import accuracy_score

# =====================================================
# CONFIG
# =====================================================
DATA_PATH = "data/data.csv"
MODEL_PATH = "models/model.pkl"
OUTPUT_FILE = "fairness_age_results.json"

# =====================================================
# LOAD DATA
# =====================================================
df = pd.read_csv(DATA_PATH)

# =====================================================
# TARGET ENCODING (same as training)
# =====================================================
df["target"] = df["target"].astype(str).str.lower().str.strip()
df["target"] = df["target"].map({"no": 0, "yes": 1})

# =====================================================
# GENDER ENCODING (same as training)
# =====================================================
df["gender"] = df["gender"].astype(str).str.lower().str.strip()
df["gender"] = df["gender"].map({"male": 1, "female": 0})

# =====================================================
# AGE BUCKETING (20-YEAR BINS) – FAIRNESS ONLY
# =====================================================
def age_bucket(age):
    if pd.isna(age):
        return "unknown"
    lower = int(age // 20) * 20
    upper = lower + 19
    return f"{lower}-{upper}"

df["age_bucket"] = df["age"].apply(age_bucket)

# =====================================================
# FEATURE / TARGET SPLIT
# IMPORTANT: exclude age_bucket from model input
# =====================================================
X = df.drop(columns=["target", "age_bucket"])
y_true = df["target"]
sensitive_features = df["age_bucket"]

# =====================================================
# LOAD MODEL
# =====================================================
model = joblib.load(MODEL_PATH)

# =====================================================
# PREDICTIONS
# =====================================================
y_pred = model.predict(X)

# =====================================================
# FAIRNESS METRICS (FAIRLEARN)
# =====================================================
metric_frame = MetricFrame(
    metrics={
        "accuracy": accuracy_score,
        "selection_rate": selection_rate
    },
    y_true=y_true,
    y_pred=y_pred,
    sensitive_features=sensitive_features
)

# Group-wise metrics
group_metrics = metric_frame.by_group.to_dict()

# Overall fairness metrics
fairness_metrics = {
    "demographic_parity_difference": demographic_parity_difference(
        y_true=y_true,
        y_pred=y_pred,
        sensitive_features=sensitive_features
    ),
    "demographic_parity_ratio": demographic_parity_ratio(
        y_true=y_true,
        y_pred=y_pred,
        sensitive_features=sensitive_features
    )
}

# =====================================================
# SAVE RESULTS
# =====================================================
output = {
    "sensitive_attribute": "age",
    "age_bucket_size": "20 years",
    "group_metrics": group_metrics,
    "fairness_metrics": fairness_metrics
}

with open(OUTPUT_FILE, "w") as f:
    json.dump(output, f, indent=2)

print("✅ Fairness analysis completed successfully")
print(f"Results saved to: {OUTPUT_FILE}")

