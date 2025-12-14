import json
import joblib
import pandas as pd

from evidently import Report
from evidently.presets import DataDriftPreset

# =====================================================
# CONFIG
# =====================================================
TRAIN_DATA_PATH = "data/data.csv"
CURRENT_DATA_PATH = "data/data.csv"   # generated / inference data
MODEL_PATH = "models/model.pkl"

HTML_REPORT_PATH = "input_drift_report.html"
JSON_SUMMARY_PATH = "input_drift_summary.json"

# =====================================================
# LOAD DATA
# =====================================================
train_df = pd.read_csv(TRAIN_DATA_PATH)
current_df = pd.read_csv(CURRENT_DATA_PATH)

# =====================================================
# CONSISTENT PREPROCESSING (SAME AS TRAINING)
# =====================================================
def preprocess(df):
    df = df.copy()

    # Target encoding (drop later)
    df["target"] = df["target"].astype(str).str.lower().str.strip()
    df["target"] = df["target"].map({"no": 0, "yes": 1})

    # Gender encoding
    df["gender"] = df["gender"].astype(str).str.lower().str.strip()
    df["gender"] = df["gender"].map({"male": 1, "female": 0})

    return df

train_df = preprocess(train_df)
current_df = preprocess(current_df)

# =====================================================
# DROP TARGET (INPUT DRIFT ONLY)
# =====================================================
train_features = train_df.drop(columns=["target"])
current_features = current_df.drop(columns=["target"])

# =====================================================
# LOAD MODEL (for pipeline consistency)
# =====================================================
model = joblib.load(MODEL_PATH)

# (Model not strictly needed for input drift,
# but loaded to ensure same feature expectations)

# =====================================================
# EVIDENTLY DATA DRIFT REPORT
# =====================================================
report = Report(metrics=[DataDriftPreset()])

eval=report.run(
    reference_data=train_features,
    current_data=current_features
)

# Save HTML report
eval.save_html(HTML_REPORT_PATH)

# =====================================================
# EXTRACT SUMMARY METRICS (JSON)
# =====================================================
#drift_result = eval.load_dict()


print("✅ Input drift analysis completed")
print(f"HTML report saved to: {HTML_REPORT_PATH}")
#print(f"JSON summary saved to: {JSON_SUMMARY_PATH}")

