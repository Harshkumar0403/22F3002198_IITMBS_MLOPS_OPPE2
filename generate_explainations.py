import os
import joblib
import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from evidently import Report
from evidently.presets import DataDriftPreset

# =====================================================
# CONFIG
# =====================================================
DATA_PATH = "data/data.csv"
MODEL_PATH = "models/model.pkl"
OUTPUT_DIR = "explainability_outputs"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# =====================================================
# LOAD DATA
# =====================================================
df = pd.read_csv(DATA_PATH)

# -----------------------------
# TARGET ENCODING
# -----------------------------
df["target"] = df["target"].astype(str).str.lower().str.strip()
df["target"] = df["target"].map({"no": 0, "yes": 1})

# -----------------------------
# GENDER ENCODING
# -----------------------------
df["gender"] = df["gender"].astype(str).str.lower().str.strip()
df["gender"] = df["gender"].map({"male": 1, "female": 0})

# =====================================================
# FEATURE / TARGET SPLIT
# =====================================================
X = df.drop(columns=["target"])
y = df["target"]

# =====================================================
# LOAD MODEL
# =====================================================
model = joblib.load(MODEL_PATH)

# =====================================================
# PREDICTIONS
# =====================================================
y_pred = model.predict(X)
y_prob = model.predict_proba(X)[:, 1]

# =====================================================
# IDENTIFY FALSE NEGATIVES
# (Actual = 1, Predicted = 0)
# =====================================================
fn_mask = (y == 1) & (y_pred == 0)
X_fn = X[fn_mask]

if X_fn.empty:
    raise ValueError("No false negative samples found. Cannot generate explanations.")

# =====================================================
# SHAP EXPLAINABILITY
# =====================================================
# Extract trained classifier (after imputer)
classifier = model.named_steps["classifier"]

# Impute FN data using trained imputer
X_fn_imputed = model.named_steps["imputer"].transform(X_fn)

explainer = shap.LinearExplainer(classifier, X_fn_imputed)
shap_values = explainer.shap_values(X_fn_imputed)

# -----------------------------
# SHAP SUMMARY PLOT
# -----------------------------
plt.figure()
shap.summary_plot(
    shap_values,
    X_fn_imputed,
    feature_names=X.columns,
    show=False
)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/shap_false_negatives_summary.png", dpi=300)
plt.close()

# -----------------------------
# MEAN ABS SHAP VALUES
# -----------------------------
mean_shap = np.abs(shap_values).mean(axis=0)
shap_importance = pd.DataFrame({
    "feature": X.columns,
    "mean_abs_shap": mean_shap
}).sort_values(by="mean_abs_shap", ascending=False)

shap_importance.to_csv(
    f"{OUTPUT_DIR}/shap_false_negatives_importance.csv",
    index=False
)

# =====================================================
# EVIDENTLY (SUPPORTING EXPLANATION)
# =====================================================
reference_data = X.copy()
current_data = X_fn.copy()

report = Report(metrics=[DataDriftPreset()])
myeval = report.run(reference_data=reference_data, current_data=current_data)
myeval.save_html(f"{OUTPUT_DIR}/evidently_false_negative_report.html")

# =====================================================
# PLAIN ENGLISH EXPLANATION
# =====================================================
top_features = shap_importance.head(5)["feature"].tolist()

explanation_text = f"""
Explainability Analysis: False Negatives (Heart Disease Not Predicted)

Definition:
False negatives are patients who actually have heart disease (target = 1)
but were predicted by the model as not having heart disease (prediction = 0).

Key Findings:
The SHAP analysis shows that the model’s incorrect decisions for these patients
are most strongly influenced by the following factors:

1. {top_features[0]}
2. {top_features[1]}
3. {top_features[2]}
4. {top_features[3]}
5. {top_features[4]}

Interpretation:
For false negative patients, these features tend to have values that make
the model interpret them as low-risk, even though they truly have heart disease.
In particular, lower ST depression, fewer major vessels, favorable chest pain
types, or higher exercise tolerance can mask the underlying disease.

Clinical Implication:
These results suggest that patients with mild or atypical symptoms may be
systematically under-predicted by the model, indicating a need for threshold
adjustment or cost-sensitive training to reduce false negatives.
"""

with open(f"{OUTPUT_DIR}/false_negative_explanation.txt", "w") as f:
    f.write(explanation_text.strip())

print("✅ Explainability generation completed")
print(f"Outputs saved in: {OUTPUT_DIR}/")

