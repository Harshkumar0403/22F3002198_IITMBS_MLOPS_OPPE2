import requests
import random
import time
import csv
import statistics

# =====================================================
# CONFIG
# =====================================================
API_URL = "http://34.174.36.236/predict"   # your service IP
OUTPUT_CSV = "result.csv"
NUM_SAMPLES = 100

# =====================================================
# HELPERS
# =====================================================
def random_row(i):
    """
    Generate one random heart-disease-style sample
    """
    gender_str = random.choice(["male", "female"])
    gender_encoded = 1 if gender_str == "male" else 0

    return {
        "sno": i,
        "age": random.randint(29, 77),
        "gender": gender_encoded,     # IMPORTANT: send INT
        "cp": random.randint(0, 3),
        "trestbps": random.randint(90, 200),
        "chol": random.randint(120, 400),
        "fbs": random.randint(0, 1),
        "restecg": random.randint(0, 2),
        "thalach": random.randint(70, 210),
        "exang": random.randint(0, 1),
        "oldpeak": round(random.uniform(0.0, 6.0), 1),
        "slope": random.randint(0, 2),
        "ca": random.randint(0, 3),
        "thal": random.randint(0, 2),
        "_gender_str": gender_str      # keep for logging only
    }

# =====================================================
# MAIN TEST LOOP
# =====================================================
results = []
latencies = []

for i in range(NUM_SAMPLES):
    row = random_row(i)

    payload = {k: v for k, v in row.items() if not k.startswith("_")}

    start = time.time()
    try:
        response = requests.post(API_URL, json=payload, timeout=5)
        latency = time.time() - start

        if response.status_code == 200:
            resp = response.json()
            prediction = resp.get("prediction")
            probability = resp.get("probability")
            error = ""
        else:
            prediction = ""
            probability = ""
            error = response.text

    except Exception as e:
        latency = time.time() - start
        prediction = ""
        probability = ""
        error = str(e)

    latencies.append(latency)

    results.append({
        "sno": row["sno"],
        "age": row["age"],
        "gender": row["_gender_str"],   # human-readable
        "cp": row["cp"],
        "trestbps": row["trestbps"],
        "chol": row["chol"],
        "fbs": row["fbs"],
        "restecg": row["restecg"],
        "thalach": row["thalach"],
        "exang": row["exang"],
        "oldpeak": row["oldpeak"],
        "slope": row["slope"],
        "ca": row["ca"],
        "thal": row["thal"],
        "prediction": prediction,
        "probability": probability,
        "latency_sec": latency,
        "error": error
    })

# =====================================================
# WRITE CSV
# =====================================================
with open(OUTPUT_CSV, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=results[0].keys())
    writer.writeheader()
    writer.writerows(results)

# =====================================================
# LATENCY STATS
# =====================================================
print("✅ Per-sample prediction test completed")
print(f"Total samples: {NUM_SAMPLES}")
print(f"Min latency (s): {min(latencies):.4f}")
print(f"Max latency (s): {max(latencies):.4f}")
print(f"Avg latency (s): {statistics.mean(latencies):.4f}")
print(f"Results saved to: {OUTPUT_CSV}")

