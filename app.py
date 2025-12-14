import os
import sys
import json
import joblib
import logging
import numpy as np
import pandas as pd

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# OpenTelemetry (optional, enabled only if endpoint provided)
from opentelemetry import trace
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter

# =====================================================
# CONFIG
# =====================================================
MODEL_PATH = os.environ.get("MODEL_PATH", "models/model.pkl")
OTEL_OTLP_ENDPOINT = os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT", "").strip()

FEATURES = [
    "sno", "age", "gender", "cp", "trestbps", "chol", "fbs",
    "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal"
]

# =====================================================
# LOGGING (JSON structured logs)
# =====================================================
class JsonFormatter(logging.Formatter):
    def format(self, record):
        return json.dumps({
            "severity": record.levelname,
            "message": record.getMessage()
        })

logger = logging.getLogger("heart-disease-api")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(JsonFormatter())
logger.handlers.clear()
logger.addHandler(handler)

logging.getLogger("opentelemetry").setLevel(logging.WARNING)

# =====================================================
# OpenTelemetry setup (optional)
# =====================================================
tracer = None
otel_enabled = bool(OTEL_OTLP_ENDPOINT)

if otel_enabled:
    try:
        provider = TracerProvider()
        exporter = OTLPSpanExporter(endpoint=OTEL_OTLP_ENDPOINT)
        provider.add_span_processor(BatchSpanProcessor(exporter))
        trace.set_tracer_provider(provider)
        tracer = trace.get_tracer(__name__)
        logger.info("OpenTelemetry tracing enabled")
    except Exception as e:
        logger.warning(f"Failed to init OpenTelemetry: {e}")
        tracer = trace.get_tracer(__name__)
else:
    tracer = trace.get_tracer(__name__)

# =====================================================
# FastAPI app
# =====================================================
app = FastAPI(title="Heart Disease Prediction API")

if otel_enabled:
    FastAPIInstrumentor.instrument_app(app)

# =====================================================
# Load model at startup
# =====================================================
model = None

@app.on_event("startup")
def startup():
    global model
    try:
        model = joblib.load(MODEL_PATH)
        logger.info(f"Model loaded successfully from {MODEL_PATH}")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        model = None

# =====================================================
# Request / Response Schemas
# =====================================================
class PredictionRequest(BaseModel):
    sno: int
    age: float
    gender: int = Field(..., description="male=1, female=0")
    cp: int
    trestbps: float
    chol: float
    fbs: int
    restecg: int
    thalach: float
    exang: int
    oldpeak: float
    slope: int
    ca: int
    thal: int

class PredictionResponse(BaseModel):
    has_heart_disease: int
    probability: float

# =====================================================
# Health endpoints
# =====================================================
@app.get("/live_check")
def live():
    return {"status": "alive"}

@app.get("/ready_check")
def ready():
    return {"status": "ready" if model is not None else "not ready"}

# =====================================================
# Prediction endpoint
# =====================================================
@app.post("/predict", response_model=PredictionResponse)
def predict(req: PredictionRequest):

    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # Prepare input in correct feature order
    try:
        input_dict = req.dict()
        X = np.array([[input_dict[f] for f in FEATURES]], dtype=float)
    except Exception as e:
        logger.error(f"Input preparation failed: {e}")
        raise HTTPException(status_code=400, detail="Invalid input")

    # Inference with tracing
    with tracer.start_as_current_span("model_inference"):
        try:
            prob = float(model.predict_proba(X)[0][1])
            pred = 1 if prob >= 0.5 else 0
        except Exception as e:
            logger.error(f"Inference failed: {e}")
            raise HTTPException(status_code=500, detail="Model inference failed")

    logger.info(f"prediction made: pred={pred}, prob={prob:.4f}")

    return PredictionResponse(
        has_heart_disease=pred,
        probability=prob
    )

# =====================================================
# Local run (for VM testing)
# =====================================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000)

