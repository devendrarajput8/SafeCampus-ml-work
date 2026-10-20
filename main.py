import os
import re
from pathlib import Path
from typing import List, Optional

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = Path(os.getenv("SAFECAMPUS_MODEL_DIR", BASE_DIR / "models"))
CATEGORY_MODEL_PATH = MODEL_DIR / "incident_category_model.joblib"
URGENCY_MODEL_PATH = MODEL_DIR / "urgency_model.joblib"

app = FastAPI(
    title="SafeCampus ML API",
    description=(
        "ML inference service for SafeCampus. Predictions are decision support only; "
        "this prototype must not delay or replace approved emergency procedures."
    ),
    version="1.0.0",
)

category_model = None
urgency_model = None

URGENT_PATTERNS = [
    r"unconscious", r"not responding", r"difficulty breathing",
    r"heavy smoke", r"flames", r"live electrical wire",
    r"threatening students", r"force entry"
]

class IncidentRequest(BaseModel):
    description: str = Field(..., min_length=3, max_length=5000)
    location_text: Optional[str] = Field(default=None, max_length=300)

class CategoryResponse(BaseModel):
    description: str
    predicted_category: str
    category_model_score: float
    model_type: str = "TF-IDF + Logistic Regression (prototype)"

class AssessmentResponse(BaseModel):
    description: str
    location_text: Optional[str] = None
    predicted_category: str
    category_model_score: float
    predicted_urgency: str
    urgency_model_score: float
    urgent_rule_triggered: bool
    matched_rules: List[str]
    recommended_action: str
    disclaimer: str

def load_models():
    global category_model, urgency_model
    if not CATEGORY_MODEL_PATH.exists() or not URGENCY_MODEL_PATH.exists():
        return False
    category_model = joblib.load(CATEGORY_MODEL_PATH)
    urgency_model = joblib.load(URGENCY_MODEL_PATH)
    return True

def urgent_rule_check(description: str):
    text = description.lower()
    matched = [pattern for pattern in URGENT_PATTERNS if re.search(pattern, text)]
    triggered = bool(matched)
    return {
        "urgent_rule_triggered": triggered,
        "matched_rules": matched,
        "recommended_action": (
            "Trigger the campus-approved urgent workflow immediately; do not wait for model inference."
            if triggered else
            "Continue the approved assessment workflow and human review."
        )
    }

@app.on_event("startup")
def startup_event():
    load_models()

@app.get("/")
def root():
    return {
        "service": "SafeCampus ML API",
        "docs": "/docs",
        "health": "/health"
    }

@app.get("/health")
def health():
    ready = category_model is not None and urgency_model is not None
    return {
        "status": "ok" if ready else "models_not_loaded",
        "category_model_loaded": category_model is not None,
        "urgency_model_loaded": urgency_model is not None,
        "model_directory": str(MODEL_DIR)
    }

def ensure_models_loaded():
    if category_model is None or urgency_model is None:
        if not load_models():
            raise HTTPException(
                status_code=503,
                detail=(
                    "Model files not found. Run the notebook first, then copy "
                    "incident_category_model.joblib and urgency_model.joblib into the models/ folder."
                )
            )

@app.post("/api/ml/classify", response_model=CategoryResponse)
def classify_incident(payload: IncidentRequest):
    ensure_models_loaded()
    text = payload.description.strip()
    if not text:
        raise HTTPException(status_code=422, detail="description cannot be blank")
    probabilities = category_model.predict_proba([text])[0]
    classes = category_model.named_steps["classifier"].classes_
    idx = int(np.argmax(probabilities))
    return {
        "description": text,
        "predicted_category": str(classes[idx]),
        "category_model_score": float(probabilities[idx])
    }

@app.post("/api/ml/assess", response_model=AssessmentResponse)
def assess_incident(payload: IncidentRequest):
    ensure_models_loaded()
    text = payload.description.strip()
    if not text:
        raise HTTPException(status_code=422, detail="description cannot be blank")

    cat_probs = category_model.predict_proba([text])[0]
    cat_classes = category_model.named_steps["classifier"].classes_
    cat_idx = int(np.argmax(cat_probs))

    urg_probs = urgency_model.predict_proba([text])[0]
    urg_classes = urgency_model.named_steps["classifier"].classes_
    urg_idx = int(np.argmax(urg_probs))

    rules = urgent_rule_check(text)
    return {
        "description": text,
        "location_text": payload.location_text,
        "predicted_category": str(cat_classes[cat_idx]),
        "category_model_score": float(cat_probs[cat_idx]),
        "predicted_urgency": str(urg_classes[urg_idx]),
        "urgency_model_score": float(urg_probs[urg_idx]),
        "urgent_rule_triggered": rules["urgent_rule_triggered"],
        "matched_rules": rules["matched_rules"],
        "recommended_action": rules["recommended_action"],
        "disclaimer": (
            "Prototype only. Model scores are not calibrated emergency probabilities. "
            "Use campus-approved rules and human oversight; never delay required alerts."
        )
    }
