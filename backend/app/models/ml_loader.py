"""
ML Model Loader for NeuroAgent-PD.

Loads trained models, scalers, and label encoders from the artifacts directory.
Models are loaded once at import time and cached for reuse.

If a model file is missing or fails to load, the agent falls back
to rule-based scoring only.
"""

import os
import joblib
from pathlib import Path


ARTIFACTS_DIR = Path(__file__).resolve().parents[2] / "artifacts"


def _load_artifact(filename):
    """Safely load a joblib artifact. Returns None if not found or fails."""
    path = ARTIFACTS_DIR / filename

    if not path.exists():
        print(f"[MLLoader] Artifact not found: {path}")
        return None

    if path.stat().st_size == 0:
        print(f"[MLLoader] Artifact is empty: {path}")
        return None

    try:
        artifact = joblib.load(path)
        return artifact
    except Exception as e:
        print(f"[MLLoader] Failed to load {filename}: {e}")
        return None


# Clinical Agent
clinical_model = _load_artifact("clinical_model.pkl")
clinical_scaler = _load_artifact("clinical_scaler.pkl")
clinical_label_encoder = _load_artifact("clinical_label_encoder.pkl")

# Speech Agent
speech_model = _load_artifact("speech_model.pkl")
speech_scaler = _load_artifact("speech_scaler.pkl")
speech_label_encoder = _load_artifact("speech_label_encoder.pkl")

# Gait Agent
gait_model = _load_artifact("gait_model.pkl")
gait_scaler = _load_artifact("gait_scaler.pkl")
gait_label_encoder = _load_artifact("gait_label_encoder.pkl")

# Progression Agent
progression_model = _load_artifact("progression_model.pkl")
progression_scaler = _load_artifact("progression_scaler.pkl")
progression_label_encoder = _load_artifact("progression_label_encoder.pkl")


def is_clinical_model_available():
    return all([clinical_model, clinical_scaler, clinical_label_encoder])


def is_speech_model_available():
    return all([speech_model, speech_scaler, speech_label_encoder])


def is_gait_model_available():
    return all([gait_model, gait_scaler, gait_label_encoder])


def is_progression_model_available():
    return all([progression_model, progression_scaler, progression_label_encoder])
