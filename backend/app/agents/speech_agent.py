import numpy as np

from app.agents.risk_utils import clamp_score, format_prediction, get_risk_level
from app.models.ml_loader import (
    is_speech_model_available,
    speech_label_encoder,
    speech_model,
    speech_scaler,
)


def _rule_based_score(speech_data):
    """Original rule-based speech risk scoring."""

    jitter = speech_data.jitter
    shimmer = speech_data.shimmer
    hnr = speech_data.hnr
    pitch_variation = speech_data.pitch_variation

    jitter_score = min(1.0, jitter / 0.04)
    shimmer_score = min(1.0, shimmer / 0.07)
    hnr_score = max(0.0, min(1.0, (25 - hnr) / 25))
    pitch_score = min(1.0, pitch_variation / 0.5)

    risk_score = (
        jitter_score * 0.3
        + shimmer_score * 0.3
        + hnr_score * 0.2
        + pitch_score * 0.2
    )

    return clamp_score(risk_score)


def _ml_prediction(speech_data):
    """
    ML-based speech risk prediction.
    Returns (predicted_class, confidence, probabilities) or None if unavailable.
    """

    if not is_speech_model_available():
        return None

    try:
        features = np.array([[
            speech_data.jitter,
            speech_data.shimmer,
            speech_data.hnr,
            speech_data.pitch_variation,
        ]])

        features_scaled = speech_scaler.transform(features)
        prediction_encoded = speech_model.predict(features_scaled)[0]
        probabilities = speech_model.predict_proba(features_scaled)[0]

        predicted_class = speech_label_encoder.inverse_transform([prediction_encoded])[0]
        confidence = float(np.max(probabilities))

        class_probs = {
            label: round(float(prob), 4)
            for label, prob in zip(speech_label_encoder.classes_, probabilities)
        }

        return {
            "predicted_class": predicted_class,
            "confidence": confidence,
            "class_probabilities": class_probs,
        }

    except Exception as e:
        print(f"[SpeechAgent] ML prediction failed: {e}")
        return None


def analyze_speech(speech_data):
    # Rule-based scoring
    risk_score = _rule_based_score(speech_data)
    risk_score = round(risk_score, 2)
    risk_level = get_risk_level(risk_score)

    if risk_level in ["high", "moderate"]:
        severity = "speech_abnormality"
    elif risk_level == "borderline":
        severity = "borderline_speech_change"
    else:
        severity = "normal_or_mild"

    # ML prediction
    ml_result = _ml_prediction(speech_data)

    result = {
        "agent_name": "speech_agent",
        "risk_score": risk_score,
        "risk_level": risk_level,
        "prediction": format_prediction("speech", risk_level),
        "severity": severity,
        "confidence": round(min(0.9, 0.5 + risk_score * 0.4), 2),
        "top_features": [
            "jitter",
            "shimmer",
            "hnr",
            "pitch_variation",
        ],
        "explanation": (
            f"Speech agent estimated a {risk_level} speech-related risk level. "
            "It analyzed jitter, shimmer, HNR, and pitch variation to estimate voice instability."
        ),
        "ml_available": ml_result is not None,
    }

    if ml_result:
        result["ml_prediction"] = ml_result["predicted_class"]
        result["ml_confidence"] = round(ml_result["confidence"], 2)
        result["ml_class_probabilities"] = ml_result["class_probabilities"]

    return result
