import numpy as np

from app.agents.risk_utils import clamp_score, format_prediction, get_risk_level
from app.models.ml_loader import (
    gait_label_encoder,
    gait_model,
    gait_scaler,
    is_gait_model_available,
)


def _rule_based_score(gait_data):
    """Original rule-based gait risk scoring."""

    walking_speed = gait_data.walking_speed
    stride_variability = gait_data.stride_variability
    freezing_index = gait_data.freezing_index
    balance_score = gait_data.balance_score

    speed_risk = max(0.0, min(1.0, (1.4 - walking_speed) / 1.4))
    stride_risk = min(1.0, stride_variability / 0.6)
    freezing_risk = min(1.0, freezing_index / 0.8)
    balance_risk = max(0.0, min(1.0, 1 - balance_score))

    risk_score = (
        speed_risk * 0.25
        + stride_risk * 0.25
        + freezing_risk * 0.3
        + balance_risk * 0.2
    )

    return clamp_score(risk_score)


def _ml_prediction(gait_data):
    """
    ML-based gait risk prediction.
    Returns (predicted_class, confidence, probabilities) or None if unavailable.
    """

    if not is_gait_model_available():
        return None

    try:
        features = np.array([[
            gait_data.walking_speed,
            gait_data.stride_variability,
            gait_data.freezing_index,
            gait_data.balance_score,
        ]])

        features_scaled = gait_scaler.transform(features)
        prediction_encoded = gait_model.predict(features_scaled)[0]
        probabilities = gait_model.predict_proba(features_scaled)[0]

        predicted_class = gait_label_encoder.inverse_transform([prediction_encoded])[0]
        confidence = float(np.max(probabilities))

        class_probs = {
            label: round(float(prob), 4)
            for label, prob in zip(gait_label_encoder.classes_, probabilities)
        }

        return {
            "predicted_class": predicted_class,
            "confidence": confidence,
            "class_probabilities": class_probs,
        }

    except Exception as e:
        print(f"[GaitAgent] ML prediction failed: {e}")
        return None


def analyze_gait(gait_data):
    # Rule-based scoring
    risk_score = _rule_based_score(gait_data)
    risk_score = round(risk_score, 2)
    risk_level = get_risk_level(risk_score)

    if risk_level in ["high", "moderate"]:
        severity = "movement_abnormality"
    elif risk_level == "borderline":
        severity = "borderline_gait_change"
    else:
        severity = "normal_or_mild"

    # ML prediction
    ml_result = _ml_prediction(gait_data)

    result = {
        "agent_name": "gait_agent",
        "risk_score": risk_score,
        "risk_level": risk_level,
        "prediction": format_prediction("gait", risk_level),
        "severity": severity,
        "confidence": round(min(0.9, 0.5 + risk_score * 0.4), 2),
        "top_features": [
            "walking_speed",
            "stride_variability",
            "freezing_index",
            "balance_score",
        ],
        "explanation": (
            f"Gait agent estimated a {risk_level} movement-related risk level. "
            "It analyzed walking speed, stride variability, freezing index, and balance score."
        ),
        "ml_available": ml_result is not None,
    }

    if ml_result:
        result["ml_prediction"] = ml_result["predicted_class"]
        result["ml_confidence"] = round(ml_result["confidence"], 2)
        result["ml_class_probabilities"] = ml_result["class_probabilities"]

    return result
