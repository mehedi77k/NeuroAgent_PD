import numpy as np

from app.agents.risk_utils import clamp_score, format_prediction, get_risk_level
from app.models.ml_loader import (
    clinical_label_encoder,
    clinical_model,
    clinical_scaler,
    is_clinical_model_available,
)


def _rule_based_score(clinical_data):
    """Original rule-based clinical risk scoring."""

    updrs = clinical_data.updrs_score
    tremor = clinical_data.tremor_score
    rigidity = clinical_data.rigidity_score
    bradykinesia = clinical_data.bradykinesia_score

    risk_score = (
        (updrs / 60) * 0.55
        + (tremor / 4) * 0.15
        + (rigidity / 4) * 0.15
        + (bradykinesia / 4) * 0.15
    )

    return clamp_score(risk_score)


def _ml_prediction(clinical_data):
    """
    ML-based clinical risk prediction.
    Returns (predicted_class, confidence, probabilities) or None if unavailable.
    """

    if not is_clinical_model_available():
        return None

    try:
        med_response = str(clinical_data.medication_response).lower()
        med_encoded = {"good": 0, "stable": 0, "moderate": 1, "partial": 1, "poor": 2, "limited": 2}.get(med_response, 1)

        features = np.array([[
            clinical_data.updrs_score,
            clinical_data.tremor_score,
            clinical_data.rigidity_score,
            clinical_data.bradykinesia_score,
            clinical_data.disease_duration_years,
            med_encoded,
        ]])

        features_scaled = clinical_scaler.transform(features)
        prediction_encoded = clinical_model.predict(features_scaled)[0]
        probabilities = clinical_model.predict_proba(features_scaled)[0]

        predicted_class = clinical_label_encoder.inverse_transform([prediction_encoded])[0]
        confidence = float(np.max(probabilities))

        class_probs = {
            label: round(float(prob), 4)
            for label, prob in zip(clinical_label_encoder.classes_, probabilities)
        }

        return {
            "predicted_class": predicted_class,
            "confidence": confidence,
            "class_probabilities": class_probs,
        }

    except Exception as e:
        print(f"[ClinicalAgent] ML prediction failed: {e}")
        return None


def analyze_clinical(clinical_data):
    updrs = clinical_data.updrs_score

    # Rule-based scoring
    risk_score = _rule_based_score(clinical_data)
    risk_score = round(risk_score, 2)
    risk_level = get_risk_level(risk_score)

    if updrs < 25:
        severity = "mild"
    elif updrs <= 45:
        severity = "moderate"
    else:
        severity = "severe"

    # ML prediction
    ml_result = _ml_prediction(clinical_data)

    result = {
        "agent_name": "clinical_agent",
        "risk_score": risk_score,
        "risk_level": risk_level,
        "prediction": format_prediction("clinical", risk_level),
        "severity": severity,
        "confidence": round(min(0.95, 0.55 + risk_score * 0.4), 2),
        "top_features": [
            "updrs_score",
            "tremor_score",
            "rigidity_score",
            "bradykinesia_score",
        ],
        "explanation": (
            f"Clinical assessment suggests {severity} symptom severity. "
            f"The clinical risk level is {risk_level}, based mainly on UPDRS, tremor, rigidity, and bradykinesia scores."
        ),
        "ml_available": ml_result is not None,
    }

    if ml_result:
        result["ml_prediction"] = ml_result["predicted_class"]
        result["ml_confidence"] = round(ml_result["confidence"], 2)
        result["ml_class_probabilities"] = ml_result["class_probabilities"]

    return result
