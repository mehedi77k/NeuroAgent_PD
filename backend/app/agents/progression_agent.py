from app.agents.risk_utils import clamp_score, get_risk_level

import numpy as np

from app.models.ml_loader import (
    is_progression_model_available,
    progression_label_encoder,
    progression_model,
    progression_scaler,
)


def _get_medication_multiplier(medication_response: str) -> float:
    value = str(medication_response or "").lower()

    if value in ["poor", "limited", "low"]:
        return 0.06

    if value in ["moderate", "partial"]:
        return 0.04

    if value in ["good", "stable"]:
        return 0.02

    return 0.03


def _get_duration_multiplier(duration_years: float) -> float:
    try:
        duration = float(duration_years)
    except (TypeError, ValueError):
        duration = 0.0

    if duration >= 8:
        return 0.05

    if duration >= 5:
        return 0.04

    if duration >= 3:
        return 0.03

    return 0.02


def _get_agent_pressure(clinical_result, gait_result, conflict_result) -> float:
    clinical_risk = clinical_result.get("risk_score", 0.0)
    gait_risk = gait_result.get("risk_score", 0.0)

    pressure = 0.0

    if clinical_risk >= 0.7:
        pressure += 0.04
    elif clinical_risk >= 0.5:
        pressure += 0.025

    if gait_risk >= 0.7:
        pressure += 0.04
    elif gait_risk >= 0.5:
        pressure += 0.025
    elif gait_risk >= 0.4:
        pressure += 0.015

    if conflict_result and conflict_result.get("conflict_detected"):
        pressure += 0.015

    return pressure


def _ml_progression_prediction(patient_case, clinical_result, gait_result, coordinator_result):
    """
    ML-based progression classification.
    Returns dict with ml_ prefixed keys, or empty dict if unavailable.
    """

    if not is_progression_model_available():
        return {"ml_available": False}

    try:
        med_response = str(getattr(patient_case.clinical, "medication_response", "")).lower()
        med_encoded = {"good": 0, "stable": 0, "moderate": 1, "partial": 1, "poor": 2, "limited": 2}.get(med_response, 1)

        features = np.array([[
            coordinator_result.get("final_risk_score", 0.0),
            getattr(patient_case.clinical, "disease_duration_years", 0.0),
            med_encoded,
            clinical_result.get("risk_score", 0.0),
            gait_result.get("risk_score", 0.0),
            getattr(patient_case, "age", 60),
        ]])

        features_scaled = progression_scaler.transform(features)
        prediction_encoded = progression_model.predict(features_scaled)[0]
        probabilities = progression_model.predict_proba(features_scaled)[0]

        predicted_class = progression_label_encoder.inverse_transform([prediction_encoded])[0]
        confidence = float(np.max(probabilities))

        class_probs = {
            label: round(float(prob), 4)
            for label, prob in zip(progression_label_encoder.classes_, probabilities)
        }

        return {
            "ml_available": True,
            "ml_prediction": predicted_class,
            "ml_confidence": round(confidence, 2),
            "ml_class_probabilities": class_probs,
        }

    except Exception as e:
        print(f"[ProgressionAgent] ML prediction failed: {e}")
        return {"ml_available": False}


def simulate_progression(
    patient_case,
    clinical_result,
    speech_result,
    gait_result,
    coordinator_result,
    conflict_result=None,
):
    """
    MVP progression simulation.

    This is not a medical prediction model.
    It creates a simple 12-month simulated risk trajectory
    using coordinated risk, disease duration, medication response,
    gait involvement, and conflict signal.

    Also runs an ML-based progression classification if the model is available.
    """

    baseline_risk = coordinator_result.get("final_risk_score", 0.0)
    baseline_risk = clamp_score(float(baseline_risk))

    disease_duration = getattr(
        patient_case.clinical,
        "disease_duration_years",
        0.0,
    )

    medication_response = getattr(
        patient_case.clinical,
        "medication_response",
        "",
    )

    duration_factor = _get_duration_multiplier(disease_duration)
    medication_factor = _get_medication_multiplier(medication_response)
    agent_pressure = _get_agent_pressure(
        clinical_result,
        gait_result,
        conflict_result,
    )

    total_annual_increase = duration_factor + medication_factor + agent_pressure

    if baseline_risk >= 0.7:
        total_annual_increase += 0.02

    total_annual_increase = min(total_annual_increase, 0.18)

    timeline = []

    for month in [0, 3, 6, 9, 12]:
        month_fraction = month / 12
        projected_score = baseline_risk + total_annual_increase * month_fraction
        projected_score = round(clamp_score(projected_score), 2)

        timeline.append(
            {
                "month": month,
                "risk_score": projected_score,
                "risk_level": get_risk_level(projected_score),
            }
        )

    final_projected_score = timeline[-1]["risk_score"]
    final_projected_level = timeline[-1]["risk_level"]

    if final_projected_level == "high":
        progression_risk = "high_progression_risk"
    elif final_projected_level == "moderate":
        progression_risk = "moderate_progression_risk"
    elif final_projected_level == "borderline":
        progression_risk = "borderline_progression_risk"
    else:
        progression_risk = "low_progression_risk"

    return {
        "agent_name": "progression_agent",
        "baseline_risk_score": round(baseline_risk, 2),
        "baseline_risk_level": get_risk_level(baseline_risk),
        "projected_12_month_risk_score": final_projected_score,
        "projected_12_month_risk_level": final_projected_level,
        "progression_risk": progression_risk,
        "timeline": timeline,
        "factors_considered": [
            "coordinated_risk_score",
            "disease_duration_years",
            "medication_response",
            "clinical_risk_score",
            "gait_risk_score",
            "conflict_signal",
        ],
        "explanation": (
            "Progression simulation estimated possible 12-month risk change using "
            "coordinated risk score, disease duration, medication response, clinical signal, "
            "gait signal, and conflict findings."
        ),
        "safety_note": (
            "This is a simulation for clinical decision-support only and does not predict "
            "the exact future disease course."
        ),
        **_ml_progression_prediction(patient_case, clinical_result, gait_result, coordinator_result),
    }