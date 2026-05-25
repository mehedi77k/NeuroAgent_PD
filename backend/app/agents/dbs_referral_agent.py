from app.agents.risk_utils import clamp_score


def _check_disease_duration(patient_case) -> bool:
    duration = getattr(patient_case.clinical, "disease_duration_years", 0)

    try:
        return float(duration) >= 5
    except (TypeError, ValueError):
        return False


def _check_medication_response(patient_case) -> bool:
    response = getattr(patient_case.clinical, "medication_response", "")
    return str(response).lower() in ["poor", "limited", "low"]


def _check_severe_motor(clinical_result) -> bool:
    risk_score = clinical_result.get("risk_score", 0.0)
    severity = clinical_result.get("severity", "")

    return risk_score >= 0.65 or severity == "severe"


def _check_gait_freezing(patient_case, gait_result) -> bool:
    freezing_index = getattr(patient_case.gait, "freezing_index", 0.0)
    gait_risk = gait_result.get("risk_score", 0.0)

    try:
        return float(freezing_index) >= 0.5 or gait_risk >= 0.6
    except (TypeError, ValueError):
        return gait_risk >= 0.6


def _check_coordinated_risk(coordinator_result) -> bool:
    level = coordinator_result.get("final_risk_level", "low")
    return level in ["moderate", "high"]


def _check_age_appropriate(patient_case) -> bool:
    age = getattr(patient_case, "age", 0)

    try:
        return int(age) <= 75
    except (TypeError, ValueError):
        return False


def evaluate_dbs_referral(
    patient_case,
    clinical_result,
    gait_result,
    coordinator_result,
    progression_result=None,
):
    """
    DBS Referral Support Agent.

    Evaluates whether the patient's clinical profile warrants
    a specialist discussion about Deep Brain Stimulation therapy.

    This agent does NOT recommend surgery. It flags cases where
    known DBS candidacy criteria are met and suggests the neurologist
    consider a referral for specialist evaluation.
    """

    criteria_checks = {
        "disease_duration_5_plus_years": {
            "met": _check_disease_duration(patient_case),
            "weight": 0.20,
            "label": "Disease duration ≥ 5 years",
        },
        "poor_medication_response": {
            "met": _check_medication_response(patient_case),
            "weight": 0.25,
            "label": "Poor or limited medication response",
        },
        "severe_motor_symptoms": {
            "met": _check_severe_motor(clinical_result),
            "weight": 0.20,
            "label": "Severe motor symptoms",
        },
        "significant_gait_freezing": {
            "met": _check_gait_freezing(patient_case, gait_result),
            "weight": 0.15,
            "label": "Significant gait freezing",
        },
        "high_coordinated_risk": {
            "met": _check_coordinated_risk(coordinator_result),
            "weight": 0.10,
            "label": "Moderate or high coordinated risk",
        },
        "age_under_75": {
            "met": _check_age_appropriate(patient_case),
            "weight": 0.10,
            "label": "Age appropriate for DBS consideration",
        },
    }

    referral_score = 0.0
    criteria_met = []
    criteria_not_met = []

    for key, check in criteria_checks.items():
        if check["met"]:
            referral_score += check["weight"]
            criteria_met.append(key)
        else:
            criteria_not_met.append(key)

    referral_score = round(clamp_score(referral_score), 2)

    if referral_score >= 0.7:
        referral_level = "strong"
        referral_recommended = True
        recommendation = (
            "Patient meets multiple DBS candidacy criteria. "
            "Consider referral to a DBS specialist or movement disorder neurosurgeon for evaluation."
        )
    elif referral_score >= 0.45:
        referral_level = "moderate"
        referral_recommended = True
        recommendation = (
            "Patient meets some DBS candidacy criteria. "
            "Discuss with the care team whether a DBS specialist consultation is appropriate."
        )
    else:
        referral_level = "not_indicated"
        referral_recommended = False
        recommendation = (
            "Patient does not currently meet sufficient DBS candidacy criteria. "
            "Continue standard treatment and reassess if symptoms progress."
        )

    if progression_result:
        projected_level = progression_result.get("projected_12_month_risk_level", "low")

        if projected_level == "high" and referral_level == "moderate":
            recommendation += (
                " Note: Progression simulation projects high risk at 12 months, "
                "which may strengthen the case for early specialist discussion."
            )

    met_count = len(criteria_met)
    total_count = len(criteria_checks)

    explanation = (
        f"DBS referral agent evaluated {total_count} candidacy criteria. "
        f"Patient meets {met_count} of {total_count} criteria "
        f"(referral score: {referral_score}). "
        f"Referral level: {referral_level}."
    )

    return {
        "agent_name": "dbs_referral_agent",
        "referral_recommended": referral_recommended,
        "referral_level": referral_level,
        "referral_score": referral_score,
        "criteria_met": criteria_met,
        "criteria_not_met": criteria_not_met,
        "criteria_details": {
            key: {
                "met": check["met"],
                "label": check["label"],
            }
            for key, check in criteria_checks.items()
        },
        "explanation": explanation,
        "recommendation": recommendation,
        "safety_note": (
            "DBS referral suggestion is for specialist discussion only. "
            "Final DBS candidacy requires comprehensive neurosurgical evaluation, "
            "neuropsychological testing, and multidisciplinary team review."
        ),
    }
