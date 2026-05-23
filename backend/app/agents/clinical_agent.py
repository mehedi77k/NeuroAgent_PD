from app.agents.risk_utils import clamp_score, format_prediction, get_risk_level


def analyze_clinical(clinical_data):
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

    risk_score = clamp_score(risk_score)
    risk_level = get_risk_level(risk_score)

    if updrs < 25:
        severity = "mild"
    elif updrs <= 45:
        severity = "moderate"
    else:
        severity = "severe"

    return {
        "agent_name": "clinical_agent",
        "risk_score": round(risk_score, 2),
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
    }