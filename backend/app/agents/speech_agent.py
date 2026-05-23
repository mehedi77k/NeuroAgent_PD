from app.agents.risk_utils import clamp_score, format_prediction, get_risk_level


def analyze_speech(speech_data):
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

    risk_score = clamp_score(risk_score)
    risk_level = get_risk_level(risk_score)

    if risk_level in ["high", "moderate"]:
        severity = "speech_abnormality"
    elif risk_level == "borderline":
        severity = "borderline_speech_change"
    else:
        severity = "normal_or_mild"

    return {
        "agent_name": "speech_agent",
        "risk_score": round(risk_score, 2),
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
    }