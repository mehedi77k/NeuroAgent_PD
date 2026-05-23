def clamp_score(score: float) -> float:
    """
    Keeps risk score between 0.0 and 1.0.
    """
    return max(0.0, min(1.0, score))


def get_risk_level(score: float) -> str:
    """
    Converts numerical risk score into readable risk level.
    """
    score = clamp_score(score)

    if score < 0.4:
        return "low"
    if score < 0.5:
        return "borderline"
    if score < 0.7:
        return "moderate"

    return "high"


def is_elevated(score: float) -> bool:
    """
    Elevated means the agent detected moderate or high concern.
    """
    return score >= 0.5


def is_borderline(score: float) -> bool:
    """
    Borderline means the signal is not clearly high,
    but still close enough to require attention.
    """
    return 0.4 <= score < 0.5


def format_prediction(agent_type: str, risk_level: str) -> str:
    """
    Creates consistent prediction labels.
    """
    if risk_level == "high":
        return f"high_{agent_type}_risk"
    if risk_level == "moderate":
        return f"moderate_{agent_type}_risk"
    if risk_level == "borderline":
        return f"borderline_{agent_type}_risk"

    return f"low_{agent_type}_risk"