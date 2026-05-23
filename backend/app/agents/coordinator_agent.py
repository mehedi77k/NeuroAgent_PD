from app.agents.risk_utils import get_risk_level, is_borderline, is_elevated


def coordinate_agents(clinical_result, speech_result, gait_result):
    clinical_risk = clinical_result["risk_score"]
    speech_risk = speech_result["risk_score"]
    gait_risk = gait_result["risk_score"]

    final_risk = (
        clinical_risk * 0.4
        + speech_risk * 0.3
        + gait_risk * 0.3
    )

    final_risk = round(final_risk, 2)
    final_risk_level = get_risk_level(final_risk)

    if final_risk_level == "high":
        final_prediction = "high_parkinson_risk"
    elif final_risk_level == "moderate":
        final_prediction = "moderate_parkinson_risk"
    elif final_risk_level == "borderline":
        final_prediction = "borderline_parkinson_risk"
    else:
        final_prediction = "low_parkinson_risk"

    all_results = [clinical_result, speech_result, gait_result]

    elevated_agents = [
        result["agent_name"]
        for result in all_results
        if is_elevated(result["risk_score"])
    ]

    borderline_agents = [
        result["agent_name"]
        for result in all_results
        if is_borderline(result["risk_score"])
    ]

    if elevated_agents and borderline_agents:
        agent_agreement = (
            f"{len(elevated_agents)} out of 3 agents detected elevated risk, "
            f"and {len(borderline_agents)} agent(s) showed borderline findings."
        )
    elif elevated_agents:
        agent_agreement = (
            f"{len(elevated_agents)} out of 3 agents detected elevated risk."
        )
    elif borderline_agents:
        agent_agreement = (
            f"No agent detected clearly elevated risk, but {len(borderline_agents)} agent(s) showed borderline findings."
        )
    else:
        agent_agreement = "No agent detected elevated or borderline risk."

    return {
        "agent_name": "coordinator_agent",
        "final_risk_score": final_risk,
        "final_risk_level": final_risk_level,
        "final_prediction": final_prediction,
        "agent_agreement": agent_agreement,
        "supporting_agents": elevated_agents,
        "borderline_agents": borderline_agents,
        "explanation": (
            "Coordinator combined clinical, speech, and gait risk scores using weighted fusion. "
            "Clinical data has the highest weight because structured clinical symptoms are the primary MVP signal."
        ),
    }