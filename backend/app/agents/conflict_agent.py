def detect_conflicts(clinical_result, speech_result, gait_result, coordinator_result):
    risks = {
        "clinical_agent": clinical_result["risk_score"],
        "speech_agent": speech_result["risk_score"],
        "gait_agent": gait_result["risk_score"],
    }

    max_agent = max(risks, key=risks.get)
    min_agent = min(risks, key=risks.get)

    max_risk = risks[max_agent]
    min_risk = risks[min_agent]
    risk_gap = round(max_risk - min_risk, 2)

    elevated_agents = coordinator_result.get("supporting_agents", [])
    borderline_agents = coordinator_result.get("borderline_agents", [])

    conflict_detected = False
    conflict_level = "none"
    conflict_type = "no_major_conflict"
    explanation = "No major conflict detected among agents."
    recommendation = "Continue standard review of the final coordinated result."

    if risk_gap >= 0.35:
        conflict_detected = True
        conflict_level = "high"
        conflict_type = "large_agent_risk_gap"
        explanation = (
            f"Large disagreement detected. {max_agent} reported a risk score of {max_risk}, "
            f"while {min_agent} reported a risk score of {min_risk}."
        )
        recommendation = (
            "Human review is recommended because agent findings differ substantially."
        )

    elif len(elevated_agents) == 1 and len(borderline_agents) >= 1:
        conflict_detected = True
        conflict_level = "moderate"
        conflict_type = "single_elevated_with_borderline_support"
        explanation = (
            "Only one agent detected clearly elevated risk, but at least one other agent showed borderline findings."
        )
        recommendation = (
            "Treat the result as clinically relevant but not fully confirmed across modalities."
        )

    elif len(elevated_agents) == 1 and len(borderline_agents) == 0:
        conflict_detected = True
        conflict_level = "moderate"
        conflict_type = "single_agent_support"
        explanation = (
            "Only one agent detected elevated risk while the other agents did not show strong supporting evidence."
        )
        recommendation = (
            "Human review is recommended before relying on the coordinated result."
        )

    elif len(elevated_agents) >= 2:
        conflict_detected = False
        conflict_level = "low"
        conflict_type = "multi_agent_support"
        explanation = (
            "Multiple agents detected elevated risk, suggesting stronger cross-modal agreement."
        )
        recommendation = (
            "Proceed with clinician review of the coordinated risk and triage recommendation."
        )

    return {
        "agent_name": "conflict_agent",
        "conflict_detected": conflict_detected,
        "conflict_level": conflict_level,
        "conflict_type": conflict_type,
        "risk_gap": risk_gap,
        "highest_risk_agent": max_agent,
        "lowest_risk_agent": min_agent,
        "explanation": explanation,
        "recommendation": recommendation,
    }