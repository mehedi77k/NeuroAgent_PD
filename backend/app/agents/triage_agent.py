def assign_triage(final_risk_score):
    if final_risk_score < 0.4:
        triage_level = "routine_monitoring"
        priority = "low"
        recommendation = "Continue routine observation and follow-up."
    elif final_risk_score < 0.7:
        triage_level = "priority_neurologist_review"
        priority = "medium"
        recommendation = "Schedule neurologist review and consider additional assessment."
    else:
        triage_level = "urgent_neurologist_review"
        priority = "high"
        recommendation = "Prioritize neurologist review due to high combined risk."

    return {
        "agent_name": "triage_agent",
        "triage_level": triage_level,
        "priority": priority,
        "recommendation": recommendation,
    }