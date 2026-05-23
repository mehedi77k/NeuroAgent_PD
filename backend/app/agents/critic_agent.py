def run_critic(
    clinical_result,
    speech_result,
    gait_result,
    coordinator_result,
    conflict_result=None,
    rag_result=None,
):
    warnings = []

    if clinical_result["confidence"] < 0.65:
        warnings.append("Clinical agent confidence is relatively low.")

    if speech_result["confidence"] < 0.65:
        warnings.append("Speech agent confidence is relatively low.")

    if gait_result["confidence"] < 0.65:
        warnings.append("Gait agent confidence is relatively low.")

    supporting_agents = coordinator_result.get("supporting_agents", [])
    borderline_agents = coordinator_result.get("borderline_agents", [])

    if len(supporting_agents) < 2:
        if borderline_agents:
            warnings.append(
                "Fewer than two agents detected elevated risk, but borderline findings are present. "
                "Human review is recommended."
            )
        else:
            warnings.append(
                "Fewer than two agents detected elevated risk. Human review is recommended."
            )

    if conflict_result and conflict_result.get("conflict_detected"):
        warnings.append(
            f"Conflict agent detected {conflict_result['conflict_level']} disagreement: "
            f"{conflict_result['explanation']}"
        )

    if coordinator_result.get("final_risk_level") in ["moderate", "high"]:
        warnings.append(
            "The coordinated risk is clinically relevant and should be reviewed by a qualified neurologist."
        )

    if rag_result:
        evidence_count = rag_result.get("evidence_count", 0)

        if evidence_count == 0 and coordinator_result.get("final_risk_level") in [
            "moderate",
            "high",
        ]:
            warnings.append(
                "No medical evidence was retrieved for this moderate/high risk case. Human review is required."
            )
        elif evidence_count > 0:
            warnings.append(
                f"Medical evidence agent retrieved {evidence_count} supporting evidence item(s) for clinician review."
            )

    warnings.append(
        "This system is a clinical decision-support prototype and does not provide a final medical diagnosis."
    )

    return {
        "agent_name": "critic_agent",
        "requires_human_review": True,
        "warnings": warnings,
    }