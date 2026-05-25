from app.agents.clinical_agent import analyze_clinical
from app.agents.speech_agent import analyze_speech
from app.agents.gait_agent import analyze_gait
from app.agents.coordinator_agent import coordinate_agents
from app.agents.triage_agent import assign_triage
from app.agents.conflict_agent import detect_conflicts
from app.agents.rag_agent import retrieve_medical_evidence
from app.agents.progression_agent import simulate_progression
from app.agents.dbs_referral_agent import evaluate_dbs_referral
from app.agents.explainability_agent import generate_explanations
from app.agents.critic_agent import run_critic
from app.agents.report_generator import generate_llm_report


def analyze_patient_case(patient_case):
    clinical_result = analyze_clinical(patient_case.clinical)
    speech_result = analyze_speech(patient_case.speech)
    gait_result = analyze_gait(patient_case.gait)

    coordinator_result = coordinate_agents(
        clinical_result,
        speech_result,
        gait_result,
    )

    triage_result = assign_triage(
        coordinator_result["final_risk_score"],
    )

    conflict_result = detect_conflicts(
        clinical_result,
        speech_result,
        gait_result,
        coordinator_result,
    )

    rag_result = retrieve_medical_evidence(
        patient_case,
        clinical_result,
        speech_result,
        gait_result,
        coordinator_result,
        conflict_result,
    )

    progression_result = simulate_progression(
        patient_case,
        clinical_result,
        speech_result,
        gait_result,
        coordinator_result,
        conflict_result,
    )

    dbs_referral_result = evaluate_dbs_referral(
        patient_case,
        clinical_result,
        gait_result,
        coordinator_result,
        progression_result,
    )

    explainability_result = generate_explanations(
        patient_case,
        clinical_result,
        speech_result,
        gait_result,
        coordinator_result,
    )

    critic_result = run_critic(
        clinical_result,
        speech_result,
        gait_result,
        coordinator_result,
        conflict_result,
        rag_result,
        dbs_referral_result,
    )

    patient_name = getattr(patient_case, "patient_name", "Unknown Patient")

    dbs_summary = ""
    if dbs_referral_result["referral_recommended"]:
        dbs_summary = (
            f" DBS referral level: {dbs_referral_result['referral_level']}. "
            f"{len(dbs_referral_result['criteria_met'])} of 6 candidacy criteria met."
        )
    else:
        dbs_summary = " DBS referral is not currently indicated."

    report = {
        "patient_id": patient_case.patient_id,
        "patient_name": patient_name,
        "summary": (
            f"Patient {patient_name} ({patient_case.patient_id}) was analyzed by clinical, "
            f"speech, and gait agents. The final risk level is "
            f"{coordinator_result['final_prediction']} with a risk score of "
            f"{coordinator_result['final_risk_score']}."
        ),
        "doctor_facing_explanation": (
            f"The system used multimodal agent outputs to estimate Parkinson-related risk. "
            f"The coordinated risk level is {coordinator_result['final_risk_level']}. "
            f"Conflict analysis result: {conflict_result['conflict_type']}. "
            f"The medical evidence agent retrieved {rag_result['evidence_count']} evidence item(s) "
            f"for clinician review. The progression simulation estimated a 12-month projected "
            f"risk level of {progression_result['projected_12_month_risk_level']} with a projected "
            f"risk score of {progression_result['projected_12_month_risk_score']}."
            f"{dbs_summary} "
            "This result should be reviewed by a qualified neurologist before any clinical decision."
        ),
        "medical_evidence_summary": (
            f"{rag_result['evidence_count']} relevant medical evidence item(s) were retrieved "
            "to support clinician review."
        ),
        "progression_summary": (
            f"Progression simulation projects the 12-month risk level as "
            f"{progression_result['projected_12_month_risk_level']} with a risk score of "
            f"{progression_result['projected_12_month_risk_score']}."
        ),
        "dbs_referral_summary": (
            f"DBS referral level: {dbs_referral_result['referral_level']}. "
            f"Referral score: {dbs_referral_result['referral_score']}. "
            f"Criteria met: {len(dbs_referral_result['criteria_met'])} of 6."
        ),
    }

    agent_results_dict = {
        "clinical": clinical_result,
        "speech": speech_result,
        "gait": gait_result,
        "coordinator": coordinator_result,
        "triage": triage_result,
        "conflict": conflict_result,
        "rag": rag_result,
        "progression": progression_result,
        "dbs_referral": dbs_referral_result,
        "explainability": explainability_result,
        "critic": critic_result,
    }

    llm_report_result = generate_llm_report(
        patient_case.model_dump(),
        agent_results_dict,
        report,
    )

    report["llm_generated"] = llm_report_result["llm_generated"]
    report["llm_report"] = llm_report_result["llm_report"]
    report["llm_provider"] = llm_report_result["llm_provider"]
    report["fallback_used"] = llm_report_result["fallback_used"]

    return {
        "patient": patient_case.model_dump(),
        "agent_results": agent_results_dict,
        "report": report,
    }