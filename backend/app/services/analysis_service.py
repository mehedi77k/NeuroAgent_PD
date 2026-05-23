from app.agents.clinical_agent import analyze_clinical
from app.agents.speech_agent import analyze_speech
from app.agents.gait_agent import analyze_gait
from app.agents.coordinator_agent import coordinate_agents
from app.agents.triage_agent import assign_triage
from app.agents.conflict_agent import detect_conflicts
from app.agents.rag_agent import retrieve_medical_evidence
from app.agents.progression_agent import simulate_progression
from app.agents.critic_agent import run_critic


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

    critic_result = run_critic(
        clinical_result,
        speech_result,
        gait_result,
        coordinator_result,
        conflict_result,
        rag_result,
    )

    patient_name = getattr(patient_case, "patient_name", "Unknown Patient")

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
            f"risk score of {progression_result['projected_12_month_risk_score']}. "
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
    }

    return {
        "patient": patient_case.model_dump(),
        "agent_results": {
            "clinical": clinical_result,
            "speech": speech_result,
            "gait": gait_result,
            "coordinator": coordinator_result,
            "triage": triage_result,
            "conflict": conflict_result,
            "rag": rag_result,
            "progression": progression_result,
            "critic": critic_result,
        },
        "report": report,
    }