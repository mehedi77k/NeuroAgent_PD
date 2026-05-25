def _explain_clinical(clinical_data, clinical_result):
    """
    Extracts feature contributions from the clinical agent's scoring logic.

    Clinical weights:
    - updrs_score: 0.55 (normalized by /60)
    - tremor_score: 0.15 (normalized by /4)
    - rigidity_score: 0.15 (normalized by /4)
    - bradykinesia_score: 0.15 (normalized by /4)
    """

    updrs = clinical_data.updrs_score
    tremor = clinical_data.tremor_score
    rigidity = clinical_data.rigidity_score
    bradykinesia = clinical_data.bradykinesia_score

    updrs_contribution = (updrs / 60) * 0.55
    tremor_contribution = (tremor / 4) * 0.15
    rigidity_contribution = (rigidity / 4) * 0.15
    bradykinesia_contribution = (bradykinesia / 4) * 0.15

    total = updrs_contribution + tremor_contribution + rigidity_contribution + bradykinesia_contribution

    if total == 0:
        total = 1.0

    return {
        "agent": "clinical_agent",
        "features": [
            {
                "name": "updrs_score",
                "label": "UPDRS Score",
                "value": updrs,
                "weight": 0.55,
                "contribution": round(updrs_contribution, 4),
                "percentage": round((updrs_contribution / total) * 100, 1),
            },
            {
                "name": "tremor_score",
                "label": "Tremor Score",
                "value": tremor,
                "weight": 0.15,
                "contribution": round(tremor_contribution, 4),
                "percentage": round((tremor_contribution / total) * 100, 1),
            },
            {
                "name": "rigidity_score",
                "label": "Rigidity Score",
                "value": rigidity,
                "weight": 0.15,
                "contribution": round(rigidity_contribution, 4),
                "percentage": round((rigidity_contribution / total) * 100, 1),
            },
            {
                "name": "bradykinesia_score",
                "label": "Bradykinesia Score",
                "value": bradykinesia,
                "weight": 0.15,
                "contribution": round(bradykinesia_contribution, 4),
                "percentage": round((bradykinesia_contribution / total) * 100, 1),
            },
        ],
        "risk_score": clinical_result.get("risk_score", 0.0),
    }


def _explain_speech(speech_data, speech_result):
    """
    Extracts feature contributions from the speech agent's scoring logic.

    Speech weights:
    - jitter: 0.30 (normalized by /0.04)
    - shimmer: 0.30 (normalized by /0.07)
    - hnr: 0.20 (normalized as (25 - hnr) / 25)
    - pitch_variation: 0.20 (normalized by /0.5)
    """

    jitter = speech_data.jitter
    shimmer = speech_data.shimmer
    hnr = speech_data.hnr
    pitch_variation = speech_data.pitch_variation

    jitter_score = min(1.0, jitter / 0.04)
    shimmer_score = min(1.0, shimmer / 0.07)
    hnr_score = max(0.0, min(1.0, (25 - hnr) / 25))
    pitch_score = min(1.0, pitch_variation / 0.5)

    jitter_contribution = jitter_score * 0.3
    shimmer_contribution = shimmer_score * 0.3
    hnr_contribution = hnr_score * 0.2
    pitch_contribution = pitch_score * 0.2

    total = jitter_contribution + shimmer_contribution + hnr_contribution + pitch_contribution

    if total == 0:
        total = 1.0

    return {
        "agent": "speech_agent",
        "features": [
            {
                "name": "jitter",
                "label": "Jitter",
                "value": jitter,
                "weight": 0.30,
                "contribution": round(jitter_contribution, 4),
                "percentage": round((jitter_contribution / total) * 100, 1),
            },
            {
                "name": "shimmer",
                "label": "Shimmer",
                "value": shimmer,
                "weight": 0.30,
                "contribution": round(shimmer_contribution, 4),
                "percentage": round((shimmer_contribution / total) * 100, 1),
            },
            {
                "name": "hnr",
                "label": "HNR",
                "value": hnr,
                "weight": 0.20,
                "contribution": round(hnr_contribution, 4),
                "percentage": round((hnr_contribution / total) * 100, 1),
            },
            {
                "name": "pitch_variation",
                "label": "Pitch Variation",
                "value": pitch_variation,
                "weight": 0.20,
                "contribution": round(pitch_contribution, 4),
                "percentage": round((pitch_contribution / total) * 100, 1),
            },
        ],
        "risk_score": speech_result.get("risk_score", 0.0),
    }


def _explain_gait(gait_data, gait_result):
    """
    Extracts feature contributions from the gait agent's scoring logic.

    Gait weights:
    - walking_speed: 0.25 (normalized as (1.4 - speed) / 1.4)
    - stride_variability: 0.25 (normalized by /0.6)
    - freezing_index: 0.30 (normalized by /0.8)
    - balance_score: 0.20 (normalized as 1 - balance)
    """

    walking_speed = gait_data.walking_speed
    stride_variability = gait_data.stride_variability
    freezing_index = gait_data.freezing_index
    balance_score = gait_data.balance_score

    speed_risk = max(0.0, min(1.0, (1.4 - walking_speed) / 1.4))
    stride_risk = min(1.0, stride_variability / 0.6)
    freezing_risk = min(1.0, freezing_index / 0.8)
    balance_risk = max(0.0, min(1.0, 1 - balance_score))

    speed_contribution = speed_risk * 0.25
    stride_contribution = stride_risk * 0.25
    freezing_contribution = freezing_risk * 0.30
    balance_contribution = balance_risk * 0.20

    total = speed_contribution + stride_contribution + freezing_contribution + balance_contribution

    if total == 0:
        total = 1.0

    return {
        "agent": "gait_agent",
        "features": [
            {
                "name": "walking_speed",
                "label": "Walking Speed",
                "value": walking_speed,
                "weight": 0.25,
                "contribution": round(speed_contribution, 4),
                "percentage": round((speed_contribution / total) * 100, 1),
            },
            {
                "name": "stride_variability",
                "label": "Stride Variability",
                "value": stride_variability,
                "weight": 0.25,
                "contribution": round(stride_contribution, 4),
                "percentage": round((stride_contribution / total) * 100, 1),
            },
            {
                "name": "freezing_index",
                "label": "Freezing Index",
                "value": freezing_index,
                "weight": 0.30,
                "contribution": round(freezing_contribution, 4),
                "percentage": round((freezing_contribution / total) * 100, 1),
            },
            {
                "name": "balance_score",
                "label": "Balance Score",
                "value": balance_score,
                "weight": 0.20,
                "contribution": round(balance_contribution, 4),
                "percentage": round((balance_contribution / total) * 100, 1),
            },
        ],
        "risk_score": gait_result.get("risk_score", 0.0),
    }


def _explain_coordinator(coordinator_result):
    """
    Explains the coordinator's weighted fusion.

    Coordinator weights:
    - clinical_agent: 0.40
    - speech_agent: 0.30
    - gait_agent: 0.30
    """

    final_risk = coordinator_result.get("final_risk_score", 0.0)

    return {
        "agent": "coordinator_agent",
        "fusion_weights": [
            {
                "agent": "clinical_agent",
                "label": "Clinical Agent",
                "weight": 0.40,
                "percentage": 40.0,
            },
            {
                "agent": "speech_agent",
                "label": "Speech Agent",
                "weight": 0.30,
                "percentage": 30.0,
            },
            {
                "agent": "gait_agent",
                "label": "Gait Agent",
                "weight": 0.30,
                "percentage": 30.0,
            },
        ],
        "final_risk_score": final_risk,
    }


def generate_explanations(
    patient_case,
    clinical_result,
    speech_result,
    gait_result,
    coordinator_result,
):
    """
    Explainability Agent.

    Generates feature-level contribution explanations for each
    domain agent and the coordinator fusion. This provides
    transparency into how each feature influenced the risk score.

    In the current MVP, contributions are derived directly from
    the rule-based scoring weights. When ML models are integrated,
    these will be replaced with SHAP values or similar methods.
    """

    clinical_explanation = _explain_clinical(
        patient_case.clinical,
        clinical_result,
    )

    speech_explanation = _explain_speech(
        patient_case.speech,
        speech_result,
    )

    gait_explanation = _explain_gait(
        patient_case.gait,
        gait_result,
    )

    coordinator_explanation = _explain_coordinator(coordinator_result)

    return {
        "agent_name": "explainability_agent",
        "clinical": clinical_explanation,
        "speech": speech_explanation,
        "gait": gait_explanation,
        "coordinator": coordinator_explanation,
        "method": "rule_based_weight_decomposition",
        "explanation": (
            "Feature contributions are computed from the rule-based scoring weights "
            "used by each domain agent. Each percentage shows how much a feature "
            "contributed to that agent's final risk score relative to other features."
        ),
        "safety_note": (
            "Explainability outputs are for clinical transparency only. "
            "They do not constitute a causal explanation of disease progression."
        ),
    }
