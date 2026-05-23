from pathlib import Path


KNOWLEDGE_DIR = Path(__file__).resolve().parents[2] / "data" / "medical_knowledge"


def read_knowledge_file(filename: str) -> str:
    file_path = KNOWLEDGE_DIR / filename

    if not file_path.exists():
        return ""

    return file_path.read_text(encoding="utf-8").strip()


def extract_evidence_text(file_content: str) -> str:
    """
    Simple parser for MVP.
    It extracts the Evidence section if available.
    Otherwise returns the whole file content.
    """
    if not file_content:
        return "Evidence file was not found or is empty."

    marker = "Evidence:"
    if marker in file_content:
        evidence_part = file_content.split(marker, 1)[1]
        evidence_part = evidence_part.split("Clinical Use:", 1)[0]
        return evidence_part.strip()

    return file_content.strip()


def should_retrieve_for_level(risk_level: str) -> bool:
    return risk_level in ["borderline", "moderate", "high"]


def retrieve_medical_evidence(
    patient_case,
    clinical_result,
    speech_result,
    gait_result,
    coordinator_result,
    conflict_result=None,
):
    evidence_items = []

    clinical_level = clinical_result.get("risk_level", "low")
    speech_level = speech_result.get("risk_level", "low")
    gait_level = gait_result.get("risk_level", "low")
    final_level = coordinator_result.get("final_risk_level", "low")

    if should_retrieve_for_level(clinical_level):
        filename = "parkinsons_motor_symptoms.txt"
        content = read_knowledge_file(filename)

        evidence_items.append(
            {
                "topic": "motor symptoms",
                "source": filename,
                "matched_reason": (
                    f"Clinical agent detected {clinical_level} motor-related risk."
                ),
                "text": extract_evidence_text(content),
            }
        )

    if should_retrieve_for_level(speech_level):
        filename = "parkinsons_speech_changes.txt"
        content = read_knowledge_file(filename)

        evidence_items.append(
            {
                "topic": "speech changes",
                "source": filename,
                "matched_reason": (
                    f"Speech agent detected {speech_level} voice-related risk."
                ),
                "text": extract_evidence_text(content),
            }
        )

    if should_retrieve_for_level(gait_level):
        filename = "parkinsons_gait_changes.txt"
        content = read_knowledge_file(filename)

        evidence_items.append(
            {
                "topic": "gait changes",
                "source": filename,
                "matched_reason": (
                    f"Gait agent detected {gait_level} movement-related risk."
                ),
                "text": extract_evidence_text(content),
            }
        )

    conflict_detected = False
    if conflict_result:
        conflict_detected = conflict_result.get("conflict_detected", False)

    if final_level in ["moderate", "high"] or conflict_detected:
        filename = "clinical_safety_notes.txt"
        content = read_knowledge_file(filename)

        evidence_items.append(
            {
                "topic": "clinical safety",
                "source": filename,
                "matched_reason": (
                    "Final coordinated risk or conflict analysis indicates that clinician review is required."
                ),
                "text": extract_evidence_text(content),
            }
        )

    evidence_found = len(evidence_items) > 0

    return {
        "agent_name": "rag_agent",
        "evidence_found": evidence_found,
        "evidence_count": len(evidence_items),
        "evidence": evidence_items,
        "safety_note": (
            "Retrieved evidence is for clinical decision-support only and does not provide final diagnosis."
        ),
    }