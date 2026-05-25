import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import faiss
from sentence_transformers import SentenceTransformer


BASE_DIR = Path(__file__).resolve().parents[2]

VECTOR_DIR = BASE_DIR / "data" / "medical_knowledge" / "vector_store"
INDEX_PATH = VECTOR_DIR / "index.faiss"
CHUNKS_PATH = VECTOR_DIR / "chunks.json"

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

_model = None
_index = None
_chunks = None


def _load_model():
    global _model

    if _model is None:
        _model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    return _model


def _load_vector_store():
    global _index, _chunks

    if _index is None or _chunks is None:
        if not INDEX_PATH.exists() or not CHUNKS_PATH.exists():
            return None, []

        _index = faiss.read_index(str(INDEX_PATH))

        with open(CHUNKS_PATH, "r", encoding="utf-8") as file:
            _chunks = json.load(file)

    return _index, _chunks


def _safe_get(obj, attr, default=""):
    if obj is None:
        return default

    return getattr(obj, attr, default)


def _safe_float(value, default=0.0):
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _shorten_text(text: str, max_chars: int = 1000) -> str:
    if not text:
        return ""

    text = " ".join(str(text).split())

    if len(text) <= max_chars:
        return text

    return text[:max_chars].rsplit(" ", 1)[0] + "..."


def _build_patient_profile(patient_case) -> Dict[str, Any]:
    clinical = getattr(patient_case, "clinical", None)
    speech = getattr(patient_case, "speech", None)
    gait = getattr(patient_case, "gait", None)

    return {
        "updrs_score": _safe_float(_safe_get(clinical, "updrs_score")),
        "tremor_score": _safe_float(_safe_get(clinical, "tremor_score")),
        "rigidity_score": _safe_float(_safe_get(clinical, "rigidity_score")),
        "bradykinesia_score": _safe_float(_safe_get(clinical, "bradykinesia_score")),
        "disease_duration_years": _safe_float(
            _safe_get(clinical, "disease_duration_years")
        ),
        "medication_response": str(
            _safe_get(clinical, "medication_response", "")
        ).lower(),
        "jitter": _safe_float(_safe_get(speech, "jitter")),
        "shimmer": _safe_float(_safe_get(speech, "shimmer")),
        "hnr": _safe_float(_safe_get(speech, "hnr")),
        "pitch_variation": _safe_float(_safe_get(speech, "pitch_variation")),
        "walking_speed": _safe_float(_safe_get(gait, "walking_speed")),
        "stride_length": _safe_float(_safe_get(gait, "stride_length")),
        "stride_variability": _safe_float(_safe_get(gait, "stride_variability")),
        "freezing_index": _safe_float(_safe_get(gait, "freezing_index")),
        "balance_score": _safe_float(_safe_get(gait, "balance_score")),
    }


def _make_patient_specific_queries(
    patient_case,
    clinical_result: Dict[str, Any],
    speech_result: Dict[str, Any],
    gait_result: Dict[str, Any],
    coordinator_result: Dict[str, Any],
    conflict_result: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, str]]:
    profile = _build_patient_profile(patient_case)

    queries: List[Dict[str, str]] = []

    brady = profile["bradykinesia_score"]
    tremor = profile["tremor_score"]
    rigidity = profile["rigidity_score"]
    updrs = profile["updrs_score"]
    med_response = profile["medication_response"]
    freezing = profile["freezing_index"]
    balance = profile["balance_score"]
    walking_speed = profile["walking_speed"]
    hnr = profile["hnr"]

    final_risk = coordinator_result.get("final_risk_level", "")
    clinical_risk = clinical_result.get("risk_level", "")
    gait_risk = gait_result.get("risk_level", "")
    speech_risk = speech_result.get("risk_level", "")

    # Core motor criteria query
    if brady >= 2 and (tremor >= 2 or rigidity >= 2):
        queries.append(
            {
                "focus": "Motor parkinsonism criteria",
                "reason": (
                    "Patient has clinically relevant bradykinesia with tremor "
                    "and/or rigidity, so the RAG search focuses on the MDS "
                    "definition of parkinsonism."
                ),
                "query": (
                    "MDS Parkinson disease diagnostic criteria motor parkinsonism "
                    "bradykinesia combined with rest tremor or rigidity"
                ),
            }
        )
    elif brady >= 2:
        queries.append(
            {
                "focus": "Bradykinesia-dominant presentation",
                "reason": (
                    "Patient has bradykinesia but less prominent tremor/rigidity, "
                    "so the RAG search focuses on how bradykinesia is used in "
                    "parkinsonism diagnosis."
                ),
                "query": (
                    "MDS diagnostic criteria bradykinesia required criterion "
                    "parkinsonism Parkinson disease"
                ),
            }
        )
    elif tremor >= 2:
        queries.append(
            {
                "focus": "Tremor-dominant presentation",
                "reason": (
                    "Patient has tremor-dominant findings, so the RAG search "
                    "focuses on rest tremor and supportive diagnostic criteria."
                ),
                "query": (
                    "MDS Parkinson disease rest tremor supportive criterion "
                    "diagnostic criteria"
                ),
            }
        )
    else:
        queries.append(
            {
                "focus": "Low or incomplete motor criteria",
                "reason": (
                    "Motor findings are not strongly diagnostic, so the RAG search "
                    "focuses on diagnostic uncertainty and criteria requirements."
                ),
                "query": (
                    "MDS Parkinson disease diagnostic criteria uncertainty "
                    "criteria not fulfilled parkinsonism"
                ),
            }
        )

    # Medication response query
    if med_response in ["good", "excellent", "strong", "marked"]:
        queries.append(
            {
                "focus": "Supportive levodopa response",
                "reason": (
                    "Medication response is favorable, so the RAG search focuses "
                    "on dopaminergic response as a supportive clinical feature."
                ),
                "query": (
                    "MDS Parkinson disease supportive criteria clear beneficial "
                    "response to dopaminergic therapy levodopa response"
                ),
            }
        )
    elif med_response in ["poor", "none", "minimal", "no"]:
        queries.append(
            {
                "focus": "Poor medication response / exclusion review",
                "reason": (
                    "Medication response is poor or absent, so the RAG search "
                    "focuses on exclusion criteria and diagnostic caution."
                ),
                "query": (
                    "MDS Parkinson disease absolute exclusion criteria absence "
                    "of response to high dose levodopa"
                ),
            }
        )
    else:
        queries.append(
            {
                "focus": "Medication response context",
                "reason": (
                    "Medication response is moderate or uncertain, so the RAG "
                    "search focuses on supportive criteria and diagnostic review."
                ),
                "query": (
                    "MDS Parkinson disease supportive criteria dopaminergic "
                    "response diagnostic review"
                ),
            }
        )

    # Gait/red flag query
    if freezing >= 0.4 or balance < 0.5 or walking_speed < 0.8:
        queries.append(
            {
                "focus": "Gait and red flag review",
                "reason": (
                    "Gait impairment, freezing, or low balance score is present, "
                    "so the RAG search focuses on red flags and clinician review."
                ),
                "query": (
                    "MDS Parkinson disease red flags early gait impairment "
                    "freezing of gait postural instability recurrent falls"
                ),
            }
        )

    # Speech-related query
    if hnr and hnr < 18:
        queries.append(
            {
                "focus": "Speech abnormality context",
                "reason": (
                    "Speech metrics are abnormal, so the RAG search includes "
                    "Parkinson-related speech and clinical context."
                ),
                "query": (
                    "Parkinson disease speech changes hypophonia clinical "
                    "features diagnostic context"
                ),
            }
        )

    # High-risk criteria query
    if str(final_risk).lower() == "high" or updrs >= 30:
        queries.append(
            {
                "focus": "Clinically established Parkinson disease criteria",
                "reason": (
                    "Overall risk or UPDRS score is high, so the RAG search "
                    "focuses on clinically established Parkinson disease criteria."
                ),
                "query": (
                    "MDS clinically established Parkinson disease supportive "
                    "criteria absence of absolute exclusion criteria red flags"
                ),
            }
        )
    else:
        queries.append(
            {
                "focus": "Clinically probable Parkinson disease criteria",
                "reason": (
                    "Overall risk is not clearly high, so the RAG search focuses "
                    "on clinically probable Parkinson disease and uncertainty."
                ),
                "query": (
                    "MDS clinically probable Parkinson disease red flags "
                    "supportive criteria diagnostic criteria"
                ),
            }
        )

    # Conflict query
    if conflict_result and conflict_result.get("conflict_detected"):
        queries.append(
            {
                "focus": "Conflicting agent findings",
                "reason": (
                    "The system detected conflicting findings across agents, so "
                    "the RAG search focuses on diagnostic uncertainty and review."
                ),
                "query": (
                    "MDS Parkinson disease diagnostic uncertainty conflicting "
                    "features red flags clinician review"
                ),
            }
        )

    # Add compact patient facts to each query
    patient_facts = (
        f"UPDRS {updrs}, tremor {tremor}, rigidity {rigidity}, "
        f"bradykinesia {brady}, medication response {med_response}, "
        f"freezing index {freezing}, balance score {balance}, "
        f"walking speed {walking_speed}, final risk {final_risk}, "
        f"clinical risk {clinical_risk}, gait risk {gait_risk}, speech risk {speech_risk}"
    )

    for item in queries:
        item["query"] = f"{item['query']} Patient context: {patient_facts}"

    return queries


def _semantic_search_single_query(query: str, top_k: int = 4) -> List[Dict[str, Any]]:
    index, chunks = _load_vector_store()

    if index is None or not chunks:
        return []

    model = _load_model()

    query_embedding = model.encode([query], convert_to_numpy=True).astype("float32")
    faiss.normalize_L2(query_embedding)

    scores, indices = index.search(query_embedding, top_k)

    results = []

    for score, idx in zip(scores[0], indices[0]):
        if idx < 0 or idx >= len(chunks):
            continue

        chunk = chunks[idx]

        results.append(
            {
                "source": chunk.get("source_file"),
                "page": chunk.get("page"),
                "chunk_id": chunk.get("chunk_id"),
                "similarity_score": round(float(score), 4),
                "text": chunk.get("text"),
            }
        )

    return results


def _semantic_search_multi_query(
    query_items: List[Dict[str, str]],
    per_query_top_k: int = 3,
    final_top_k: int = 5,
) -> List[Dict[str, Any]]:
    collected: List[Dict[str, Any]] = []

    for query_item in query_items:
        results = _semantic_search_single_query(
            query=query_item["query"],
            top_k=per_query_top_k,
        )

        for result in results:
            result["evidence_focus"] = query_item["focus"]
            result["matched_reason"] = query_item["reason"]
            result["search_query"] = query_item["query"]
            collected.append(result)

    # Deduplicate by chunk ID.
    deduped: Dict[str, Dict[str, Any]] = {}

    for item in collected:
        chunk_id = item.get("chunk_id")

        if not chunk_id:
            continue

        existing = deduped.get(chunk_id)

        if existing is None:
            deduped[chunk_id] = item
        else:
            if item.get("similarity_score", 0) > existing.get("similarity_score", 0):
                deduped[chunk_id] = item

    ranked = sorted(
        deduped.values(),
        key=lambda item: item.get("similarity_score", 0),
        reverse=True,
    )

    # Keep page/focus diversity where possible.
    diversified = []
    used_pages = set()
    used_focuses = set()

    for item in ranked:
        page = item.get("page")
        focus = item.get("evidence_focus")

        if page not in used_pages or focus not in used_focuses:
            diversified.append(item)
            used_pages.add(page)
            used_focuses.add(focus)

        if len(diversified) >= final_top_k:
            break

    if len(diversified) < final_top_k:
        for item in ranked:
            if item not in diversified:
                diversified.append(item)

            if len(diversified) >= final_top_k:
                break

    return diversified[:final_top_k]


def retrieve_medical_evidence(
    patient_case,
    clinical_result,
    speech_result,
    gait_result,
    coordinator_result,
    conflict_result=None,
):
    query_items = _make_patient_specific_queries(
        patient_case=patient_case,
        clinical_result=clinical_result,
        speech_result=speech_result,
        gait_result=gait_result,
        coordinator_result=coordinator_result,
        conflict_result=conflict_result,
    )

    retrieved_chunks = _semantic_search_multi_query(
        query_items=query_items,
        per_query_top_k=3,
        final_top_k=5,
    )

    evidence = []

    for item in retrieved_chunks:
        evidence.append(
            {
                "topic": item.get("evidence_focus", "MDS Parkinson clinical diagnostic criteria"),
                "source": item.get("source"),
                "page": item.get("page"),
                "chunk_id": item.get("chunk_id"),
                "similarity_score": item.get("similarity_score"),
                "matched_reason": item.get("matched_reason"),
                "search_query": item.get("search_query"),
                "text": _shorten_text(item.get("text", "")),
            }
        )

    return {
        "agent_name": "rag_agent",
        "retrieval_method": "patient_specific_multi_query_pdf_vector_search_faiss",
        "knowledge_base": "MDS Clinical Diagnostic Criteria for Parkinson's Disease",
        "query_strategy": "patient_specific_multi_query",
        "patient_specific_queries": query_items,
        "evidence_found": len(evidence) > 0,
        "evidence_count": len(evidence),
        "evidence": evidence,
        "safety_note": (
            "This evidence is for clinician review only. It is not a standalone "
            "diagnosis or treatment recommendation."
        ),
    }