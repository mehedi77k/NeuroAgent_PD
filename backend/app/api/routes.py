from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from app.data.demo_patients import DEMO_PATIENTS
from app.db.session import get_db
from app.models.analysis_history import AnalysisResult, DoctorFeedback
from app.schemas.history_schema import (
    AnalysisHistoryResponse,
    FeedbackCreate,
    FeedbackResponse,
)
from app.schemas.patient_schema import PatientCase
from app.services.analysis_service import analyze_patient_case


router = APIRouter()


def get_patient_summary(patient: dict) -> dict:
    return {
        "patient_id": patient["patient_id"],
        "patient_name": patient.get("patient_name", "Unknown Patient"),
        "age": patient["age"],
        "gender": patient["gender"],
        "disease_duration_years": patient["clinical"]["disease_duration_years"],
        "medication_response": patient["clinical"]["medication_response"],
        "notes": patient.get("notes", ""),
    }


@router.get("/patients")
def list_patients(search: Optional[str] = Query(default=None)):
    patients = DEMO_PATIENTS

    if search:
        query = search.strip().lower()
        patients = [
            patient
            for patient in DEMO_PATIENTS
            if query in patient["patient_id"].lower()
            or query in patient.get("patient_name", "").lower()
        ]

    return [get_patient_summary(patient) for patient in patients]


@router.get("/patients/{patient_id}")
def get_patient_by_id(patient_id: str):
    for patient in DEMO_PATIENTS:
        if patient["patient_id"].lower() == patient_id.lower():
            return patient

    raise HTTPException(
        status_code=404,
        detail=f"Patient with ID {patient_id} was not found.",
    )


@router.get("/demo-patient")
def get_demo_patient():
    return DEMO_PATIENTS[0]


@router.post("/analyze")
def analyze_patient(
    patient_case: PatientCase,
    db: Session = Depends(get_db),
):
    analysis = analyze_patient_case(patient_case)

    coordinator = analysis["agent_results"]["coordinator"]
    triage = analysis["agent_results"]["triage"]

    analysis_record = AnalysisResult(
        patient_id=patient_case.patient_id,
        patient_name=getattr(patient_case, "patient_name", "Unknown Patient"),
        final_risk_score=coordinator.get("final_risk_score"),
        final_risk_level=coordinator.get("final_risk_level"),
        final_prediction=coordinator.get("final_prediction"),
        triage_level=triage.get("triage_level"),
        priority=triage.get("priority"),
        full_analysis_json=analysis,
    )

    db.add(analysis_record)
    db.commit()
    db.refresh(analysis_record)

    analysis["analysis_id"] = analysis_record.id

    return analysis


@router.get("/history", response_model=list[AnalysisHistoryResponse])
def get_analysis_history(
    limit: int = Query(default=20, ge=1, le=100),
    db: Session = Depends(get_db),
):
    records = (
        db.query(AnalysisResult)
        .order_by(AnalysisResult.created_at.desc())
        .limit(limit)
        .all()
    )

    return records


@router.get("/history/{patient_id}", response_model=list[AnalysisHistoryResponse])
def get_patient_history(
    patient_id: str,
    limit: int = Query(default=20, ge=1, le=100),
    db: Session = Depends(get_db),
):
    records = (
        db.query(AnalysisResult)
        .filter(AnalysisResult.patient_id == patient_id)
        .order_by(AnalysisResult.created_at.desc())
        .limit(limit)
        .all()
    )

    return records


@router.get("/history/detail/{analysis_id}")
def get_analysis_detail(
    analysis_id: int,
    db: Session = Depends(get_db),
):
    record = (
        db.query(AnalysisResult)
        .filter(AnalysisResult.id == analysis_id)
        .first()
    )

    if not record:
        raise HTTPException(
            status_code=404,
            detail="Analysis result was not found.",
        )

    return {
        "id": record.id,
        "patient_id": record.patient_id,
        "patient_name": record.patient_name,
        "final_risk_score": record.final_risk_score,
        "final_risk_level": record.final_risk_level,
        "final_prediction": record.final_prediction,
        "triage_level": record.triage_level,
        "priority": record.priority,
        "created_at": record.created_at,
        "analysis": record.full_analysis_json,
    }


@router.post("/feedback", response_model=FeedbackResponse)
def submit_feedback(
    feedback: FeedbackCreate,
    db: Session = Depends(get_db),
):
    if feedback.analysis_id is not None:
        analysis_record = (
            db.query(AnalysisResult)
            .filter(AnalysisResult.id == feedback.analysis_id)
            .first()
        )

        if not analysis_record:
            raise HTTPException(
                status_code=404,
                detail="Analysis result was not found for this feedback.",
            )

    feedback_record = DoctorFeedback(
        patient_id=feedback.patient_id,
        analysis_id=feedback.analysis_id,
        action=feedback.action,
        comment=feedback.comment,
    )

    db.add(feedback_record)
    db.commit()
    db.refresh(feedback_record)

    return feedback_record