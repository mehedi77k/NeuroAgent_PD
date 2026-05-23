from pydantic import BaseModel
from typing import Optional


class ClinicalData(BaseModel):
    updrs_score: float
    tremor_score: float
    rigidity_score: float
    bradykinesia_score: float
    disease_duration_years: float
    medication_response: str


class SpeechData(BaseModel):
    jitter: float
    shimmer: float
    hnr: float
    pitch_variation: float


class GaitData(BaseModel):
    walking_speed: float
    stride_variability: float
    freezing_index: float
    balance_score: float


class PatientCase(BaseModel):
    patient_id: str
    patient_name: Optional[str] = "Unknown Patient"
    age: int
    gender: str
    clinical: ClinicalData
    speech: SpeechData
    gait: GaitData
    notes: Optional[str] = None