from datetime import datetime
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict


class AnalysisHistoryResponse(BaseModel):
    id: int
    patient_id: str
    patient_name: Optional[str] = None

    final_risk_score: Optional[float] = None
    final_risk_level: Optional[str] = None
    final_prediction: Optional[str] = None

    triage_level: Optional[str] = None
    priority: Optional[str] = None

    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


class FeedbackCreate(BaseModel):
    patient_id: str
    analysis_id: Optional[int] = None
    action: Literal[
        "approve",
        "reject",
        "request_more_data",
        "escalate_to_neurologist",
    ]
    comment: Optional[str] = None


class FeedbackResponse(BaseModel):
    id: int
    patient_id: str
    analysis_id: Optional[int] = None
    action: str
    comment: Optional[str] = None
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)