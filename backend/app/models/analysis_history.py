from sqlalchemy import JSON, Column, DateTime, Float, ForeignKey, Integer, String, func

from app.db.session import Base


class AnalysisResult(Base):
    __tablename__ = "analysis_results"

    id = Column(Integer, primary_key=True, index=True)

    patient_id = Column(String(50), index=True, nullable=False)
    patient_name = Column(String(120), nullable=True)

    final_risk_score = Column(Float, nullable=True)
    final_risk_level = Column(String(50), nullable=True)
    final_prediction = Column(String(120), nullable=True)

    triage_level = Column(String(120), nullable=True)
    priority = Column(String(50), nullable=True)

    full_analysis_json = Column(JSON, nullable=False)

    created_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )


class DoctorFeedback(Base):
    __tablename__ = "doctor_feedback"

    id = Column(Integer, primary_key=True, index=True)

    patient_id = Column(String(50), index=True, nullable=False)
    analysis_id = Column(
        Integer,
        ForeignKey("analysis_results.id"),
        nullable=True,
    )

    action = Column(String(50), nullable=False)
    comment = Column(String(500), nullable=True)

    created_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )