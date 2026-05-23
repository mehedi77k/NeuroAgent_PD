from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router
from app.db.session import init_db


app = FastAPI(
    title="NeuroAgent-PD API",
    description="Dockerized multi-agent Parkinson's clinical decision-support MVP",
    version="0.4.0",
)


@app.on_event("startup")
def on_startup():
    """
    Initialize database tables when the backend starts.

    This will create:
    - analysis_results
    - doctor_feedback
    """
    init_db()


app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


app.include_router(router)


@app.get("/")
def root():
    return {
        "message": "NeuroAgent-PD backend is running",
        "version": "0.4.0",
        "docs": "/docs",
    }


@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "service": "neuroagent-pd-backend",
        "version": "0.4.0",
    }