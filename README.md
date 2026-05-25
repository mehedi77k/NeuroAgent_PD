# NeuroAgent_PD: Multi-Agent Parkinson’s Disease Clinical Decision-Support Platform

![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-Backend-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![React](https://img.shields.io/badge/React-Frontend-20232A?style=for-the-badge&logo=react&logoColor=61DAFB)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-Database-316192?style=for-the-badge&logo=postgresql&logoColor=white)
![FAISS](https://img.shields.io/badge/FAISS-Vector_Search-blue?style=for-the-badge)
![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?style=for-the-badge&logo=docker&logoColor=white)
![Status](https://img.shields.io/badge/Status-MVP_Prototype-success?style=for-the-badge)

**NeuroAgent_PD** is a full-stack multi-agent clinical decision-support prototype for Parkinson’s disease assessment. It combines structured clinical features, speech indicators, gait indicators, coordinated risk scoring, triage support, conflict detection, PDF-based RAG medical evidence retrieval, progression simulation, DBS specialist discussion flagging, explainability, critic safety review, doctor feedback, and report generation into a single clinician-facing dashboard.

> **Medical Safety Notice**  
> NeuroAgent_PD is a research and academic prototype. It does **not** provide a final medical diagnosis, prescription, treatment decision, surgical recommendation, or replacement for a qualified neurologist. All outputs must be reviewed by a licensed medical professional before any clinical interpretation or action.

---

## Table of Contents

- [Overview](#overview)
- [Demo Workflow](#demo-workflow)
- [Core Features](#core-features)
- [Agent System](#agent-system)
- [RAG Medical Evidence Pipeline](#rag-medical-evidence-pipeline)
- [Technology Stack](#technology-stack)
- [System Architecture](#system-architecture)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Run with Docker](#run-with-docker)
- [Run Manually](#run-manually)
- [Environment Variables](#environment-variables)
- [API Endpoints](#api-endpoints)
- [Patient Case Schema](#patient-case-schema)
- [Example Analysis Response](#example-analysis-response)
- [Database Models](#database-models)
- [Machine Learning Training Pipeline](#machine-learning-training-pipeline)
- [LLM Report Generation](#llm-report-generation)
- [Clinical Safety Notes](#clinical-safety-notes)
- [Development Commands](#development-commands)
- [Troubleshooting](#troubleshooting)
- [Known Limitations](#known-limitations)
- [Future Improvements](#future-improvements)
- [Repository](#repository)
- [Project Status](#project-status)
- [Disclaimer](#disclaimer)

---

## Overview

NeuroAgent_PD is designed as a doctor-facing Parkinson’s disease clinical decision-support dashboard.

The system allows clinicians, researchers, or evaluators to:

- View demo Parkinson’s patient cases
- Search patients by patient ID or name
- Inspect structured patient profile data
- Analyze clinical, speech, and gait features
- Run a full multi-agent Parkinson’s risk assessment
- Generate coordinated final risk prediction
- Review triage recommendation
- Detect disagreement between agent outputs
- Retrieve patient-specific medical evidence from a PDF-based RAG knowledge base
- Review source file, page number, similarity score, matched reason, and retrieved evidence text
- Simulate 12-month progression risk
- Evaluate DBS specialist discussion indicators
- View feature-level explainability
- Review automated critic warnings
- Generate doctor-facing clinical reports
- Save analysis results to PostgreSQL
- Submit doctor feedback for each analysis

The platform uses a modular architecture:

```text
React Frontend Dashboard
        ↓
FastAPI Backend
        ↓
Multi-Agent Clinical AI Layer
        ↓
PostgreSQL + PDF Vector Knowledge Base
```

---

## Demo Workflow

Use this flow when demonstrating the project:

```text
1. Start the full system with Docker Compose.
2. Open the frontend at http://localhost:5173.
3. Select a patient case from the patient list.
4. Review the patient profile, clinical scores, speech features, and gait features.
5. Click Run Agent Analysis.
6. Open Agent Analysis to review clinical, speech, gait, coordinator, triage, conflict, progression, DBS, explainability, and critic outputs.
7. Open Medical Evidence to review PDF-based RAG evidence from the MDS Parkinson’s diagnostic criteria document.
8. Open Reports to review the doctor-facing summary.
9. Submit doctor feedback such as approve, reject, request more data, or escalate to neurologist.
10. Review saved analysis history.
```

Important demo note:

```text
The first RAG-enabled analysis may take longer because the sentence-transformer embedding model is loaded into memory. Later analyses are usually faster.
```

---

## Core Features

### Patient Case Management

- Demo patient dataset support
- Patient list endpoint
- Patient search by patient ID or name
- Full patient profile endpoint
- Structured clinical, speech, and gait inputs
- Patient notes support
- Patient-level analysis history

### Multi-Agent Clinical Analysis

The system runs multiple specialized agents over the same patient case:

- Clinical Agent
- Speech Agent
- Gait Agent
- Coordinator Agent
- Triage Agent
- Conflict Agent
- Medical Evidence / RAG Agent
- Progression Agent
- DBS Specialist Discussion Agent
- Explainability Agent
- Critic Agent
- Report Generator

### Risk Scoring

Risk level categories:

```text
low
borderline
moderate
high
```

Risk score range:

```text
0.0 to 1.0
```

Coordinator fusion weights:

```text
Clinical Agent: 40%
Speech Agent:   30%
Gait Agent:     30%
```

### Rule-Based + ML-Supported Agents

The domain agents support:

- Rule-based clinical scoring
- Optional machine learning prediction
- Model confidence score
- Model class probability output
- Model artifact availability check
- Automatic fallback when model artifacts are missing

If trained model artifacts are not available, the system continues with rule-based scoring.

### Doctor-Facing Dashboard

The React frontend provides a clinical dashboard for:

- Selecting patient cases
- Running multi-agent analysis
- Viewing individual agent outputs
- Viewing final coordinated risk
- Reviewing triage recommendation
- Reading generated clinical reports
- Checking PDF-based medical evidence
- Reviewing progression simulation
- Checking DBS specialist discussion indicators
- Reviewing explainability output
- Submitting doctor feedback
- Viewing saved analysis history

### Analysis History

Each analysis is stored in PostgreSQL with:

- Patient ID
- Patient name
- Final risk score
- Final risk level
- Final prediction
- Triage level
- Priority
- Full analysis JSON
- Creation timestamp

### Doctor Feedback

Doctors can submit feedback for an analysis.

Supported feedback actions:

```text
approve
reject
request_more_data
escalate_to_neurologist
```

Feedback can include:

- Patient ID
- Analysis ID
- Action
- Optional doctor comment
- Timestamp

---

## Agent System

### 1. Clinical Agent

Analyzes structured clinical symptoms.

Input features:

- UPDRS score
- Tremor score
- Rigidity score
- Bradykinesia score
- Disease duration
- Medication response

Outputs:

- Clinical risk score
- Clinical risk level
- Severity label
- Confidence
- Top contributing features
- Optional ML prediction
- Optional ML confidence
- Optional ML class probabilities

---

### 2. Speech Agent

Analyzes speech-related indicators.

Input features:

- Jitter
- Shimmer
- HNR
- Pitch variation

Outputs:

- Speech risk score
- Speech risk level
- Speech abnormality severity
- Confidence
- Optional ML prediction
- Optional ML confidence
- Optional ML class probabilities

---

### 3. Gait Agent

Analyzes gait and movement-related indicators.

Input features:

- Walking speed
- Stride variability
- Freezing index
- Balance score

Outputs:

- Gait risk score
- Gait risk level
- Movement abnormality severity
- Confidence
- Optional ML prediction
- Optional ML confidence
- Optional ML class probabilities

---

### 4. Coordinator Agent

Combines outputs from the clinical, speech, and gait agents into one final risk assessment.

Fusion logic:

```text
final_risk =
  clinical_risk * 0.40 +
  speech_risk   * 0.30 +
  gait_risk     * 0.30
```

Outputs:

- Final risk score
- Final risk level
- Final prediction
- Supporting agents
- Borderline agents
- Agreement summary

---

### 5. Triage Agent

Maps the final coordinated risk score to a clinical review priority.

Example triage outputs:

```text
routine_monitoring
priority_neurologist_review
urgent_neurologist_review
```

Outputs:

- Triage level
- Priority
- Recommendation
- Safety guidance

---

### 6. Conflict Agent

Detects disagreement between the domain agents.

It checks:

- Difference between highest and lowest agent scores
- Whether only one agent is strongly positive
- Whether multiple agents agree
- Whether any result is borderline
- Whether conflict requires clinician review

Outputs:

- Conflict detected or not
- Conflict level
- Conflict type
- Highest-risk agent
- Lowest-risk agent
- Recommendation

---

### 7. Medical Evidence / RAG Agent

Retrieves patient-specific medical evidence from a PDF-based vector knowledge base.

Current knowledge source:

```text
MDS Clinical Diagnostic Criteria for Parkinson's Disease
```

Current RAG pipeline:

```text
PDF → text extraction → chunking → sentence-transformer embeddings → FAISS vector index → patient-specific evidence retrieval
```

The RAG Agent builds patient-specific queries using:

- Clinical risk level
- Speech risk level
- Gait risk level
- Final coordinated risk level
- UPDRS score
- Tremor score
- Rigidity score
- Bradykinesia score
- Medication response
- Freezing index
- Balance score
- Walking speed
- Conflict signal

Outputs:

- Retrieval method
- Query strategy
- Patient-specific queries
- Evidence count
- Retrieved evidence items
- Evidence topic
- Source PDF file
- Page number
- Chunk ID
- Similarity score
- Matched reason
- Patient-specific search query
- Retrieved evidence text
- Clinical safety note

Important:

```text
The RAG Agent retrieves evidence for clinician review only.
It does not provide a standalone diagnosis.
```

---

### 8. Progression Agent

Simulates a 12-month Parkinson’s risk trajectory.

Inputs considered:

- Coordinated risk score
- Clinical risk score
- Gait risk score
- Disease duration
- Medication response
- Patient age
- Conflict signal

Timeline points:

```text
0 months
3 months
6 months
9 months
12 months
```

Outputs:

- Baseline risk score
- Baseline risk level
- Projected 12-month risk score
- Projected 12-month risk level
- Progression category
- Timeline
- Optional ML progression prediction

---

### 9. DBS Specialist Discussion Agent

Evaluates whether the patient profile may justify discussion with a DBS specialist.

Criteria checked:

- Disease duration of 5+ years
- Poor or limited medication response
- Severe motor symptoms
- Significant gait freezing
- Moderate or high coordinated risk
- Age under 75

Outputs:

- Referral discussion flag
- Referral level
- Referral score
- Criteria met
- Criteria not met
- Recommendation
- Safety note

Important:

```text
This agent does not recommend surgery.
It only flags whether specialist DBS evaluation discussion may be appropriate.
```

---

### 10. Explainability Agent

Generates feature-level explanations for the clinical, speech, gait, and coordinator outputs.

Current explanation method:

```text
rule_based_weight_decomposition
```

It explains:

- Clinical feature contributions
- Speech feature contributions
- Gait feature contributions
- Coordinator fusion weights

Future versions can replace this with SHAP, LIME, or model-native explainability when validated ML models are used.

---

### 11. Critic Agent

Reviews the full analysis for safety and reliability.

It checks:

- Low-confidence results
- Weak multi-agent support
- Conflict signals
- Moderate or high coordinated risk
- Evidence availability
- DBS discussion signal
- Need for human review

Outputs:

- Warnings
- Safety notes
- Human-review requirement
- Clinical caution messages

---

### 12. Report Generator

Generates a doctor-facing clinical report.

The report can include:

- Patient summary
- Final coordinated risk
- Triage recommendation
- Agent agreement summary
- Conflict notes
- Medical evidence summary
- Progression summary
- DBS specialist discussion summary
- Safety warning
- LLM-generated report when enabled
- Template fallback report when LLM is disabled or unavailable

Supported optional LLM providers:

- Groq
- Gemini

---

## RAG Medical Evidence Pipeline

The project includes a PDF-based RAG pipeline for medical evidence retrieval.

### Knowledge Base Location

```text
backend/data/medical_knowledge/pdf_sources/
```

Current PDF source:

```text
backend/data/medical_knowledge/pdf_sources/mds-clinical-diagnostic-criteria-for-parkinson-disease.pdf
```

### Vector Store Location

```text
backend/data/medical_knowledge/vector_store/
```

Generated files:

```text
index.faiss
chunks.json
```

### RAG Build Script

```text
backend/scripts/build_rag_index.py
```

### Build or Rebuild the RAG Index

Run this command from the project root:

```bash
docker compose run --rm backend python scripts/build_rag_index.py
```

Expected output:

```text
RAG index built successfully.
Chunks: <number_of_chunks>
Index path: /app/data/medical_knowledge/vector_store/index.faiss
```

### When to Rebuild the RAG Index

Rebuild the index when:

- A new PDF is added
- An existing PDF is renamed
- An existing PDF is replaced
- `index.faiss` is deleted
- `chunks.json` is deleted
- Evidence retrieval returns no results even though PDFs exist

### RAG Dependencies

The backend uses:

```text
pypdf
sentence-transformers
faiss-cpu
```

### RAG Output Example

```json
{
  "agent_name": "rag_agent",
  "retrieval_method": "patient_specific_multi_query_pdf_vector_search_faiss",
  "knowledge_base": "MDS Clinical Diagnostic Criteria for Parkinson's Disease",
  "query_strategy": "patient_specific_multi_query",
  "evidence_found": true,
  "evidence_count": 4,
  "evidence": [
    {
      "topic": "Motor parkinsonism criteria",
      "source": "mds-clinical-diagnostic-criteria-for-parkinson-disease.pdf",
      "page": 5,
      "chunk_id": "mds-clinical-diagnostic-criteria-for-parkinson-disease_p5_c1",
      "similarity_score": 0.7675,
      "matched_reason": "Patient has clinically relevant bradykinesia with tremor and/or rigidity, so the RAG search focuses on the MDS definition of parkinsonism.",
      "search_query": "MDS Parkinson disease diagnostic criteria motor parkinsonism bradykinesia combined with rest tremor or rigidity Patient context: ...",
      "text": "Retrieved evidence text..."
    }
  ],
  "safety_note": "This evidence is for clinician review only. It is not a standalone diagnosis or treatment recommendation."
}
```

---

## Technology Stack

| Layer | Technologies |
|---|---|
| Frontend | React, Vite, JavaScript, CSS |
| Backend | Python 3.11, FastAPI, Uvicorn |
| Database | PostgreSQL |
| ORM | SQLAlchemy |
| Validation | Pydantic |
| ML / Data | pandas, numpy, scikit-learn, joblib |
| RAG / Evidence Retrieval | pypdf, sentence-transformers, FAISS |
| Medical Knowledge Base | PDF sources, vector_store/index.faiss, chunks.json |
| LLM Integration | Groq API, Gemini API, httpx |
| Containerization | Docker, Docker Compose |
| API Documentation | FastAPI Swagger UI |
| Development Runtime | Node.js, Python virtual environment |

---

## System Architecture

```text
┌──────────────────────────────────────────────────────────────────────┐
│                              User Layer                              │
│                                                                      │
│                  Doctor / Clinician / Research User                  │
│                                                                      │
│  - Select patient case                                               │
│  - Run Parkinson’s multi-agent analysis                              │
│  - Review risk, triage, evidence, explainability, and report         │
│  - Submit feedback                                                   │
└──────────────────────────────────────┬───────────────────────────────┘
                                       │
                                       ▼
┌──────────────────────────────────────────────────────────────────────┐
│                           Frontend Layer                             │
│                                                                      │
│                         React + Vite Dashboard                       │
│                                                                      │
│  Main UI Areas:                                                       │
│  - Patient selection                                                  │
│  - Patient profile                                                    │
│  - Agent analysis results                                             │
│  - Risk and triage summary                                            │
│  - Medical evidence                                                   │
│  - Progression simulation                                             │
│  - DBS specialist discussion flag                                     │
│  - Explainability                                                     │
│  - Clinical report                                                    │
│  - Doctor feedback                                                    │
│  - Analysis history                                                   │
└──────────────────────────────────────┬───────────────────────────────┘
                                       │
                                       │ REST API
                                       ▼
┌──────────────────────────────────────────────────────────────────────┐
│                            Backend Layer                             │
│                                                                      │
│                              FastAPI API                             │
│                                                                      │
│  Endpoints:                                                           │
│  - /patients                                                          │
│  - /patients/{patient_id}                                             │
│  - /demo-patient                                                      │
│  - /analyze                                                           │
│  - /history                                                           │
│  - /history/{patient_id}                                              │
│  - /history/detail/{analysis_id}                                      │
│  - /feedback                                                          │
└──────────────────────────────────────┬───────────────────────────────┘
                                       │
                                       ▼
┌──────────────────────────────────────────────────────────────────────┐
│                         Multi-Agent AI Layer                         │
│                                                                      │
│  Clinical Agent                                                       │
│  Speech Agent                                                         │
│  Gait Agent                                                           │
│        │                                                             │
│        ▼                                                             │
│  Coordinator Agent                                                    │
│        │                                                             │
│        ├── Triage Agent                                               │
│        ├── Conflict Agent                                             │
│        ├── Medical Evidence / RAG Agent                               │
│        ├── Progression Agent                                          │
│        ├── DBS Specialist Discussion Agent                            │
│        ├── Explainability Agent                                       │
│        ├── Critic Agent                                               │
│        └── Report Generator                                           │
└──────────────────────────────────────┬───────────────────────────────┘
                                       │
                ┌──────────────────────┴──────────────────────┐
                ▼                                             ▼
┌──────────────────────────────────────┐     ┌──────────────────────────────────────┐
│          Data Persistence Layer       │     │       Medical Knowledge Layer         │
│                                      │     │                                      │
│              PostgreSQL              │     │      PDF Sources + FAISS Index        │
│                                      │     │                                      │
│  Tables:                             │     │  - MDS diagnostic criteria PDF        │
│  - analysis_results                  │     │  - chunks.json                        │
│  - index.faiss                       │     │  - index.faiss                        │
│  - doctor_feedback                   │     │                                      │
└──────────────────────────────────────┘     └──────────────────────────────────────┘
```

---

## Project Structure

```text
NeuroAgent_PD
│
├── backend
│   ├── app
│   │   ├── agents
│   │   │   ├── clinical_agent.py
│   │   │   ├── speech_agent.py
│   │   │   ├── gait_agent.py
│   │   │   ├── coordinator_agent.py
│   │   │   ├── triage_agent.py
│   │   │   ├── conflict_agent.py
│   │   │   ├── rag_agent.py
│   │   │   ├── progression_agent.py
│   │   │   ├── dbs_referral_agent.py
│   │   │   ├── explainability_agent.py
│   │   │   ├── critic_agent.py
│   │   │   ├── report_generator.py
│   │   │   └── risk_utils.py
│   │   │
│   │   ├── api
│   │   │   └── routes.py
│   │   │
│   │   ├── core
│   │   │   └── config.py
│   │   │
│   │   ├── data
│   │   │   ├── demo_patients.py
│   │   │   └── demo_patients.csv
│   │   │
│   │   ├── db
│   │   │   └── session.py
│   │   │
│   │   ├── models
│   │   │   ├── analysis_history.py
│   │   │   └── ml_loader.py
│   │   │
│   │   ├── schemas
│   │   │   ├── patient_schema.py
│   │   │   └── history_schema.py
│   │   │
│   │   ├── services
│   │   │   └── analysis_service.py
│   │   │
│   │   └── main.py
│   │
│   ├── artifacts
│   │   ├── clinical_model.pkl
│   │   ├── speech_model.pkl
│   │   ├── gait_model.pkl
│   │   ├── progression_model.pkl
│   │   └── metadata / scalers / label encoders
│   │
│   ├── data
│   │   ├── demo_patients.csv
│   │   └── medical_knowledge
│   │       ├── pdf_sources
│   │       │   └── mds-clinical-diagnostic-criteria-for-parkinson-disease.pdf
│   │       ├── vector_store
│   │       │   ├── index.faiss
│   │       │   └── chunks.json
│   │       ├── clinical_safety_notes.txt
│   │       ├── parkinsons_gait_changes.txt
│   │       ├── parkinsons_motor_symptoms.txt
│   │       └── parkinsons_speech_changes.txt
│   │
│   ├── datasets
│   │   ├── clinical_agent_dataset.csv
│   │   ├── speech_agent_dataset.csv
│   │   ├── gait_agent_dataset.csv
│   │   ├── progression_agent_dataset.csv
│   │   └── generate_training_data.py
│   │
│   ├── scripts
│   │   ├── build_rag_index.py
│   │   └── train_models.py
│   │
│   ├── Dockerfile
│   ├── README.md
│   └── requirements.txt
│
├── frontend
│   ├── src
│   │   ├── App.css
│   │   ├── App.jsx
│   │   └── main.jsx
│   │
│   ├── Dockerfile
│   ├── index.html
│   ├── package.json
│   ├── package-lock.json
│   └── vite.config.js
│
├── docs
│   ├── DOMAIN_MAPPING.md
│   ├── MVP_Scope.md
│   ├── PATIENT_SCHEMA.MD
│   └── SYSTEM_ARCHITECTURE.md
│
├── .env.example
├── .gitignore
├── docker-compose.yml
└── README.md
```

---

## Prerequisites

Install the following before running the project:

- Git
- Docker
- Docker Compose
- Python 3.11+
- Node.js 20+
- PostgreSQL, if running without Docker
- pgAdmin 4, optional for database inspection

---

## Run with Docker

### 1. Clone the Repository

```bash
git clone https://github.com/mehedi77k/NeuroAgent_PD.git
cd NeuroAgent_PD
```

### 2. Create `.env`

Create a `.env` file in the project root.

```env
POSTGRES_DB=neuroagent
POSTGRES_USER=neuro_user
POSTGRES_PASSWORD=change_this_password

DATABASE_URL=postgresql+psycopg://neuro_user:change_this_password@db:5432/neuroagent

BACKEND_PORT=8000
FRONTEND_PORT=5173

LLM_ENABLED=false
GROQ_API_KEY=
GEMINI_API_KEY=
```

### 3. Start the Full Application

```bash
docker compose up --build
```

### 4. Build or Rebuild the RAG Index

If the vector store files already exist, the app can run directly.

If you add or replace PDFs, rebuild the RAG index:

```bash
docker compose run --rm backend python scripts/build_rag_index.py
```

### 5. Open the Application

Frontend:

```text
http://localhost:5173
```

Backend API:

```text
http://localhost:8000
```

Swagger API Docs:

```text
http://localhost:8000/docs
```

Health Check:

```text
http://localhost:8000/health
```

PostgreSQL host port:

```text
localhost:5435
```

Default Docker database values:

```text
Database: neuroagent
User: neuro_user
Password: change_this_password
Host: localhost
Port: 5435
```

---

## Run Manually

Use this method if you do not want to use Docker.

### 1. Clone the Repository

```bash
git clone https://github.com/mehedi77k/NeuroAgent_PD.git
cd NeuroAgent_PD
```

### 2. Create PostgreSQL Database

```sql
CREATE DATABASE neuroagent;
CREATE USER neuro_user WITH PASSWORD 'change_this_password';
GRANT ALL PRIVILEGES ON DATABASE neuroagent TO neuro_user;
```

### 3. Configure Environment Variables

Create a `.env` file in the project root.

```env
POSTGRES_DB=neuroagent
POSTGRES_USER=neuro_user
POSTGRES_PASSWORD=change_this_password

DATABASE_URL=postgresql+psycopg://neuro_user:change_this_password@localhost:5432/neuroagent

BACKEND_PORT=8000
FRONTEND_PORT=5173

LLM_ENABLED=false
GROQ_API_KEY=
GEMINI_API_KEY=
```

### 4. Install Backend Dependencies

```bash
cd backend
python -m venv .venv
```

Activate the virtual environment.

Windows:

```bash
.venv\Scripts\activate
```

Linux/macOS:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

### 5. Build RAG Index Manually

From the `backend` directory:

```bash
python scripts/build_rag_index.py
```

### 6. Run the Backend

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Backend URL:

```text
http://localhost:8000
```

Swagger Docs:

```text
http://localhost:8000/docs
```

### 7. Install Frontend Dependencies

Open a new terminal:

```bash
cd frontend
npm install
```

### 8. Run the Frontend

```bash
npm run dev
```

Frontend URL:

```text
http://localhost:5173
```

---

## Environment Variables

| Variable | Description | Example |
|---|---|---|
| `POSTGRES_DB` | PostgreSQL database name | `neuroagent` |
| `POSTGRES_USER` | PostgreSQL username | `neuro_user` |
| `POSTGRES_PASSWORD` | PostgreSQL password | `change_this_password` |
| `DATABASE_URL` | SQLAlchemy database connection URL | `postgresql+psycopg://neuro_user:change_this_password@db:5432/neuroagent` |
| `BACKEND_PORT` | Backend exposed port | `8000` |
| `FRONTEND_PORT` | Frontend exposed port | `5173` |
| `LLM_ENABLED` | Enables or disables LLM report generation | `false` |
| `GROQ_API_KEY` | Groq API key for LLM report generation | empty |
| `GEMINI_API_KEY` | Gemini API key for LLM report generation | empty |

Frontend API URL used by Docker Compose:

```env
VITE_API_URL=http://localhost:8000
```

---

## API Endpoints

Base URL:

```text
http://localhost:8000
```

### Health

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Backend root response |
| `GET` | `/health` | Backend health check |

### Patient Cases

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/patients` | List demo patient summaries |
| `GET` | `/patients?search=<query>` | Search patients by patient ID or name |
| `GET` | `/patients/{patient_id}` | Get full patient profile |
| `GET` | `/demo-patient` | Get first demo patient case |

### Analysis

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/analyze` | Run full multi-agent Parkinson’s analysis and save result |

### Analysis History

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/history` | Get latest saved analysis results |
| `GET` | `/history?limit=20` | Get limited saved analysis results |
| `GET` | `/history/{patient_id}` | Get analysis history for one patient |
| `GET` | `/history/detail/{analysis_id}` | Get full analysis detail by analysis ID |

### Doctor Feedback

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/feedback` | Submit doctor feedback for an analysis |

Allowed feedback actions:

```text
approve
reject
request_more_data
escalate_to_neurologist
```

---

## Patient Case Schema

Example request body for:

```text
POST /analyze
```

```json
{
  "patient_id": "PD-001",
  "patient_name": "Robert Chen",
  "age": 64,
  "gender": "male",
  "clinical": {
    "updrs_score": 38,
    "tremor_score": 3,
    "rigidity_score": 2,
    "bradykinesia_score": 3,
    "disease_duration_years": 4,
    "medication_response": "moderate"
  },
  "speech": {
    "jitter": 0.021,
    "shimmer": 0.034,
    "hnr": 17.4,
    "pitch_variation": 0.28
  },
  "gait": {
    "walking_speed": 0.82,
    "stride_variability": 0.31,
    "freezing_index": 0.42,
    "balance_score": 0.58
  },
  "notes": "Patient reports increasing hand tremor and walking instability."
}
```

---

## Example Analysis Response

The `/analyze` endpoint returns a full multi-agent result.

```json
{
  "patient": {
    "patient_id": "PD-001",
    "patient_name": "Robert Chen",
    "age": 64,
    "gender": "male",
    "clinical": {},
    "speech": {},
    "gait": {},
    "notes": "Patient reports increasing hand tremor and walking instability."
  },
  "agent_results": {
    "clinical": {},
    "speech": {},
    "gait": {},
    "coordinator": {},
    "triage": {},
    "conflict": {},
    "rag": {
      "agent_name": "rag_agent",
      "retrieval_method": "patient_specific_multi_query_pdf_vector_search_faiss",
      "knowledge_base": "MDS Clinical Diagnostic Criteria for Parkinson's Disease",
      "query_strategy": "patient_specific_multi_query",
      "evidence_found": true,
      "evidence_count": 4,
      "evidence": []
    },
    "progression": {},
    "dbs_referral": {},
    "explainability": {},
    "critic": {}
  },
  "report": {
    "patient_id": "PD-001",
    "patient_name": "Robert Chen",
    "summary": "...",
    "doctor_facing_explanation": "...",
    "medical_evidence_summary": "...",
    "progression_summary": "...",
    "dbs_referral_summary": "...",
    "llm_generated": false,
    "llm_report": "",
    "llm_provider": "none",
    "fallback_used": true
  },
  "analysis_id": 1
}
```

---

## Database Models

### `analysis_results`

Stores saved multi-agent analysis outputs.

| Field | Description |
|---|---|
| `id` | Primary key |
| `patient_id` | Patient identifier |
| `patient_name` | Patient name |
| `final_risk_score` | Final coordinated risk score |
| `final_risk_level` | Final coordinated risk level |
| `final_prediction` | Final prediction label |
| `triage_level` | Triage recommendation level |
| `priority` | Priority level |
| `full_analysis_json` | Complete analysis result as JSON |
| `created_at` | Analysis timestamp |

### `doctor_feedback`

Stores doctor review actions and comments.

| Field | Description |
|---|---|
| `id` | Primary key |
| `patient_id` | Patient identifier |
| `analysis_id` | Linked analysis result ID |
| `action` | Doctor feedback action |
| `comment` | Optional doctor comment |
| `created_at` | Feedback timestamp |

---

## Machine Learning Training Pipeline

The project includes a model training script:

```text
backend/scripts/train_models.py
```

The training pipeline trains ML models for:

- Clinical Agent
- Speech Agent
- Gait Agent
- Progression Agent

Expected dataset directory:

```text
backend/datasets
```

Expected dataset files:

```text
clinical_agent_dataset.csv
speech_agent_dataset.csv
gait_agent_dataset.csv
progression_agent_dataset.csv
```

Run training from the project root:

```bash
python backend/scripts/train_models.py
```

Generated model artifacts are saved to:

```text
backend/artifacts
```

Expected output artifacts include:

```text
clinical_model.pkl
clinical_scaler.pkl
clinical_label_encoder.pkl
clinical_metadata.json

speech_model.pkl
speech_scaler.pkl
speech_label_encoder.pkl
speech_metadata.json

gait_model.pkl
gait_scaler.pkl
gait_label_encoder.pkl
gait_metadata.json

progression_model.pkl
progression_scaler.pkl
progression_label_encoder.pkl
progression_metadata.json
```

Important:

```text
The included datasets should be treated as synthetic or experimental unless replaced with validated clinical datasets.
The trained models are not clinically validated and must not be used for final diagnosis.
```

---

## LLM Report Generation

The report generator can optionally create a doctor-facing clinical report using an external LLM.

Supported providers:

- Groq
- Gemini

Enable LLM report generation:

```env
LLM_ENABLED=true
GROQ_API_KEY=your_groq_api_key
GEMINI_API_KEY=your_gemini_api_key
```

Provider behavior:

```text
1. Try Groq if GROQ_API_KEY is configured.
2. Try Gemini if GEMINI_API_KEY is configured.
3. Use the fallback template report if no LLM provider is available or the LLM call fails.
```

Default behavior:

```env
LLM_ENABLED=false
```

This keeps the system usable without external API keys.

---

## Clinical Safety Notes

NeuroAgent_PD must be treated as a prototype.

The system should not be used for:

- Final diagnosis
- Medication prescription
- Treatment adjustment
- Surgical recommendation
- Emergency triage without physician review
- Replacing neurologist assessment
- Real patient deployment without privacy, safety, and regulatory review

The system can be used for:

- Academic demonstration
- AI healthcare workflow prototyping
- Multi-agent system experimentation
- Clinical decision-support UI research
- Synthetic dataset model-training experiments
- Explainability and risk-scoring experiments
- RAG-based medical evidence retrieval experiments

Required human oversight:

```text
All moderate-risk, high-risk, conflict-detected, DBS-flagged, low-confidence, or safety-warning outputs require qualified clinician review.
```

---

## Development Commands

### Docker

Start all services:

```bash
docker compose up --build
```

Start without rebuilding:

```bash
docker compose up
```

Stop services:

```bash
docker compose down
```

Stop services and remove database volume:

```bash
docker compose down -v
```

View all logs:

```bash
docker compose logs -f
```

View backend logs:

```bash
docker compose logs -f backend
```

View frontend logs:

```bash
docker compose logs -f frontend
```

Check running containers:

```bash
docker compose ps
```

Restart backend:

```bash
docker compose restart backend
```

Restart frontend:

```bash
docker compose restart frontend
```

Build backend only:

```bash
docker compose build --no-cache backend
```

### Backend

Run backend manually:

```bash
cd backend
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Install dependencies:

```bash
cd backend
pip install -r requirements.txt
```

Train models:

```bash
python backend/scripts/train_models.py
```

Build RAG index with Docker:

```bash
docker compose run --rm backend python scripts/build_rag_index.py
```

Build RAG index manually from backend folder:

```bash
python scripts/build_rag_index.py
```

### Frontend

Install dependencies:

```bash
cd frontend
npm install
```

Run development server:

```bash
npm run dev
```

Build production frontend:

```bash
npm run build
```

Preview production build:

```bash
npm run preview
```

---

## Troubleshooting

### Backend Cannot Connect to PostgreSQL

Check running containers:

```bash
docker compose ps
```

Check database logs:

```bash
docker compose logs db
```

Inside Docker, the database host must be:

```text
db
```

Docker database URL:

```env
DATABASE_URL=postgresql+psycopg://neuro_user:change_this_password@db:5432/neuroagent
```

Manual local database URL:

```env
DATABASE_URL=postgresql+psycopg://neuro_user:change_this_password@localhost:5432/neuroagent
```

---

### Frontend Cannot Connect to Backend

Check backend health:

```text
http://localhost:8000/health
```

Check frontend API URL:

```env
VITE_API_URL=http://localhost:8000
```

Restart frontend after changing environment variables:

```bash
docker compose restart frontend
```

or manually:

```bash
npm run dev
```

---

### Swagger Docs Not Opening

Check backend logs:

```bash
docker compose logs backend --tail=100
```

Then open:

```text
http://localhost:8000/docs
```

---

### RAG Evidence Is Not Showing

Check that the vector store exists:

```text
backend/data/medical_knowledge/vector_store/index.faiss
backend/data/medical_knowledge/vector_store/chunks.json
```

If missing, rebuild:

```bash
docker compose run --rm backend python scripts/build_rag_index.py
```

Then restart backend:

```bash
docker compose restart backend
```

---

### RAG Analysis Is Slow

The first RAG-enabled analysis may be slower because the sentence-transformer model is loaded into memory.

Recommended checks:

```bash
docker compose logs -f backend
```

```bash
docker stats
```

If the backend is CPU-limited, first request latency is expected. Later requests should usually be faster after the model is loaded.

---

### FAISS Import Error

If this error appears:

```text
ModuleNotFoundError: No module named 'faiss'
```

Rebuild the backend image:

```bash
docker compose down
docker compose build --no-cache backend
docker compose up
```

Make sure `backend/requirements.txt` contains:

```text
faiss-cpu
sentence-transformers
pypdf
```

---

### Docker Build Fails During apt-get

The backend Dockerfile should not require `apt-get` for the current MVP setup.

If Docker build fails while trying to access Debian package mirrors, check that `backend/Dockerfile` does not contain:

```text
apt-get
build-essential
curl
```

---

### ML Prediction Is Not Showing

The system uses ML prediction only when trained artifacts exist in:

```text
backend/artifacts
```

Train the models:

```bash
python backend/scripts/train_models.py
```

Make sure these files exist:

```text
backend/datasets/clinical_agent_dataset.csv
backend/datasets/speech_agent_dataset.csv
backend/datasets/gait_agent_dataset.csv
backend/datasets/progression_agent_dataset.csv
```

If artifacts are missing, the agents use rule-based scoring.

---

### LLM Report Is Empty

Check `.env`:

```env
LLM_ENABLED=true
GROQ_API_KEY=your_key
```

or:

```env
LLM_ENABLED=true
GEMINI_API_KEY=your_key
```

If no API key is configured, the report generator uses fallback mode.

---

### Database Tables Are Missing

The backend initializes tables on startup.

Restart backend:

```bash
docker compose restart backend
```

or manually restart Uvicorn:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

---

### Port Already in Use

Default ports:

```text
Backend:    8000
Frontend:   5173
PostgreSQL: 5435
```

If another project is using the same ports, update the port mappings in `docker-compose.yml`.

---

## Repository

```text
https://github.com/mehedi77k/NeuroAgent_PD
```

---

## Project Status

```text
Status: MVP Prototype
Project Type: Parkinson’s Disease Multi-Agent Clinical Decision-Support Platform
Frontend: React + Vite
Backend: FastAPI
Database: PostgreSQL
ORM: SQLAlchemy
ML Support: scikit-learn + joblib artifacts
RAG Support: PDF + sentence-transformers + FAISS
LLM Support: Optional Groq / Gemini report generation
Containerization: Docker Compose
Primary Use: Academic, research, and prototype demonstration
Medical Use: Decision-support only, not diagnosis
Current API Version: 0.4.0
```

---

## Disclaimer

NeuroAgent_PD is provided for academic and research purposes only. It is not a medical device and is not intended to diagnose, treat, cure, or prevent any disease. The predictions, recommendations, explanations, retrieved evidence, and reports generated by this system must be reviewed by qualified healthcare professionals before any clinical interpretation or action.