# NeuroAgent_PD: Multi-Agent Parkinson’s Disease Clinical Decision-Support Platform

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-316192?style=for-the-badge&logo=postgresql&logoColor=white)
![SQLAlchemy](https://img.shields.io/badge/SQLAlchemy-CB1B1B?style=for-the-badge)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)
![React](https://img.shields.io/badge/React-20232A?style=for-the-badge&logo=react&logoColor=61DAFB)
![Vite](https://img.shields.io/badge/Vite-646CFF?style=for-the-badge&logo=vite&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white)
![Status](https://img.shields.io/badge/Status-MVP%20Prototype-success?style=for-the-badge)

**NeuroAgent_PD** is a full-stack multi-agent clinical decision-support prototype for Parkinson’s disease assessment. It combines structured clinical data, speech indicators, gait indicators, coordinated risk scoring, triage support, conflict detection, medical evidence retrieval, progression simulation, DBS referral support, explainability, critic review, doctor feedback, and LLM-assisted report generation into a single doctor-facing web platform.

> **Medical Safety Notice**  
> NeuroAgent_PD is a research and academic prototype. It does **not** provide a final medical diagnosis, treatment decision, prescription, or replacement for a qualified neurologist. All outputs must be reviewed by a licensed medical professional before any clinical decision is made.

---

## Table of Contents

- [Overview](#overview)
- [Core Features](#core-features)
- [Agent System](#agent-system)
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

---

## Overview

NeuroAgent_PD is designed as a doctor-facing Parkinson’s disease clinical decision-support dashboard.

The system allows clinicians, researchers, or evaluators to:

- View demo Parkinson’s patient cases
- Search patients by ID or name
- Inspect structured patient profile data
- Analyze clinical, speech, and gait features
- Run a full multi-agent Parkinson’s risk assessment
- Generate coordinated final risk prediction
- Review clinical triage recommendation
- Detect disagreement between agent outputs
- Retrieve relevant medical evidence
- Simulate 12-month progression risk
- Evaluate DBS referral discussion indicators
- View feature-level explainability
- Review automated critic warnings
- Generate doctor-facing clinical reports
- Save analysis results to PostgreSQL
- Submit doctor feedback for each analysis

The platform uses a modular architecture:

```text
Frontend Dashboard → FastAPI Backend → Multi-Agent AI Layer → PostgreSQL Storage
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
- DBS Referral Support Agent
- Explainability Agent
- Critic Agent
- LLM Report Generator

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

The React frontend provides a professional dashboard experience for:

- Selecting patient cases
- Running multi-agent analysis
- Viewing individual agent outputs
- Viewing final coordinated risk
- Reviewing triage recommendation
- Reading generated clinical report
- Checking medical evidence
- Reviewing progression simulation
- Checking DBS referral indicators
- Reviewing explainability output
- Submitting doctor feedback
- Viewing saved analysis history

### Analysis History

Each analysis can be stored in PostgreSQL with:

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

Retrieves relevant medical evidence from local medical knowledge files.

Evidence topics include:

- Parkinson’s clinical symptoms
- Motor symptom patterns
- Speech changes
- Gait changes
- Safety notes for clinical interpretation

Outputs:

- Evidence count
- Retrieved evidence items
- Source file
- Matched reason
- Safety note

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

### 9. DBS Referral Support Agent

Evaluates whether the patient profile may justify discussion with a DBS specialist.

Criteria checked:

- Disease duration of 5+ years
- Poor or limited medication response
- Severe motor symptoms
- Significant gait freezing
- Moderate or high coordinated risk
- Age under 75

Outputs:

- Referral recommended or not
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
- DBS referral signal
- Need for human review

Outputs:

- Warnings
- Safety notes
- Human-review requirement
- Clinical caution messages

---

### 12. LLM Report Generator

Generates an optional doctor-facing clinical report using an external LLM.

Supported providers:

- Groq
- Gemini

If LLM generation is disabled or API keys are missing, the system falls back to the template-based report.

---

## Technology Stack

| Layer | Technologies |
|---|---|
| Frontend | React, Vite, JavaScript, CSS |
| Backend | Python, FastAPI, Uvicorn |
| Database | PostgreSQL |
| ORM | SQLAlchemy |
| Validation | Pydantic |
| ML / Data | pandas, numpy, scikit-learn, joblib |
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
│  - Select patient case                                               |
│  - Run Parkinson’s multi-agent analysis                              |
│  - Review risk, triage, evidence, explainability, and report         |
│  - Submit feedback                                                   |
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
│  - DBS referral support                                               │
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
│        ├── DBS Referral Support Agent                                 │
│        ├── Explainability Agent                                       │
│        ├── Critic Agent                                               │
│        └── LLM Report Generator                                       │
└──────────────────────────────────────┬───────────────────────────────┘
                                       │
                                       ▼
┌──────────────────────────────────────────────────────────────────────┐
│                         Data Persistence Layer                       │
│                                                                      │
│                              PostgreSQL                              │
│                                                                      │
│  Tables:                                                              │
│  - analysis_results                                                   │
│  - doctor_feedback                                                    │
└──────────────────────────────────────────────────────────────────────┘
```

---

## Project Structure

```text
NeuroAgent_AI
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
│   │   │   └── demo_patients.py
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
│   │   └── trained ML artifacts
│   │
│   ├── data
│   │   ├── demo_patients.csv
│   │   └── medical_knowledge
│   │
│   ├── datasets
│   │   ├── clinical_agent_dataset.csv
│   │   ├── speech_agent_dataset.csv
│   │   ├── gait_agent_dataset.csv
│   │   ├── progression_agent_dataset.csv
│   │   └── generate_training_data.py
│   │
│   ├── scripts
│   │   └── train_models.py
│   │
│   ├── Dockerfile
│   ├── README.md
│   └── requirements.txt
│
├── frontend
│   ├── src
│   │   ├── services
│   │   │   └── api.js
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
git clone https://github.com/mehedi77k/NeuroAgent_AI.git
cd NeuroAgent_AI
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

### 4. Open the Application

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
git clone https://github.com/mehedi77k/NeuroAgent_AI.git
cd NeuroAgent_AI
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

### 5. Run the Backend

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

### 6. Install Frontend Dependencies

Open a new terminal:

```bash
cd frontend
npm install
```

### 7. Run the Frontend

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
    "rag": {},
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

NeuroAgent-PD must be treated as a prototype.

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

## Known Limitations

- This is an MVP prototype, not a clinically validated medical system.
- The demo patient data is limited.
- Included training datasets are synthetic or experimental unless replaced.
- Current ML models require external clinical validation before any real-world use.
- Rule-based scoring formulas are simplified.
- No user authentication is currently implemented.
- No doctor/admin role-based access control is currently implemented.
- No formal database migration tool such as Alembic is currently configured.
- Current evidence retrieval uses local medical knowledge files.
- LLM reports depend on external API providers when enabled.
- No production-grade patient privacy layer is currently implemented.
- No audit logging is currently implemented.
- No automated test suite is currently included.
- No real EHR/FHIR integration is currently implemented.
- The system is not cleared for clinical deployment.

---

## Future Improvements

- Add authentication and authorization
- Add doctor/admin roles
- Add protected API routes
- Add Alembic database migrations
- Add patient data privacy controls
- Add audit logs for doctor actions
- Add model versioning
- Add dataset versioning
- Add SHAP or LIME explainability for trained ML models
- Add vector-based medical evidence retrieval
- Add citations for retrieved medical evidence
- Add PDF report export
- Add patient report download
- Add CSV upload for batch analysis
- Add real-time dashboard analytics
- Add model performance monitoring
- Add clinical validation workflow
- Add clinician approval workflow
- Add notification system
- Add deployment configuration for cloud hosting
- Add CI/CD with GitHub Actions
- Add backend unit tests
- Add frontend component tests
- Add end-to-end tests
- Add Docker production profile
- Add HTTPS reverse proxy setup
- Add environment-specific configuration
- Add FHIR or EHR integration layer
- Add privacy-preserving storage for real patient records
- Add configurable hospital-specific clinical workflows

---

## Repository

```text
https://github.com/mehedi77k/NeuroAgent_AI
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
LLM Support: Optional Groq / Gemini report generation
Containerization: Docker Compose
Primary Use: Academic, research, and prototype demonstration
Medical Use: Decision-support only, not diagnosis
Current API Version: 0.4.0
```

---

## Disclaimer

NeuroAgent-PD is provided for academic and research purposes only. It is not a medical device and is not intended to diagnose, treat, cure, or prevent any disease. The predictions, recommendations, explanations, and reports generated by this system must be reviewed by qualified healthcare professionals before any clinical interpretation or action.