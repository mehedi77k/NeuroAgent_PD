# NeuroAgent-PD MVP Scope

## Project Title

NeuroAgent-PD: An Autonomous Multi-Agent Clinical Decision-Support System for Explainable Parkinson’s Disease Analysis

## One-Line Pitch

NeuroAgent-PD is an autonomous multi-agent healthcare AI system that analyzes clinical, speech, and gait data to support Parkinson’s disease risk assessment, severity estimation, autonomous triage, and explainable neurologist-facing reports.

## Main User

The main user is a neurologist or hospital doctor.

## MVP Goal

The MVP demonstrates how multiple AI agents can analyze different Parkinson’s-related data sources, collaborate through a coordinator agent, assign triage priority, and generate an explainable doctor-facing report.

## Included in MVP

- Patient case selection or upload
- Clinical Data Agent
- Speech Feature Agent
- Gait Feature Agent
- Medical Knowledge RAG Agent
- Coordinator Agent
- Triage Agent
- Critic / Safety Agent
- Explainability Agent
- Doctor dashboard
- Doctor feedback buttons
- Simulated progression timeline

## Not Included in MVP

- Final medical diagnosis
- Prescription generation
- Real hospital integration
- Robotic surgery control
- Real-time wearable device connection
- Full MRI deep learning pipeline

## Competition Domain Mapping

| Domain | MVP Feature |
|---|---|
| Healthcare AI Agents | Multiple specialized clinical AI agents |
| Multi-Agent Collaboration | Coordinator combines agent outputs |
| Autonomous Triage | Risk-based triage priority |
| Robotic Surgery Agents | DBS referral support only |
| Agent-Based Simulation | Simulated progression timeline |
| Intelligent Automation | Auto report and follow-up checklist |
| Human-Agent Interaction | Doctor dashboard and feedback |
| Autonomous Coordination | Coordinator routes and validates decisions |

## MVP Workflow

1. Doctor selects patient case.
2. Clinical Agent analyzes clinical data.
3. Speech Agent analyzes speech features.
4. Gait Agent analyzes movement features.
5. RAG Agent retrieves Parkinson evidence.
6. Coordinator Agent combines outputs.
7. Triage Agent assigns priority.
8. Critic Agent checks safety and confidence.
9. Explainability Agent generates report.
10. Doctor reviews final output.

## Safety Statement

This system is a clinical decision-support prototype. It does not replace neurologists and does not provide final diagnosis, prescription, or treatment decisions.