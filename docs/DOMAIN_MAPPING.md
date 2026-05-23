# Competition Domain Mapping

## 1. Healthcare AI Agents

The system contains multiple specialized healthcare AI agents:

- Clinical Data Agent
- Speech Analysis Agent
- Gait Analysis Agent
- RAG Medical Knowledge Agent
- Coordinator Agent
- Critic / Safety Agent
- Explainability Agent

## 2. Multi-Agent Collaboration

Each agent analyzes one part of the patient case. The Coordinator Agent combines all outputs into a final risk score, severity level, and triage decision.

## 3. Autonomous Triage

The Triage Agent classifies patients into:

- Routine monitoring
- Priority neurologist review
- Urgent neurologist review

## 4. Robotic Surgery Agents

The MVP includes a DBS Referral Support Agent. It does not control surgery. It only flags severe cases that may need specialist discussion.

## 5. Agent-Based Simulation

The system simulates Parkinson’s progression over 12 months using synthetic patient timelines.

## 6. Intelligent Automation

The system automatically generates:

- Risk summary
- Triage decision
- Doctor-facing report
- Follow-up checklist
- Safety warning

## 7. Human-Agent Interaction

Doctors can:

- View agent outputs
- Review explanations
- Approve report
- Reject report
- Request more data
- Add feedback

## 8. Autonomous Coordination

The Coordinator Agent decides:

- Which agents contributed
- Whether agents agree or disagree
- Whether confidence is high enough
- Whether human review is required