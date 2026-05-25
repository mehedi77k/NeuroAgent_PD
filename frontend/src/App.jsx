import React, { useEffect, useMemo, useState } from "react";
import "./App.css";
import {
  analyzePatient,
  getDemoPatient,
  getHistory,
  getPatientById,
  getPatientHistory,
  getPatients,
  submitDoctorFeedback,
} from "./services/api";

const toPercent = (value) => {
  if (typeof value !== "number") return 0;
  return Math.round(value * 100);
};

const formatLabel = (value) => {
  if (!value) return "N/A";

  return String(value)
    .split("_")
    .map((word) => word.charAt(0).toUpperCase() + word.slice(1))
    .join(" ");
};

const safeValue = (value) => {
  if (value === undefined || value === null || value === "") return "N/A";
  return value;
};

const formatDateTime = (value) => {
  if (!value) return "N/A";

  const date = new Date(value);

  if (Number.isNaN(date.getTime())) {
    return safeValue(value);
  }

  return date.toLocaleString();
};

const getPriorityClass = (priority) => {
  const value = String(priority || "").toLowerCase();

  if (value === "high") return "status-red";
  if (value === "medium") return "status-amber";
  if (value === "low") return "status-green";

  return "status-neutral";
};

const getRiskClass = (score) => {
  if (typeof score !== "number") return "risk-low";

  if (score >= 0.7) return "risk-high";
  if (score >= 0.4) return "risk-med";

  return "risk-low";
};

const getConflictClass = (level) => {
  const value = String(level || "").toLowerCase();

  if (value === "high") return "conflict-high";
  if (value === "moderate") return "conflict-moderate";
  if (value === "low") return "conflict-low";
  if (value === "none") return "conflict-none";

  return "conflict-neutral";
};

const getFeatureUnit = (key) => {
  const units = {
    walking_speed: " m/s",
    hnr: " dB",
  };

  return units[key] || "";
};

const getPageTitle = (activePage) => {
  const titles = {
    "patient-case": "Patient Case",
    "agent-analysis": "Agent Analysis",
    "medical-evidence": "Medical Evidence",
    reports: "Reports",
    settings: "Settings",
  };

  return titles[activePage] || "Patient Case";
};

const getPageSubtitle = (activePage) => {
  const subtitles = {
    "patient-case": "Load and review the current Parkinson’s patient case.",
    "agent-analysis": "Review clinical, speech, gait, triage, and safety agent outputs.",
    "medical-evidence": "Review retrieved MDS diagnostic evidence from the RAG knowledge base.",
    reports: "Review the generated doctor-facing clinical summary.",
    settings: "MVP configuration and project status.",
  };

  return subtitles[activePage] || "";
};

const REQUIRED_SAFETY_NOTE =
  "This is clinical decision support only. Not a final medical diagnosis.";

const AgentCard = ({ title, agentData, features, featureData }) => {
  if (!agentData) return null;

  return (
    <div className="card agent-card">
      <div className="agent-card-header">
        <div>
          <h3>{title}</h3>
          <p className="agent-subtitle">{formatLabel(agentData.prediction)}</p>
        </div>

        <span className={`badge ${getRiskClass(agentData.risk_score)}`}>
          {toPercent(agentData.risk_score)}% Risk
        </span>
      </div>

      <div className="agent-stats">
        <div className="stat-item">
          <label>Confidence</label>
          <span>{toPercent(agentData.confidence)}%</span>
        </div>

        <div className="stat-item">
          <label>Severity</label>
          <span>{formatLabel(agentData.severity)}</span>
        </div>

        <div className="stat-item">
          <label>Risk Level</label>
          <span className="risk-level-text">
            {safeValue(formatLabel(agentData.risk_level))}
          </span>
        </div>

        {agentData.ml_available && (
          <div className="stat-item ml-stat">
            <label>ML Prediction</label>
            <span>{formatLabel(agentData.ml_prediction)}</span>
          </div>
        )}
      </div>

      {agentData.ml_available && (
        <div className="ml-prediction-bar">
          <div className="ml-prediction-header">
            <span className="ml-badge">ML Model</span>
            <span className="ml-confidence">
              {toPercent(agentData.ml_confidence)}% confidence
            </span>
          </div>

          {agentData.ml_class_probabilities && (
            <div className="ml-probabilities">
              {Object.entries(agentData.ml_class_probabilities).map(
                ([cls, prob]) => (
                  <div key={cls} className="ml-prob-item">
                    <span className="ml-prob-label">{formatLabel(cls)}</span>
                    <div className="ml-prob-bar-track">
                      <div
                        className="ml-prob-bar-fill"
                        style={{ width: `${Math.round(prob * 100)}%` }}
                      />
                    </div>
                    <span className="ml-prob-value">
                      {Math.round(prob * 100)}%
                    </span>
                  </div>
                ),
              )}
            </div>
          )}
        </div>
      )}

      <p className="agent-explanation">{safeValue(agentData.explanation)}</p>

      <div className="feature-badges">
        {agentData.top_features?.map((feature) => (
          <span key={feature} className="feature-badge">
            {formatLabel(feature)}
          </span>
        ))}
      </div>

      <div className="feature-details">
        {features.map((feature) => (
          <div key={feature.key} className="feature-row">
            <span className="feature-label">{feature.label}</span>
            <span className="feature-value">
              {safeValue(featureData?.[feature.key])}
              {featureData?.[feature.key] !== undefined
                ? getFeatureUnit(feature.key)
                : ""}
            </span>
          </div>
        ))}
      </div>
    </div>
  );
};

const ConflictPanel = ({ conflict }) => {
  if (!conflict) return null;

  const detected = Boolean(conflict.conflict_detected);
  const riskGap =
    typeof conflict.risk_gap === "number"
      ? conflict.risk_gap.toFixed(2)
      : safeValue(conflict.risk_gap);

  return (
    <div
      className={`card conflict-panel ${getConflictClass(
        conflict.conflict_level,
      )}`}
    >
      <div className="conflict-header">
        <h3>Agent Conflict Analysis</h3>
        <span className="conflict-status-badge">
          {detected ? "Conflict Detected" : "No Major Conflict"}
        </span>
      </div>

      <div className="conflict-type">
        {safeValue(formatLabel(conflict.conflict_type))}
      </div>

      <p className="conflict-summary">{safeValue(conflict.explanation)}</p>

      <div className="conflict-details">
        <div>
          <label>Conflict Detected</label>
          <strong>{detected ? "Yes" : "No"}</strong>
        </div>

        <div>
          <label>Conflict Level</label>
          <strong>{safeValue(formatLabel(conflict.conflict_level))}</strong>
        </div>

        <div>
          <label>Risk Gap</label>
          <strong>{riskGap}</strong>
        </div>

        <div>
          <label>Highest Risk Agent</label>
          <strong>{safeValue(formatLabel(conflict.highest_risk_agent))}</strong>
        </div>

        <div>
          <label>Lowest Risk Agent</label>
          <strong>{safeValue(formatLabel(conflict.lowest_risk_agent))}</strong>
        </div>
      </div>

      <div className="conflict-recommendation">
        <label>Recommendation</label>
        <p>{safeValue(conflict.recommendation)}</p>
      </div>
    </div>
  );
};

const EvidenceCards = ({ evidence }) => {
  const items = Array.isArray(evidence) ? evidence : [];

  if (items.length === 0) {
    return (
      <div className="empty-state compact">
        <h3>No evidence retrieved</h3>
        <p>
          Run analysis again after confirming that the RAG index exists and the
          backend response contains agent_results.rag.evidence.
        </p>
      </div>
    );
  }

  return (
    <div className="evidence-list">
      {items.map((item, index) => {
        const evidenceKey = item?.chunk_id || item?.source || `evidence-${index}`;
        const similarity =
          typeof item?.similarity_score === "number"
            ? item.similarity_score.toFixed(3)
            : safeValue(item?.similarity_score);

        return (
          <div className="evidence-card" key={evidenceKey}>
            <div className="evidence-card-header">
              <div>
                <p className="eyebrow">Evidence #{index + 1}</p>
                <h3>{safeValue(item?.topic)}</h3>
              </div>

              <span className="evidence-score">Similarity {similarity}</span>
            </div>

            <div className="evidence-meta-grid">
              <div>
                <span>Source</span>
                <strong>{safeValue(item?.source)}</strong>
              </div>

              <div>
                <span>Page</span>
                <strong>{safeValue(item?.page)}</strong>
              </div>

              <div>
                <span>Chunk ID</span>
                <strong>{safeValue(item?.chunk_id)}</strong>
              </div>
            </div>

            <div className="evidence-reason">
              <strong>Matched Reason</strong>
              <p>{safeValue(item?.matched_reason)}</p>
            </div>

            {item?.search_query && (
              <div className="evidence-reason">
                <strong>Patient-Specific Search Query</strong>
                <p>{item.search_query}</p>
              </div>
            )}

            <div className="evidence-text">
              <strong>Retrieved Evidence</strong>
              <p>{safeValue(item?.text)}</p>
            </div>
          </div>
        );
      })}
    </div>
  );
};

const MedicalEvidencePanel = ({ rag, summary }) => {
  if (!rag) {
    return (
      <div className="card">
        <h3>Medical Evidence Agent</h3>
        <p>No RAG evidence is available. Run agent analysis first.</p>
      </div>
    );
  }

  const evidenceFoundLabel =
    rag?.evidence_found === true
      ? "Yes"
      : rag?.evidence_found === false
        ? "No"
        : "N/A";

  const evidenceCount = rag?.evidence_count ?? rag?.evidence?.length ?? 0;

  const safetyNote =
    rag?.safety_note && rag?.safety_note !== REQUIRED_SAFETY_NOTE
      ? `${rag.safety_note} ${REQUIRED_SAFETY_NOTE}`
      : rag?.safety_note || REQUIRED_SAFETY_NOTE;

  return (
    <div className="medical-evidence-page">
      <div className="card evidence-overview-card">
        <div className="section-header">
          <div>
            <p className="eyebrow">RAG Agent</p>
            <h3>Medical Evidence Review</h3>
          </div>

          <span className="status-pill status-green">
            {safeValue(rag?.retrieval_method)}
          </span>
        </div>

        <div className="evidence-summary-grid">
          <div>
            <span>Knowledge Base</span>
            <strong>{safeValue(rag?.knowledge_base)}</strong>
          </div>

          <div>
            <span>Evidence Found</span>
            <strong>{evidenceFoundLabel}</strong>
          </div>

          <div>
            <span>Evidence Count</span>
            <strong>{safeValue(evidenceCount)}</strong>
          </div>
        </div>

        {summary && (
          <div className="evidence-summary-text">
            <strong>Report Evidence Summary</strong>
            <p>{safeValue(summary)}</p>
          </div>
        )}

        <div className="safety-note-box">
          <strong>Safety Note</strong>
          <p>{safeValue(safetyNote)}</p>
        </div>
      </div>

      <EvidenceCards evidence={rag?.evidence} />
    </div>
  );
};

const ProgressionPanel = ({ progression }) => {
  if (!progression) return null;

  const timeline = Array.isArray(progression.timeline)
    ? progression.timeline
    : [];

  return (
    <div className="card progression-panel">
      <div className="progression-header">
        <div>
          <h3>Progression Simulation Agent</h3>
          <p>Simulated 12-month risk trajectory for clinical review.</p>
        </div>

        <span
          className={`progression-badge ${getRiskClass(
            progression.projected_12_month_risk_score,
          )}`}
        >
          {toPercent(progression.projected_12_month_risk_score)}% Projected
        </span>
      </div>

      <div className="progression-summary-grid">
        <div className="progression-summary-item">
          <label>Baseline Risk</label>
          <span>{toPercent(progression.baseline_risk_score)}%</span>
        </div>

        <div className="progression-summary-item">
          <label>Baseline Level</label>
          <span>{formatLabel(progression.baseline_risk_level)}</span>
        </div>

        <div className="progression-summary-item">
          <label>12-Month Risk</label>
          <span>{toPercent(progression.projected_12_month_risk_score)}%</span>
        </div>

        <div className="progression-summary-item">
          <label>12-Month Level</label>
          <span>{formatLabel(progression.projected_12_month_risk_level)}</span>
        </div>
      </div>

      {timeline.length > 0 && (
        <div className="progression-timeline">
          {timeline.map((item) => (
            <div
              key={item.month}
              className={`timeline-card ${getRiskClass(item.risk_score)}`}
            >
              <span className="timeline-month">Month {item.month}</span>
              <strong>{toPercent(item.risk_score)}%</strong>
              <small>{formatLabel(item.risk_level)}</small>
            </div>
          ))}
        </div>
      )}

      {progression.factors_considered?.length > 0 && (
        <div className="progression-factors">
          <label>Factors Considered</label>

          <div>
            {progression.factors_considered.map((factor) => (
              <span key={factor} className="progression-factor-badge">
                {formatLabel(factor)}
              </span>
            ))}
          </div>
        </div>
      )}

      <p className="progression-explanation">
        {safeValue(progression.explanation)}
      </p>

      <div className="progression-safety-note">
        {safeValue(progression.safety_note)}
      </div>
    </div>
  );
};

const DBSReferralPanel = ({ dbsReferral }) => {
  if (!dbsReferral) return null;

  const referralLevel = dbsReferral.referral_level || "not_indicated";
  const referralClass = `referral-${referralLevel.replace("_", "-")}`;

  const criteriaDetails = dbsReferral.criteria_details || {};
  const criteriaKeys = Object.keys(criteriaDetails);

  return (
    <div className={`card dbs-referral-panel ${referralClass}`}>
      <div className="dbs-header">
        <div>
          <h3>DBS Referral Support Agent</h3>
          <p>Deep Brain Stimulation candidacy evaluation for specialist discussion.</p>
        </div>

        <span className={`dbs-referral-badge ${referralClass}`}>
          {formatLabel(referralLevel)}
        </span>
      </div>

      <div className="dbs-summary-grid">
        <div className="dbs-summary-item">
          <label>Referral Recommended</label>
          <span>{dbsReferral.referral_recommended ? "Yes" : "No"}</span>
        </div>

        <div className="dbs-summary-item">
          <label>Referral Level</label>
          <span>{formatLabel(referralLevel)}</span>
        </div>

        <div className="dbs-summary-item">
          <label>Referral Score</label>
          <span>{toPercent(dbsReferral.referral_score)}%</span>
        </div>

        <div className="dbs-summary-item">
          <label>Criteria Met</label>
          <span>
            {dbsReferral.criteria_met?.length || 0} of{" "}
            {(dbsReferral.criteria_met?.length || 0) +
              (dbsReferral.criteria_not_met?.length || 0)}
          </span>
        </div>
      </div>

      {criteriaKeys.length > 0 && (
        <div className="dbs-criteria-grid">
          {criteriaKeys.map((key) => {
            const criterion = criteriaDetails[key];
            const isMet = criterion?.met;

            return (
              <div
                key={key}
                className={`dbs-criterion ${isMet ? "met" : "not-met"}`}
              >
                <span className="dbs-criterion-icon">
                  {isMet ? "✓" : "✗"}
                </span>
                <span className="dbs-criterion-label">
                  {criterion?.label || formatLabel(key)}
                </span>
              </div>
            );
          })}
        </div>
      )}

      <div className="dbs-recommendation">
        <label>Recommendation</label>
        <p>{safeValue(dbsReferral.recommendation)}</p>
      </div>

      <p className="dbs-explanation">{safeValue(dbsReferral.explanation)}</p>

      <div className="dbs-safety-note">
        {safeValue(dbsReferral.safety_note)}
      </div>
    </div>
  );
};

const getBarColorClass = (percentage) => {
  if (percentage >= 40) return "high";
  if (percentage >= 25) return "med";
  return "low";
};

const ExplainabilityPanel = ({ explainability }) => {
  if (!explainability) return null;

  const agents = [
    { key: "clinical", title: "Clinical Agent" },
    { key: "speech", title: "Speech Agent" },
    { key: "gait", title: "Gait Agent" },
  ];

  const coordinator = explainability.coordinator;

  return (
    <div className="card explainability-panel">
      <div className="explainability-header">
        <div>
          <h3>Explainability Agent</h3>
          <p className="agent-subtitle">
            Feature contribution breakdown for each domain agent.
          </p>
        </div>

        <span className="explainability-method-badge">
          {formatLabel(explainability.method)}
        </span>
      </div>

      <div className="explainability-agents-grid">
        {agents.map(({ key, title }) => {
          const agentData = explainability[key];
          if (!agentData) return null;

          return (
            <div key={key} className="explain-agent-card">
              <h4>{title}</h4>

              <div className="explain-feature-list">
                {agentData.features?.map((feature) => (
                  <div key={feature.name} className="explain-feature-item">
                    <div className="explain-feature-header">
                      <span className="explain-feature-name">
                        {feature.label}
                      </span>
                      <span className="explain-feature-pct">
                        {feature.percentage}%
                      </span>
                    </div>

                    <div className="explain-bar-track">
                      <div
                        className={`explain-bar-fill ${getBarColorClass(
                          feature.percentage,
                        )}`}
                        style={{ width: `${Math.min(feature.percentage, 100)}%` }}
                      />
                    </div>

                    <div className="explain-feature-meta">
                      <span>Value: {feature.value}</span>
                      <span>Weight: {feature.weight}</span>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          );
        })}
      </div>

      {coordinator && (
        <div className="coordinator-fusion-section">
          <h4>Coordinator Fusion Weights</h4>

          <div className="fusion-weights-list">
            {coordinator.fusion_weights?.map((item) => (
              <div key={item.agent} className="fusion-weight-item">
                <span className="fusion-weight-label">{item.label}</span>

                <div className="fusion-weight-bar-track">
                  <div
                    className="fusion-weight-bar-fill"
                    style={{ width: `${item.percentage}%` }}
                  />
                </div>

                <span className="fusion-weight-pct">{item.percentage}%</span>
              </div>
            ))}
          </div>
        </div>
      )}

      <p className="explainability-explanation">
        {safeValue(explainability.explanation)}
      </p>

      <div className="explainability-safety-note">
        {safeValue(explainability.safety_note)}
      </div>
    </div>
  );
};

const PatientListPanel = ({
  patients,
  filteredPatients,
  selectedPatientId,
  searchValue,
  onSearchChange,
  onSelectPatient,
  loading,
}) => {
  const trimmedSearch = searchValue.trim();
  const showNoPatients = !loading && patients.length === 0;
  const showNoMatches =
    !loading &&
    patients.length > 0 &&
    filteredPatients.length === 0 &&
    trimmedSearch;

  return (
    <div className="card patient-list-panel">
      <div className="patient-list-header">
        <h3>Patient Cases</h3>
        <span className="patient-count">{filteredPatients.length}</span>
      </div>

      <input
        className="patient-search-input"
        type="text"
        placeholder="Search by patient name or ID..."
        value={searchValue}
        onChange={(event) => onSearchChange(event.target.value)}
      />

      <div className="patient-list">
        {loading && (
          <div className="patient-list-message">Loading patients...</div>
        )}

        {showNoPatients && (
          <div className="patient-list-message">No patients available.</div>
        )}

        {showNoMatches && (
          <div className="patient-list-message">No matching patient found.</div>
        )}

        {!loading &&
          filteredPatients.length > 0 &&
          filteredPatients.map((entry) => {
            const medicationResponse =
              entry.medication_response ?? entry.clinical?.medication_response;
            const isSelected = selectedPatientId === entry.patient_id;

            return (
              <button
                key={entry.patient_id}
                type="button"
                className={`patient-list-item${isSelected ? " selected" : ""}`}
                onClick={() => onSelectPatient(entry.patient_id)}
              >
                <div className="patient-name-block">
                  <strong>{safeValue(entry.patient_name)}</strong>
                  <span>{safeValue(entry.patient_id)}</span>
                </div>

                <div className="patient-meta">
                  <span>Age {safeValue(entry.age)}</span>
                  <span>{formatLabel(entry.gender)}</span>
                </div>

                <div className="patient-meta">
                  <span>Medication Response</span>
                  <span>{formatLabel(medicationResponse)}</span>
                </div>
              </button>
            );
          })}
      </div>
    </div>
  );
};

const HistoryPanel = ({
  records,
  loading,
  onRefresh,
  title = "Analysis History",
}) => {
  const items = Array.isArray(records) ? records : [];

  return (
    <div className="card history-panel">
      <div className="history-header">
        <div>
          <h3>{title}</h3>
          <p>Saved analysis records from the clinical review workflow.</p>
        </div>

        <button
          className="btn btn-secondary"
          type="button"
          onClick={onRefresh}
          disabled={loading}
        >
          {loading ? "Refreshing..." : "Refresh History"}
        </button>
      </div>

      {loading && <div className="patient-list-message">Loading history...</div>}

      {!loading && items.length === 0 && (
        <div className="patient-list-message">No analysis history yet.</div>
      )}

      {!loading && items.length > 0 && (
        <div className="history-list">
          {items.map((record) => (
            <div className="history-card" key={record.id}>
              <div className="history-card-header">
                <div>
                  <h4>{safeValue(record.patient_name)}</h4>
                  <p>{safeValue(record.patient_id)}</p>
                </div>

                <span className={`badge ${getRiskClass(record.final_risk_score)}`}>
                  {typeof record.final_risk_score === "number"
                    ? `${toPercent(record.final_risk_score)}% Risk`
                    : "Risk N/A"}
                </span>
              </div>

              <div className="history-meta-grid">
                <div>
                  <label>Final Risk Level</label>
                  <span>{formatLabel(record.final_risk_level)}</span>
                </div>

                <div>
                  <label>Final Prediction</label>
                  <span>{formatLabel(record.final_prediction)}</span>
                </div>

                <div>
                  <label>Triage Level</label>
                  <span>{formatLabel(record.triage_level)}</span>
                </div>

                <div>
                  <label>Priority</label>
                  <span>{formatLabel(record.priority)}</span>
                </div>

                <div>
                  <label>Created At</label>
                  <span>{formatDateTime(record.created_at)}</span>
                </div>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
};

function App() {
  const [patient, setPatient] = useState(null);
  const [analysis, setAnalysis] = useState(null);
  const [patients, setPatients] = useState([]);
  const [patientSearch, setPatientSearch] = useState("");
  const [loadingPatients, setLoadingPatients] = useState(false);
  const [loadingPatient, setLoadingPatient] = useState(false);
  const [loadingAnalysis, setLoadingAnalysis] = useState(false);
  const [error, setError] = useState("");
  const [statusMsg, setStatusMsg] = useState("");
  const [activePage, setActivePage] = useState("patient-case");
  const [history, setHistory] = useState([]);
  const [patientHistory, setPatientHistory] = useState([]);
  const [loadingHistory, setLoadingHistory] = useState(false);
  const [feedbackComment, setFeedbackComment] = useState("");

  const loadPatientList = async () => {
    setLoadingPatients(true);
    setError("");

    try {
      const data = await getPatients();
      setPatients(Array.isArray(data) ? data : []);
    } catch (err) {
      console.error(err);
      setError("Could not load patients. Check if the backend is running.");
    } finally {
      setLoadingPatients(false);
    }
  };

  const loadHistory = async () => {
    setLoadingHistory(true);
    setError("");

    try {
      const data = await getHistory();
      setHistory(Array.isArray(data) ? data : []);
    } catch (err) {
      console.error(err);
      setError("Could not load analysis history. Check if the backend is running.");
    } finally {
      setLoadingHistory(false);
    }
  };

  const loadPatientHistory = async (patientId) => {
    if (!patientId) {
      setPatientHistory([]);
      return;
    }

    setLoadingHistory(true);
    setError("");

    try {
      const data = await getPatientHistory(patientId);
      setPatientHistory(Array.isArray(data) ? data : []);
    } catch (err) {
      console.error(err);
      setError("Could not load patient analysis history.");
    } finally {
      setLoadingHistory(false);
    }
  };

  useEffect(() => {
    loadPatientList();
    loadHistory();
  }, []);

  const filteredPatients = useMemo(() => {
    const term = patientSearch.trim().toLowerCase();

    if (!term) return patients;

    return patients.filter((entry) => {
      const id = String(entry.patient_id || "").toLowerCase();
      const name = String(entry.patient_name || "").toLowerCase();

      return id.includes(term) || name.includes(term);
    });
  }, [patients, patientSearch]);

  const coordinator = analysis?.agent_results?.coordinator;
  const triage = analysis?.agent_results?.triage;
  const critic = analysis?.agent_results?.critic;
  const conflict = analysis?.agent_results?.conflict;
  const rag = analysis?.agent_results?.rag;
  const progression = analysis?.agent_results?.progression;
  const dbsReferral = analysis?.agent_results?.dbs_referral;
  const explainability = analysis?.agent_results?.explainability;
  const medicalEvidenceSummary = analysis?.report?.medical_evidence_summary;

  const diseaseDuration =
    patient?.clinical?.disease_duration_years ?? patient?.disease_duration_years;
  const medicationResponse =
    patient?.clinical?.medication_response ?? patient?.medication_response;

  const displayedHistory = patient ? patientHistory : history;

  const handleSearchChange = (value) => {
    setPatientSearch(value);
  };

  const handleSelectPatient = async (patientId) => {
    if (!patientId) return;

    setLoadingPatient(true);
    setError("");
    setStatusMsg("");
    setAnalysis(null);

    try {
      const selectedPatient = await getPatientById(patientId);
      setPatient(selectedPatient);
      setActivePage("patient-case");

      await loadPatientHistory(selectedPatient?.patient_id);

      const loadedName = selectedPatient?.patient_name || patientId;
      setStatusMsg(`Loaded patient ${loadedName}`);
      setTimeout(() => setStatusMsg(""), 2500);
    } catch (err) {
      console.error(err);
      setError("Could not load patient profile. Check if the backend is running.");
    } finally {
      setLoadingPatient(false);
    }
  };

  const handleLoadDemo = async () => {
    setLoadingPatient(true);
    setError("");
    setStatusMsg("");

    try {
      const data = await getDemoPatient();
      const loadedPatient = data.patient ?? data;

      setPatient(loadedPatient);
      setAnalysis(null);
      setActivePage("patient-case");

      await loadPatientHistory(loadedPatient?.patient_id);

      const demoName = loadedPatient?.patient_name;
      setStatusMsg(demoName ? `Loaded patient ${demoName}` : "Demo patient loaded");
      setTimeout(() => setStatusMsg(""), 2500);
    } catch (err) {
      console.error(err);
      setError("Could not load demo patient. Check if the backend is running.");
    } finally {
      setLoadingPatient(false);
    }
  };

  const handleRunAnalysis = async () => {
    if (!patient) {
      setStatusMsg("Load a patient case first");
      setTimeout(() => setStatusMsg(""), 2500);
      return;
    }

    setLoadingAnalysis(true);
    setError("");
    setStatusMsg("");

    try {
      const result = await analyzePatient(patient);
      setAnalysis(result);
      setActivePage("agent-analysis");

      const analyzedPatientId = result?.patient?.patient_id ?? patient?.patient_id;
      await loadPatientHistory(analyzedPatientId);

      setStatusMsg("Agent analysis completed and saved to history");
      setTimeout(() => setStatusMsg(""), 2500);
    } catch (err) {
      console.error(err);
      setError("AI analysis failed. Check backend connection and request format.");
    } finally {
      setLoadingAnalysis(false);
    }
  };

  const handleNavigation = (page) => {
    setActivePage(page);
    setError("");

    if (page === "agent-analysis" && !analysis) {
      setStatusMsg("Run agent analysis first.");
      setTimeout(() => setStatusMsg(""), 2500);
      return;
    }

    if (page === "medical-evidence" && !analysis) {
      setStatusMsg("Run agent analysis first to view retrieved medical evidence.");
      setTimeout(() => setStatusMsg(""), 2500);
      return;
    }

    if (page === "reports") {
      if (patient?.patient_id) {
        loadPatientHistory(patient.patient_id);
      } else {
        loadHistory();
      }

      if (!analysis) {
        setStatusMsg("No current report yet. Showing saved analysis history.");
        setTimeout(() => setStatusMsg(""), 2500);
        return;
      }
    }

    setStatusMsg("");
  };

  const handleSubmitFeedback = async (action) => {
    if (!analysis) {
      setStatusMsg("Run agent analysis before submitting feedback.");
      setTimeout(() => setStatusMsg(""), 2500);
      return;
    }

    setError("");
    setStatusMsg("");

    try {
      await submitDoctorFeedback({
        patient_id: analysis?.patient?.patient_id ?? patient?.patient_id,
        analysis_id: analysis?.analysis_id ?? null,
        action,
        comment: feedbackComment,
      });

      setFeedbackComment("");
      setStatusMsg("Feedback saved");
      setTimeout(() => setStatusMsg(""), 2500);

      const analyzedPatientId = analysis?.patient?.patient_id ?? patient?.patient_id;

      if (analyzedPatientId) {
        await loadPatientHistory(analyzedPatientId);
      } else {
        await loadHistory();
      }
    } catch (err) {
      console.error(err);
      setError("Could not save doctor feedback. Check backend connection.");
    }
  };

  const handleRefreshHistory = () => {
    if (patient?.patient_id) {
      loadPatientHistory(patient.patient_id);
    } else {
      loadHistory();
    }
  };

  const handleLogout = () => {
    setPatient(null);
    setAnalysis(null);
    setPatientHistory([]);
    setFeedbackComment("");
    setError("");
    setStatusMsg("Demo session cleared");
    setActivePage("patient-case");
    setTimeout(() => setStatusMsg(""), 2500);
  };

  return (
    <div className="dashboard-container">
      <aside className="sidebar">
        <div className="brand">
          <h1>NeuroAgent-PD</h1>
          <span>Clinical AI v0.1</span>
        </div>

        <nav>
          <ul>
            <li className={activePage === "patient-case" ? "active" : ""}>
              <button type="button" onClick={() => handleNavigation("patient-case")}>
                Patient Case
              </button>
            </li>

            <li className={activePage === "agent-analysis" ? "active" : ""}>
              <button
                type="button"
                onClick={() => handleNavigation("agent-analysis")}
              >
                Agent Analysis
              </button>
            </li>

            <li className={activePage === "medical-evidence" ? "active" : ""}>
              <button
                type="button"
                onClick={() => handleNavigation("medical-evidence")}
              >
                Medical Evidence
              </button>
            </li>

            <li className={activePage === "reports" ? "active" : ""}>
              <button type="button" onClick={() => handleNavigation("reports")}>
                Reports
              </button>
            </li>

            <li className={activePage === "settings" ? "active" : ""}>
              <button type="button" onClick={() => handleNavigation("settings")}>
                Settings
              </button>
            </li>
          </ul>
        </nav>

        <button className="logout-btn" type="button" onClick={handleLogout}>
          Logout
        </button>
      </aside>

      <main className="main-content">
        <header className="top-bar">
          <h2>Diagnostic Dashboard</h2>

          <div className="top-bar-right">
            <span className="icon-btn">?</span>
            <span className="icon-btn">🔔</span>

            <div className="doctor-profile">
              <strong>Dr. Neurologist</strong>
              <span>Senior Clinician</span>
            </div>
          </div>
        </header>

        <section className="content-body">
          <div className="breadcrumb">Patients &gt; {getPageTitle(activePage)}</div>

          <div className="page-header">
            <div>
              <h1>{getPageTitle(activePage)}</h1>
              <p className="page-subtitle">{getPageSubtitle(activePage)}</p>
            </div>

            <div className="header-actions">
              <button
                className="btn btn-secondary"
                type="button"
                onClick={handleLoadDemo}
                disabled={loadingPatient || loadingAnalysis}
              >
                {loadingPatient ? "Loading..." : "Load Demo Patient"}
              </button>

              <button
                className="btn btn-primary"
                type="button"
                onClick={handleRunAnalysis}
                disabled={loadingAnalysis || loadingPatient || !patient}
              >
                {loadingAnalysis ? "Analyzing..." : "Run Agent Analysis"}
              </button>
            </div>
          </div>

          {error && <div className="error-banner">{error}</div>}
          {statusMsg && <div className="status-toast">{statusMsg}</div>}

          {activePage === "patient-case" && (
            <div className="patient-case-layout">
              <PatientListPanel
                patients={patients}
                filteredPatients={filteredPatients}
                selectedPatientId={patient?.patient_id}
                searchValue={patientSearch}
                onSearchChange={handleSearchChange}
                onSelectPatient={handleSelectPatient}
                loading={loadingPatients}
              />

              <div className="patient-case-panel">
                {loadingPatient && !patient && (
                  <div className="empty-state">
                    <h2>Loading patient profile</h2>
                    <p>Please wait while we fetch the patient details.</p>
                  </div>
                )}

                {!loadingPatient && !patient && (
                  <div className="empty-state">
                    <h2>No patient selected</h2>
                    <p>
                      Select a patient from the patient cases menu or load the demo
                      patient.
                    </p>
                  </div>
                )}

                {patient && (
                  <div className="card patient-profile">
                    <div className="card-title-row">
                      <div>
                        <h3>Patient Profile</h3>
                        <p className="patient-profile-name">
                          {safeValue(patient.patient_name)}
                        </p>
                      </div>

                      <span className="patient-id-badge">
                        {safeValue(patient.patient_id)}
                      </span>
                    </div>

                    <div className="patient-section">
                      <h4 className="patient-section-title">Clinical Snapshot</h4>

                      <div className="patient-data">
                        <div className="data-box">
                          <label>Age</label>
                          <span>
                            {typeof patient.age === "number"
                              ? `${patient.age} Years`
                              : safeValue(patient.age)}
                          </span>
                        </div>

                        <div className="data-box">
                          <label>Gender</label>
                          <span>{formatLabel(patient.gender)}</span>
                        </div>

                        <div className="data-box">
                          <label>Disease Duration</label>
                          <span>
                            {typeof diseaseDuration === "number"
                              ? `${diseaseDuration} Years`
                              : safeValue(diseaseDuration)}
                          </span>
                        </div>

                        <div className="data-box">
                          <label>Medication Response</label>
                          <span>{formatLabel(medicationResponse)}</span>
                        </div>
                      </div>
                    </div>

                    <div className="patient-section">
                      <h4 className="patient-section-title">Clinical Scores</h4>

                      <div className="patient-data">
                        <div className="data-box">
                          <label>UPDRS Score</label>
                          <span>{safeValue(patient.clinical?.updrs_score)}</span>
                        </div>

                        <div className="data-box">
                          <label>Tremor Score</label>
                          <span>{safeValue(patient.clinical?.tremor_score)}</span>
                        </div>

                        <div className="data-box">
                          <label>Rigidity Score</label>
                          <span>{safeValue(patient.clinical?.rigidity_score)}</span>
                        </div>

                        <div className="data-box">
                          <label>Bradykinesia Score</label>
                          <span>
                            {safeValue(patient.clinical?.bradykinesia_score)}
                          </span>
                        </div>
                      </div>
                    </div>

                    <div className="patient-section">
                      <h4 className="patient-section-title">Speech Features</h4>

                      <div className="patient-data">
                        <div className="data-box">
                          <label>Jitter</label>
                          <span>{safeValue(patient.speech?.jitter)}</span>
                        </div>

                        <div className="data-box">
                          <label>Shimmer</label>
                          <span>{safeValue(patient.speech?.shimmer)}</span>
                        </div>

                        <div className="data-box">
                          <label>HNR</label>
                          <span>{safeValue(patient.speech?.hnr)}</span>
                        </div>

                        <div className="data-box">
                          <label>Pitch Variation</label>
                          <span>{safeValue(patient.speech?.pitch_variation)}</span>
                        </div>
                      </div>
                    </div>

                    <div className="patient-section">
                      <h4 className="patient-section-title">Gait Features</h4>

                      <div className="patient-data">
                        <div className="data-box">
                          <label>Walking Speed</label>
                          <span>{safeValue(patient.gait?.walking_speed)}</span>
                        </div>

                        <div className="data-box">
                          <label>Stride Variability</label>
                          <span>{safeValue(patient.gait?.stride_variability)}</span>
                        </div>

                        <div className="data-box">
                          <label>Freezing Index</label>
                          <span>{safeValue(patient.gait?.freezing_index)}</span>
                        </div>

                        <div className="data-box">
                          <label>Balance Score</label>
                          <span>{safeValue(patient.gait?.balance_score)}</span>
                        </div>
                      </div>
                    </div>

                    <div className="clinical-notes">
                      <label>Clinical Notes</label>
                      <p>{safeValue(patient.notes)}</p>
                    </div>
                  </div>
                )}
              </div>
            </div>
          )}

          {activePage === "agent-analysis" && (
            <>
              {!analysis && (
                <div className="empty-state">
                  <h2>Analysis not generated yet</h2>
                  <p>Select a patient and run agent analysis to view results.</p>
                </div>
              )}

              {analysis && (
                <div className="analysis-layout">
                  <div className="analysis-agent-row">
                    <AgentCard
                      title="Clinical Agent"
                      agentData={analysis.agent_results.clinical}
                      featureData={patient?.clinical}
                      features={[
                        { label: "UPDRS Score", key: "updrs_score" },
                        { label: "Tremor Score", key: "tremor_score" },
                        { label: "Rigidity Score", key: "rigidity_score" },
                        { label: "Bradykinesia Score", key: "bradykinesia_score" },
                      ]}
                    />

                    <AgentCard
                      title="Speech Agent"
                      agentData={analysis.agent_results.speech}
                      featureData={patient?.speech}
                      features={[
                        { label: "Jitter", key: "jitter" },
                        { label: "Shimmer", key: "shimmer" },
                        { label: "HNR", key: "hnr" },
                        { label: "Pitch Variation", key: "pitch_variation" },
                      ]}
                    />

                    <AgentCard
                      title="Gait Agent"
                      agentData={analysis.agent_results.gait}
                      featureData={patient?.gait}
                      features={[
                        { label: "Walking Speed", key: "walking_speed" },
                        { label: "Stride Variability", key: "stride_variability" },
                        { label: "Freezing Index", key: "freezing_index" },
                        { label: "Balance Score", key: "balance_score" },
                      ]}
                    />
                  </div>

                  <ExplainabilityPanel
                    explainability={analysis?.agent_results?.explainability}
                  />

                  {(coordinator || triage) && (
                    <div className="analysis-decision-row">
                      {coordinator && (
                        <div
                          className={`card risk-card ${getRiskClass(
                            coordinator.final_risk_score,
                          )}`}
                        >
                          <div className="risk-header">
                            <div className="risk-value">
                              <span>
                                {toPercent(coordinator.final_risk_score)}
                              </span>
                              /100
                            </div>

                            <h3>Unified Risk Analysis</h3>
                          </div>

                          <div className="prediction-label">
                            {formatLabel(coordinator.final_prediction)}
                          </div>

                          <div className="risk-level-text">
                            Final Risk Level:{" "}
                            {safeValue(formatLabel(coordinator.final_risk_level))}
                          </div>

                          <p className="agreement-text">
                            {safeValue(coordinator.agent_agreement)}
                          </p>

                          <p className="risk-explanation">
                            {safeValue(coordinator.explanation)}
                          </p>

                          <div className="supporting-agent-group">
                            <p>Elevated Supporting Agents</p>

                            <div className="supporting-agent-list">
                              {coordinator.supporting_agents?.length > 0 ? (
                                coordinator.supporting_agents.map((agent) => (
                                  <span key={agent} className="support-badge">
                                    {formatLabel(agent)}
                                  </span>
                                ))
                              ) : (
                                <span className="support-badge">
                                  No elevated-risk agents
                                </span>
                              )}
                            </div>
                          </div>

                          <div className="borderline-agents">
                            <p>Borderline Agents</p>

                            <div className="borderline-agent-list">
                              {coordinator.borderline_agents?.length > 0 ? (
                                coordinator.borderline_agents.map((agent) => (
                                  <span key={agent} className="borderline-badge">
                                    {formatLabel(agent)}
                                  </span>
                                ))
                              ) : (
                                <span className="borderline-badge">
                                  No borderline agents
                                </span>
                              )}
                            </div>
                          </div>
                        </div>
                      )}

                      {triage && (
                        <div
                          className={`card triage-panel ${getPriorityClass(
                            triage.priority,
                          )}`}
                        >
                          <div className="triage-header">
                            <span className="priority-badge">
                              Priority {formatLabel(triage.priority)}
                            </span>

                            <h3>Triage Level</h3>
                          </div>

                          <div className="triage-type">
                            {formatLabel(triage.triage_level)}
                          </div>

                          <div className="recommendation-box">
                            <label>Recommendation</label>
                            <p>{safeValue(triage.recommendation)}</p>
                          </div>

                          <button
                            className="btn btn-accent full-width"
                            type="button"
                            onClick={() =>
                              handleSubmitFeedback("escalate_to_neurologist")
                            }
                          >
                            Escalate to Neurologist
                          </button>
                        </div>
                      )}
                    </div>
                  )}

                  {(conflict || critic) && (
                    <div className="analysis-safety-row">
                      <ConflictPanel conflict={conflict} />

                      {critic && (
                        <div className="card safety-monitor">
                          <div className="safety-header">
                            <span className="safety-icon">🛡️</span>
                            <h3>Safety Monitor</h3>
                          </div>

                          <div className="review-status">
                            Status:{" "}
                            <strong>
                              {critic.requires_human_review
                                ? "Requires Human Review"
                                : "No Critical Warning"}
                            </strong>
                          </div>

                          <ul className="warnings-list">
                            {critic.warnings?.length ? (
                              critic.warnings.map((warning, index) => (
                                <li key={`${warning}-${index}`}>{warning}</li>
                              ))
                            ) : (
                              <li>No warnings reported.</li>
                            )}
                          </ul>

                          <footer className="safety-disclaimer">
                            This is clinical decision support only. Not a final
                            medical diagnosis.
                          </footer>
                        </div>
                      )}
                    </div>
                  )}

                  <ProgressionPanel
                    progression={analysis?.agent_results?.progression}
                  />

                  <DBSReferralPanel
                    dbsReferral={analysis?.agent_results?.dbs_referral}
                  />

                  {analysis.report && (
                    <div className="analysis-report-preview">
                      <div className="card report-section">
                        <h3>Doctor Summary</h3>

                        {analysis.report?.llm_generated && analysis.report?.llm_report && (
                          <div className="llm-report-section">
                            <div className="llm-report-header">
                              <label>AI-Generated Clinical Report</label>
                              <span className="llm-provider-badge">
                                {formatLabel(analysis.report?.llm_provider)}
                              </span>
                            </div>

                            <div className="llm-report-content">
                              {analysis.report.llm_report.split("\n").map((line, idx) => (
                                <p key={idx}>{line || "\u00A0"}</p>
                              ))}
                            </div>

                            <div className="llm-report-disclaimer">
                              AI-generated report for clinical decision-support only.
                              Must be reviewed by a qualified neurologist before any clinical action.
                            </div>
                          </div>
                        )}

                        <div className="summary-box">
                          <label>
                            {analysis.report?.llm_generated
                              ? "Template Summary (Fallback)"
                              : "Summary"}
                          </label>
                          <p>{safeValue(analysis.report?.summary)}</p>
                        </div>

                        <div className="explanation-box">
                          <label>Doctor-Facing Explanation</label>
                          <p>{safeValue(analysis.report?.doctor_facing_explanation)}</p>
                        </div>
                      </div>
                    </div>
                  )}
                </div>
              )}
            </>
          )}

          {activePage === "medical-evidence" && (
            <div className="page-stack">
              {!analysis && (
                <div className="card">
                  <h3>Medical Evidence Not Available</h3>
                  <p>
                    Load a patient and run agent analysis to retrieve evidence from
                    the RAG knowledge base.
                  </p>
                </div>
              )}

              {analysis && (
                <MedicalEvidencePanel
                  rag={rag}
                  summary={medicalEvidenceSummary}
                />
              )}
            </div>
          )}

          {activePage === "reports" && (
            <div className="reports-layout">
              {!analysis && (
                <div className="empty-state compact-empty-state">
                  <h2>No current report generated</h2>
                  <p>
                    Load a patient and run agent analysis to generate a new
                    doctor-facing report. Saved analysis history is shown below.
                  </p>
                </div>
              )}

              {analysis && (
                <div className="card report-section">
                  <h3>Doctor’s Clinical Summary</h3>

                  <div className="summary-box">
                    <label>
                      {analysis.report?.llm_generated
                        ? "Template Summary (Fallback)"
                        : "Automated Synthesis"}
                    </label>
                    <p>{safeValue(analysis.report?.summary)}</p>
                  </div>

                  {analysis.report?.llm_generated && analysis.report?.llm_report && (
                    <div className="llm-report-section">
                      <div className="llm-report-header">
                        <label>AI-Generated Clinical Report</label>
                        <span className="llm-provider-badge">
                          {formatLabel(analysis.report?.llm_provider)}
                        </span>
                      </div>

                      <div className="llm-report-content">
                        {analysis.report.llm_report.split("\n").map((line, idx) => (
                          <p key={idx}>{line || "\u00A0"}</p>
                        ))}
                      </div>

                      <div className="llm-report-disclaimer">
                        AI-generated report for clinical decision-support only.
                        Must be reviewed by a qualified neurologist before any clinical action.
                      </div>
                    </div>
                  )}

                  <div className="explanation-box">
                    <label>Clinical Interpretation</label>
                    <p>{safeValue(analysis.report?.doctor_facing_explanation)}</p>
                  </div>

                  {rag && (
                    <div className="report-evidence-section">
                      <div className="evidence-header">
                        <h4>Medical Evidence</h4>

                        <span className="evidence-count-badge">
                          {safeValue(rag?.evidence_count ?? rag?.evidence?.length)}
                        </span>
                      </div>

                      <div className="evidence-status-grid">
                        <div className="evidence-status-item">
                          <label>Evidence Count</label>
                          <span>
                            {safeValue(rag?.evidence_count ?? rag?.evidence?.length)}
                          </span>
                        </div>
                      </div>

                      <p className="evidence-text">{safeValue(medicalEvidenceSummary)}</p>

                      <EvidenceCards evidence={rag?.evidence} />

                      <div className="evidence-safety-note">
                        {safeValue(
                          rag?.safety_note && rag?.safety_note !== REQUIRED_SAFETY_NOTE
                            ? `${rag?.safety_note} ${REQUIRED_SAFETY_NOTE}`
                            : rag?.safety_note || REQUIRED_SAFETY_NOTE,
                        )}
                      </div>
                    </div>
                  )}

                  {progression && (
                    <div className="report-progression-section">
                      <div className="progression-header">
                        <div>
                          <h4>Progression Simulation</h4>
                          <p>Projected 12-month risk trajectory.</p>
                        </div>

                        <span
                          className={`progression-badge ${getRiskClass(
                            progression.projected_12_month_risk_score,
                          )}`}
                        >
                          {toPercent(progression.projected_12_month_risk_score)}%
                          Projected
                        </span>
                      </div>

                      <p className="evidence-text">
                        {safeValue(analysis.report?.progression_summary)}
                      </p>

                      <div className="progression-timeline">
                        {progression.timeline?.map((item) => (
                          <div
                            key={item.month}
                            className={`timeline-card ${getRiskClass(
                              item.risk_score,
                            )}`}
                          >
                            <span className="timeline-month">Month {item.month}</span>
                            <strong>{toPercent(item.risk_score)}%</strong>
                            <small>{formatLabel(item.risk_level)}</small>
                          </div>
                        ))}
                      </div>

                      <div className="progression-safety-note">
                        {safeValue(progression.safety_note)}
                      </div>
                    </div>
                  )}

                  {dbsReferral && (
                    <div className="report-dbs-section">
                      <div className="dbs-header">
                        <div>
                          <h4>DBS Referral Assessment</h4>
                          <p>Deep Brain Stimulation candidacy evaluation.</p>
                        </div>

                        <span
                          className={`dbs-referral-badge referral-${(
                            dbsReferral.referral_level || "not-indicated"
                          ).replace("_", "-")}`}
                        >
                          {formatLabel(dbsReferral.referral_level)}
                        </span>
                      </div>

                      <p className="evidence-text">
                        {safeValue(analysis.report?.dbs_referral_summary)}
                      </p>

                      <div className="dbs-recommendation">
                        <label>Recommendation</label>
                        <p>{safeValue(dbsReferral.recommendation)}</p>
                      </div>

                      <div className="dbs-safety-note">
                        {safeValue(dbsReferral.safety_note)}
                      </div>
                    </div>
                  )}

                  <div className="metadata-grid">
                    <div className="metadata-item">
                      <label>Analysis ID</label>
                      <span>{safeValue(analysis?.analysis_id)}</span>
                    </div>

                    <div className="metadata-item">
                      <label>Patient Name</label>
                      <span>{safeValue(analysis.report?.patient_name)}</span>
                    </div>

                    <div className="metadata-item">
                      <label>Patient ID</label>
                      <span>{safeValue(analysis.report?.patient_id)}</span>
                    </div>

                    <div className="metadata-item">
                      <label>Final Risk Level</label>
                      <span>{safeValue(formatLabel(coordinator?.final_risk_level))}</span>
                    </div>

                    <div className="metadata-item">
                      <label>Final Prediction</label>
                      <span>{safeValue(formatLabel(coordinator?.final_prediction))}</span>
                    </div>

                    <div className="metadata-item">
                      <label>Conflict Type</label>
                      <span>{safeValue(formatLabel(conflict?.conflict_type))}</span>
                    </div>

                    <div className="metadata-item">
                      <label>Conflict Level</label>
                      <span>{safeValue(formatLabel(conflict?.conflict_level))}</span>
                    </div>

                    <div className="metadata-item">
                      <label>Triage Priority</label>
                      <span>{safeValue(formatLabel(triage?.priority))}</span>
                    </div>
                  </div>

                  <div className="feedback-box">
                    <label htmlFor="doctor-feedback-comment">
                      Optional Doctor Comment
                    </label>

                    <textarea
                      id="doctor-feedback-comment"
                      className="feedback-comment-input"
                      placeholder="Optional doctor comment..."
                      value={feedbackComment}
                      onChange={(event) => setFeedbackComment(event.target.value)}
                    />

                    <div className="report-footer-actions">
                      <button
                        className="btn btn-outline-danger"
                        type="button"
                        onClick={() => handleSubmitFeedback("reject")}
                      >
                        Reject Output
                      </button>

                      <button
                        className="btn btn-outline"
                        type="button"
                        onClick={() => handleSubmitFeedback("request_more_data")}
                      >
                        Request More Data
                      </button>

                      <button
                        className="btn btn-success"
                        type="button"
                        onClick={() => handleSubmitFeedback("approve")}
                      >
                        Approve Report
                      </button>
                    </div>
                  </div>
                </div>
              )}

              <HistoryPanel
                records={displayedHistory}
                loading={loadingHistory}
                onRefresh={handleRefreshHistory}
                title={patient ? "Patient Analysis History" : "Analysis History"}
              />
            </div>
          )}

          {activePage === "settings" && (
            <div className="card settings-placeholder">
              <h3>Settings</h3>
              <p>Settings page is not implemented yet.</p>
            </div>
          )}
        </section>
      </main>
    </div>
  );
}

export default App;