const API_BASE_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";

const parseResponse = async (response, context) => {
  if (!response.ok) {
    let details = "";

    try {
      details = await response.text();
    } catch (error) {
      details = "";
    }

    const message = details?.trim() || response.statusText || "Request failed";
    throw new Error(`${context}: ${message}`);
  }

  return response.json();
};

/**
 * Fetches patient summaries from the backend.
 * Backend endpoint: GET /patients
 */
export const getPatients = async (search = "") => {
  try {
    const query = search.trim()
      ? `?search=${encodeURIComponent(search.trim())}`
      : "";

    const response = await fetch(`${API_BASE_URL}/patients${query}`);

    return await parseResponse(response, "Failed to fetch patients");
  } catch (error) {
    console.error("API Error (getPatients):", error);
    throw error;
  }
};

/**
 * Fetches a full patient profile by ID.
 * Backend endpoint: GET /patients/{patient_id}
 */
export const getPatientById = async (patientId) => {
  try {
    const encodedId = encodeURIComponent(patientId);
    const response = await fetch(`${API_BASE_URL}/patients/${encodedId}`);

    return await parseResponse(response, "Failed to fetch patient profile");
  } catch (error) {
    console.error("API Error (getPatientById):", error);
    throw error;
  }
};

/**
 * Fetches demo patient data from the backend.
 * Backend endpoint: GET /demo-patient
 */
export const getDemoPatient = async () => {
  try {
    const response = await fetch(`${API_BASE_URL}/demo-patient`);

    return await parseResponse(response, "Failed to fetch demo patient");
  } catch (error) {
    console.error("API Error (getDemoPatient):", error);
    throw error;
  }
};

/**
 * Sends patient data for AI agent analysis.
 * Backend endpoint: POST /analyze
 *
 * Backend now also saves the analysis in PostgreSQL
 * and returns analysis.analysis_id.
 */
export const analyzePatient = async (patient) => {
  try {
    const response = await fetch(`${API_BASE_URL}/analyze`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(patient),
    });

    return await parseResponse(response, "Analysis failed");
  } catch (error) {
    console.error("API Error (analyzePatient):", error);
    throw error;
  }
};

/**
 * Fetches latest global analysis history.
 * Backend endpoint: GET /history
 */
export const getHistory = async (limit = 20) => {
  try {
    const response = await fetch(
      `${API_BASE_URL}/history?limit=${encodeURIComponent(limit)}`
    );

    return await parseResponse(response, "Failed to fetch analysis history");
  } catch (error) {
    console.error("API Error (getHistory):", error);
    throw error;
  }
};

/**
 * Fetches analysis history for a specific patient.
 * Backend endpoint: GET /history/{patient_id}
 */
export const getPatientHistory = async (patientId, limit = 20) => {
  try {
    const encodedId = encodeURIComponent(patientId);

    const response = await fetch(
      `${API_BASE_URL}/history/${encodedId}?limit=${encodeURIComponent(limit)}`
    );

    return await parseResponse(response, "Failed to fetch patient history");
  } catch (error) {
    console.error("API Error (getPatientHistory):", error);
    throw error;
  }
};

/**
 * Fetches full saved analysis detail by analysis ID.
 * Backend endpoint: GET /history/detail/{analysis_id}
 */
export const getAnalysisDetail = async (analysisId) => {
  try {
    const encodedId = encodeURIComponent(analysisId);

    const response = await fetch(
      `${API_BASE_URL}/history/detail/${encodedId}`
    );

    return await parseResponse(response, "Failed to fetch analysis detail");
  } catch (error) {
    console.error("API Error (getAnalysisDetail):", error);
    throw error;
  }
};

/**
 * Saves doctor feedback.
 * Backend endpoint: POST /feedback
 *
 * Allowed action values:
 * - approve
 * - reject
 * - request_more_data
 * - escalate_to_neurologist
 */
export const submitDoctorFeedback = async ({
  patient_id,
  analysis_id = null,
  action,
  comment = "",
}) => {
  try {
    const response = await fetch(`${API_BASE_URL}/feedback`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        patient_id,
        analysis_id,
        action,
        comment,
      }),
    });

    return await parseResponse(response, "Failed to submit doctor feedback");
  } catch (error) {
    console.error("API Error (submitDoctorFeedback):", error);
    throw error;
  }
};