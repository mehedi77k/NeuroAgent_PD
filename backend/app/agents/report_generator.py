import os
import json
import httpx


LLM_ENABLED = os.getenv("LLM_ENABLED", "false").lower() == "true"
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

GROQ_MODEL = "llama-3.1-8b-instant"
GEMINI_MODEL = "gemini-2.0-flash"

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
GEMINI_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent"


SYSTEM_PROMPT = """You are a clinical report generator for NeuroAgent-PD, a Parkinson's disease clinical decision-support system. You generate concise, professional doctor-facing reports.

Rules:
- Write in third person clinical style
- Be factual and concise — no filler or marketing language
- Always state this is decision-support only, not a diagnosis
- Include key findings from each agent
- Mention the triage recommendation
- If DBS referral is indicated, mention it
- If conflict was detected, note it requires attention
- Keep the report under 300 words
- Use clear section structure: Summary, Key Findings, Risk Assessment, Recommendations
- Do not invent data — only use what is provided"""


def _build_prompt(patient_case, agent_results, template_report):
    """Build the user prompt with all agent data for the LLM."""

    clinical = agent_results.get("clinical", {})
    speech = agent_results.get("speech", {})
    gait = agent_results.get("gait", {})
    coordinator = agent_results.get("coordinator", {})
    triage = agent_results.get("triage", {})
    conflict = agent_results.get("conflict", {})
    progression = agent_results.get("progression", {})
    dbs_referral = agent_results.get("dbs_referral", {})
    critic = agent_results.get("critic", {})

    context = f"""Generate a doctor-facing clinical report for the following patient case.

PATIENT:
- Name: {patient_case.get('patient_name', 'Unknown')}
- ID: {patient_case.get('patient_id', 'N/A')}
- Age: {patient_case.get('age', 'N/A')}
- Gender: {patient_case.get('gender', 'N/A')}
- Disease Duration: {patient_case.get('clinical', {}).get('disease_duration_years', 'N/A')} years
- Medication Response: {patient_case.get('clinical', {}).get('medication_response', 'N/A')}

AGENT RESULTS:
- Clinical Agent: risk_score={clinical.get('risk_score')}, severity={clinical.get('severity')}, risk_level={clinical.get('risk_level')}
- Speech Agent: risk_score={speech.get('risk_score')}, severity={speech.get('severity')}, risk_level={speech.get('risk_level')}
- Gait Agent: risk_score={gait.get('risk_score')}, severity={gait.get('severity')}, risk_level={gait.get('risk_level')}
- Coordinator: final_risk_score={coordinator.get('final_risk_score')}, final_risk_level={coordinator.get('final_risk_level')}, prediction={coordinator.get('final_prediction')}
- Triage: level={triage.get('triage_level')}, priority={triage.get('priority')}, recommendation={triage.get('recommendation')}
- Conflict: detected={conflict.get('conflict_detected')}, level={conflict.get('conflict_level')}, type={conflict.get('conflict_type')}
- Progression: 12-month projected risk={progression.get('projected_12_month_risk_score')}, level={progression.get('projected_12_month_risk_level')}
- DBS Referral: recommended={dbs_referral.get('referral_recommended')}, level={dbs_referral.get('referral_level')}, score={dbs_referral.get('referral_score')}
- Safety Warnings: {len(critic.get('warnings', []))} warnings, requires_human_review={critic.get('requires_human_review')}

Generate the clinical report now."""

    return context


def _call_groq(system_prompt, user_prompt):
    """Call Groq API (OpenAI-compatible endpoint)."""

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": GROQ_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.3,
        "max_tokens": 800,
    }

    response = httpx.post(
        GROQ_URL,
        headers=headers,
        json=payload,
        timeout=30.0,
    )

    response.raise_for_status()
    data = response.json()

    return data["choices"][0]["message"]["content"].strip()


def _call_gemini(system_prompt, user_prompt):
    """Call Google Gemini API."""

    url = f"{GEMINI_URL}?key={GEMINI_API_KEY}"

    payload = {
        "system_instruction": {
            "parts": [{"text": system_prompt}],
        },
        "contents": [
            {
                "parts": [{"text": user_prompt}],
            }
        ],
        "generationConfig": {
            "temperature": 0.3,
            "maxOutputTokens": 800,
        },
    }

    response = httpx.post(
        url,
        json=payload,
        timeout=30.0,
    )

    response.raise_for_status()
    data = response.json()

    candidates = data.get("candidates", [])
    if candidates:
        parts = candidates[0].get("content", {}).get("parts", [])
        if parts:
            return parts[0].get("text", "").strip()

    return ""


def _call_llm(system_prompt, user_prompt):
    """Try available LLM providers in order: Groq first, then Gemini."""

    if GROQ_API_KEY:
        try:
            return _call_groq(system_prompt, user_prompt)
        except Exception as e:
            print(f"[ReportGenerator] Groq API failed: {e}")

    if GEMINI_API_KEY:
        try:
            return _call_gemini(system_prompt, user_prompt)
        except Exception as e:
            print(f"[ReportGenerator] Gemini API failed: {e}")

    return ""


def generate_llm_report(patient_case_dict, agent_results, template_report):
    """
    Generate an LLM-powered doctor-facing report.

    Falls back to the template report if:
    - LLM_ENABLED is false
    - No API keys are configured
    - The LLM call fails

    Returns a dict with:
    - llm_generated: bool
    - llm_report: str (the generated narrative)
    - llm_provider: str (which provider was used)
    - fallback_used: bool
    """

    if not LLM_ENABLED:
        return {
            "llm_generated": False,
            "llm_report": "",
            "llm_provider": "none",
            "fallback_used": True,
            "reason": "LLM_ENABLED is set to false",
        }

    if not GROQ_API_KEY and not GEMINI_API_KEY:
        return {
            "llm_generated": False,
            "llm_report": "",
            "llm_provider": "none",
            "fallback_used": True,
            "reason": "No LLM API key configured",
        }

    user_prompt = _build_prompt(patient_case_dict, agent_results, template_report)

    llm_response = _call_llm(SYSTEM_PROMPT, user_prompt)

    if not llm_response:
        return {
            "llm_generated": False,
            "llm_report": "",
            "llm_provider": "none",
            "fallback_used": True,
            "reason": "LLM call returned empty response",
        }

    provider = "groq" if GROQ_API_KEY else "gemini"

    return {
        "llm_generated": True,
        "llm_report": llm_response,
        "llm_provider": provider,
        "fallback_used": False,
        "reason": "",
    }
