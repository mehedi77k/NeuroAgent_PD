"""
Synthetic Dataset Generator for NeuroAgent-PD ML Agents.

Generates training datasets that match the exact patient schema
used by the clinical, speech, and gait agents.

Each dataset uses clinically-informed distributions based on
published Parkinson's disease literature ranges.

IMPORTANT: These are synthetic datasets for MVP demonstration.
They do NOT replace real clinical data for production use.
"""

import numpy as np
import pandas as pd
import os

np.random.seed(42)

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
N_SAMPLES = 2000


def generate_clinical_dataset(n=N_SAMPLES):
    """
    Clinical Agent Dataset.

    Features match PatientCase.clinical schema:
    - updrs_score (0-60)
    - tremor_score (0-4)
    - rigidity_score (0-4)
    - bradykinesia_score (0-4)
    - disease_duration_years (0-15)
    - medication_response (encoded: 0=good, 1=moderate, 2=poor)

    Target: risk_level (low, borderline, moderate, high)
    """

    data = []

    # Low risk patients (40%)
    for _ in range(int(n * 0.4)):
        updrs = np.random.uniform(5, 28)
        tremor = np.random.choice([0, 1, 2], p=[0.3, 0.5, 0.2])
        rigidity = np.random.choice([0, 1, 2], p=[0.4, 0.4, 0.2])
        bradykinesia = np.random.choice([0, 1, 2], p=[0.3, 0.5, 0.2])
        duration = np.random.uniform(0, 3)
        med_response = np.random.choice([0, 1, 2], p=[0.5, 0.35, 0.15])

        data.append([updrs, tremor, rigidity, bradykinesia, duration, med_response, "low"])

    # Borderline patients (15%)
    for _ in range(int(n * 0.15)):
        updrs = np.random.uniform(18, 36)
        tremor = np.random.choice([1, 2, 3], p=[0.4, 0.4, 0.2])
        rigidity = np.random.choice([1, 2, 3], p=[0.4, 0.4, 0.2])
        bradykinesia = np.random.choice([1, 2, 3], p=[0.4, 0.4, 0.2])
        duration = np.random.uniform(1, 5)
        med_response = np.random.choice([0, 1, 2], p=[0.25, 0.5, 0.25])

        data.append([updrs, tremor, rigidity, bradykinesia, duration, med_response, "borderline"])

    # Moderate risk patients (25%)
    for _ in range(int(n * 0.25)):
        updrs = np.random.uniform(25, 50)
        tremor = np.random.choice([1, 2, 3, 4], p=[0.15, 0.35, 0.35, 0.15])
        rigidity = np.random.choice([1, 2, 3, 4], p=[0.15, 0.35, 0.35, 0.15])
        bradykinesia = np.random.choice([2, 3, 4], p=[0.3, 0.45, 0.25])
        duration = np.random.uniform(2, 9)
        med_response = np.random.choice([0, 1, 2], p=[0.15, 0.45, 0.4])

        data.append([updrs, tremor, rigidity, bradykinesia, duration, med_response, "moderate"])

    # High risk patients (20%)
    for _ in range(int(n * 0.2)):
        updrs = np.random.uniform(35, 60)
        tremor = np.random.choice([2, 3, 4], p=[0.2, 0.4, 0.4])
        rigidity = np.random.choice([2, 3, 4], p=[0.2, 0.35, 0.45])
        bradykinesia = np.random.choice([2, 3, 4], p=[0.15, 0.35, 0.5])
        duration = np.random.uniform(4, 15)
        med_response = np.random.choice([0, 1, 2], p=[0.1, 0.3, 0.6])

        data.append([updrs, tremor, rigidity, bradykinesia, duration, med_response, "high"])

    df = pd.DataFrame(data, columns=[
        "updrs_score",
        "tremor_score",
        "rigidity_score",
        "bradykinesia_score",
        "disease_duration_years",
        "medication_response_encoded",
        "risk_level",
    ])

    # Add significant noise to create overlap
    noise = np.random.normal(0, 4.0, size=len(df))
    df["updrs_score"] = np.clip(df["updrs_score"] + noise, 0, 60).round(1)
    df["disease_duration_years"] = np.clip(
        df["disease_duration_years"] + np.random.normal(0, 1.2, len(df)), 0, 15
    ).round(1)

    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    path = os.path.join(OUTPUT_DIR, "clinical_agent_dataset.csv")
    df.to_csv(path, index=False)
    print(f"Clinical dataset: {len(df)} samples -> {path}")
    print(f"  Class distribution: {df['risk_level'].value_counts().to_dict()}")

    return df


def generate_speech_dataset(n=N_SAMPLES):
    """
    Speech Agent Dataset.

    Features match PatientCase.speech schema:
    - jitter (0.001 - 0.06)
    - shimmer (0.005 - 0.1)
    - hnr (8 - 30 dB)
    - pitch_variation (0.03 - 0.6)

    Target: risk_level (low, borderline, moderate, high)
    """

    data = []

    # Low risk (40%)
    for _ in range(int(n * 0.4)):
        jitter = np.random.uniform(0.003, 0.016)
        shimmer = np.random.uniform(0.008, 0.026)
        hnr = np.random.uniform(19, 30)
        pitch_var = np.random.uniform(0.03, 0.18)

        data.append([jitter, shimmer, hnr, pitch_var, "low"])

    # Borderline (15%)
    for _ in range(int(n * 0.15)):
        jitter = np.random.uniform(0.010, 0.024)
        shimmer = np.random.uniform(0.016, 0.035)
        hnr = np.random.uniform(16, 24)
        pitch_var = np.random.uniform(0.10, 0.28)

        data.append([jitter, shimmer, hnr, pitch_var, "borderline"])

    # Moderate (25%)
    for _ in range(int(n * 0.25)):
        jitter = np.random.uniform(0.015, 0.036)
        shimmer = np.random.uniform(0.022, 0.055)
        hnr = np.random.uniform(12, 21)
        pitch_var = np.random.uniform(0.18, 0.42)

        data.append([jitter, shimmer, hnr, pitch_var, "moderate"])

    # High risk (20%)
    for _ in range(int(n * 0.2)):
        jitter = np.random.uniform(0.024, 0.055)
        shimmer = np.random.uniform(0.035, 0.090)
        hnr = np.random.uniform(8, 18)
        pitch_var = np.random.uniform(0.28, 0.58)

        data.append([jitter, shimmer, hnr, pitch_var, "high"])

    df = pd.DataFrame(data, columns=[
        "jitter",
        "shimmer",
        "hnr",
        "pitch_variation",
        "risk_level",
    ])

    # Add significant noise to create overlap
    df["jitter"] = np.clip(df["jitter"] + np.random.normal(0, 0.004, len(df)), 0.001, 0.06).round(4)
    df["shimmer"] = np.clip(df["shimmer"] + np.random.normal(0, 0.006, len(df)), 0.005, 0.1).round(4)
    df["hnr"] = np.clip(df["hnr"] + np.random.normal(0, 2.5, len(df)), 8, 30).round(1)
    df["pitch_variation"] = np.clip(df["pitch_variation"] + np.random.normal(0, 0.05, len(df)), 0.03, 0.6).round(3)

    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    path = os.path.join(OUTPUT_DIR, "speech_agent_dataset.csv")
    df.to_csv(path, index=False)
    print(f"Speech dataset: {len(df)} samples -> {path}")
    print(f"  Class distribution: {df['risk_level'].value_counts().to_dict()}")

    return df


def generate_gait_dataset(n=N_SAMPLES):
    """
    Gait Agent Dataset.

    Features match PatientCase.gait schema:
    - walking_speed (0.3 - 1.5 m/s)
    - stride_variability (0.03 - 0.7)
    - freezing_index (0.0 - 0.9)
    - balance_score (0.2 - 1.0)

    Target: risk_level (low, borderline, moderate, high)
    """

    data = []

    # Low risk (40%)
    for _ in range(int(n * 0.4)):
        speed = np.random.uniform(1.0, 1.5)
        stride = np.random.uniform(0.03, 0.18)
        freezing = np.random.uniform(0.0, 0.15)
        balance = np.random.uniform(0.75, 1.0)

        data.append([speed, stride, freezing, balance, "low"])

    # Borderline (15%)
    for _ in range(int(n * 0.15)):
        speed = np.random.uniform(0.85, 1.2)
        stride = np.random.uniform(0.10, 0.28)
        freezing = np.random.uniform(0.08, 0.35)
        balance = np.random.uniform(0.60, 0.85)

        data.append([speed, stride, freezing, balance, "borderline"])

    # Moderate (25%)
    for _ in range(int(n * 0.25)):
        speed = np.random.uniform(0.60, 1.0)
        stride = np.random.uniform(0.18, 0.48)
        freezing = np.random.uniform(0.20, 0.58)
        balance = np.random.uniform(0.42, 0.72)

        data.append([speed, stride, freezing, balance, "moderate"])

    # High risk (20%)
    for _ in range(int(n * 0.2)):
        speed = np.random.uniform(0.35, 0.78)
        stride = np.random.uniform(0.32, 0.70)
        freezing = np.random.uniform(0.40, 0.88)
        balance = np.random.uniform(0.20, 0.55)

        data.append([speed, stride, freezing, balance, "high"])

    df = pd.DataFrame(data, columns=[
        "walking_speed",
        "stride_variability",
        "freezing_index",
        "balance_score",
        "risk_level",
    ])

    # Add significant noise to create overlap
    df["walking_speed"] = np.clip(df["walking_speed"] + np.random.normal(0, 0.1, len(df)), 0.3, 1.5).round(2)
    df["stride_variability"] = np.clip(df["stride_variability"] + np.random.normal(0, 0.06, len(df)), 0.03, 0.7).round(3)
    df["freezing_index"] = np.clip(df["freezing_index"] + np.random.normal(0, 0.07, len(df)), 0.0, 0.9).round(3)
    df["balance_score"] = np.clip(df["balance_score"] + np.random.normal(0, 0.08, len(df)), 0.2, 1.0).round(2)

    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    path = os.path.join(OUTPUT_DIR, "gait_agent_dataset.csv")
    df.to_csv(path, index=False)
    print(f"Gait dataset: {len(df)} samples -> {path}")
    print(f"  Class distribution: {df['risk_level'].value_counts().to_dict()}")

    return df


def generate_progression_dataset(n=N_SAMPLES):
    """
    Progression Agent Dataset.

    Features (derived from coordinator output + patient data):
    - baseline_risk_score (0.0 - 1.0, from coordinator)
    - disease_duration_years (0 - 15)
    - medication_response_encoded (0=good, 1=moderate, 2=poor)
    - clinical_risk_score (0.0 - 1.0)
    - gait_risk_score (0.0 - 1.0)
    - age (40 - 85)

    Target: progression_category (stable, slow_progression, moderate_progression, rapid_progression)
    """

    data = []

    # Stable (35%)
    for _ in range(int(n * 0.35)):
        baseline = np.random.uniform(0.05, 0.40)
        duration = np.random.uniform(0, 3)
        med = np.random.choice([0, 1], p=[0.7, 0.3])
        clinical = np.random.uniform(0.05, 0.40)
        gait = np.random.uniform(0.05, 0.35)
        age = np.random.uniform(40, 65)

        data.append([baseline, duration, med, clinical, gait, age, "stable"])

    # Slow progression (25%)
    for _ in range(int(n * 0.25)):
        baseline = np.random.uniform(0.30, 0.55)
        duration = np.random.uniform(2, 5)
        med = np.random.choice([0, 1, 2], p=[0.3, 0.5, 0.2])
        clinical = np.random.uniform(0.30, 0.55)
        gait = np.random.uniform(0.25, 0.50)
        age = np.random.uniform(50, 72)

        data.append([baseline, duration, med, clinical, gait, age, "slow_progression"])

    # Moderate progression (25%)
    for _ in range(int(n * 0.25)):
        baseline = np.random.uniform(0.45, 0.72)
        duration = np.random.uniform(4, 8)
        med = np.random.choice([1, 2], p=[0.5, 0.5])
        clinical = np.random.uniform(0.45, 0.72)
        gait = np.random.uniform(0.40, 0.68)
        age = np.random.uniform(58, 78)

        data.append([baseline, duration, med, clinical, gait, age, "moderate_progression"])

    # Rapid progression (15%)
    for _ in range(int(n * 0.15)):
        baseline = np.random.uniform(0.60, 0.95)
        duration = np.random.uniform(6, 15)
        med = np.random.choice([1, 2], p=[0.2, 0.8])
        clinical = np.random.uniform(0.60, 0.92)
        gait = np.random.uniform(0.55, 0.90)
        age = np.random.uniform(65, 85)

        data.append([baseline, duration, med, clinical, gait, age, "rapid_progression"])

    df = pd.DataFrame(data, columns=[
        "baseline_risk_score",
        "disease_duration_years",
        "medication_response_encoded",
        "clinical_risk_score",
        "gait_risk_score",
        "age",
        "progression_category",
    ])

    # Add noise
    df["baseline_risk_score"] = np.clip(df["baseline_risk_score"] + np.random.normal(0, 0.04, len(df)), 0, 1).round(3)
    df["clinical_risk_score"] = np.clip(df["clinical_risk_score"] + np.random.normal(0, 0.04, len(df)), 0, 1).round(3)
    df["gait_risk_score"] = np.clip(df["gait_risk_score"] + np.random.normal(0, 0.04, len(df)), 0, 1).round(3)
    df["disease_duration_years"] = np.clip(df["disease_duration_years"] + np.random.normal(0, 0.5, len(df)), 0, 15).round(1)
    df["age"] = np.clip(df["age"] + np.random.normal(0, 2, len(df)), 40, 85).round(0).astype(int)

    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    path = os.path.join(OUTPUT_DIR, "progression_agent_dataset.csv")
    df.to_csv(path, index=False)
    print(f"Progression dataset: {len(df)} samples -> {path}")
    print(f"  Class distribution: {df['progression_category'].value_counts().to_dict()}")

    return df


if __name__ == "__main__":
    print("=" * 60)
    print("NeuroAgent-PD: Generating Training Datasets")
    print("=" * 60)
    print()

    clinical_df = generate_clinical_dataset()
    print()
    speech_df = generate_speech_dataset()
    print()
    gait_df = generate_gait_dataset()
    print()
    progression_df = generate_progression_dataset()

    print()
    print("=" * 60)
    print("All datasets generated successfully.")
    print("=" * 60)
    print()
    print("Next step: Run train_models.py to train ML models from these datasets.")
