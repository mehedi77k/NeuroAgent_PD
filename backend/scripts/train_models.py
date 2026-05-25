"""
Model Training Script for NeuroAgent-PD.

Trains ML models for clinical, speech, gait, and progression agents
using the generated synthetic datasets.

Models are saved to backend/artifacts/ and can be loaded by the agents.

Usage:
    python backend/scripts/train_models.py
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

DATASETS_DIR = os.path.join(os.path.dirname(__file__), "..", "datasets")
ARTIFACTS_DIR = os.path.join(os.path.dirname(__file__), "..", "artifacts")

os.makedirs(ARTIFACTS_DIR, exist_ok=True)


def train_clinical_model():
    """Train clinical agent model."""

    print("=" * 60)
    print("Training Clinical Agent Model")
    print("=" * 60)

    df = pd.read_csv(os.path.join(DATASETS_DIR, "clinical_agent_dataset.csv"))

    feature_cols = [
        "updrs_score",
        "tremor_score",
        "rigidity_score",
        "bradykinesia_score",
        "disease_duration_years",
        "medication_response_encoded",
    ]

    X = df[feature_cols].values
    y = df["risk_level"].values

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = GradientBoostingClassifier(
        n_estimators=80,
        max_depth=3,
        learning_rate=0.1,
        min_samples_leaf=15,
        random_state=42,
    )
    model.fit(X_train_scaled, y_train)

    # Calibrate probabilities for more realistic confidence
    calibrated_model = CalibratedClassifierCV(model, cv=5, method="isotonic")
    calibrated_model.fit(X_train_scaled, y_train)

    y_pred = calibrated_model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="macro")

    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"Test Macro F1: {f1:.4f}")
    print(f"Classes: {list(le.classes_)}")
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    # Save artifacts
    joblib.dump(calibrated_model, os.path.join(ARTIFACTS_DIR, "clinical_model.pkl"))
    joblib.dump(scaler, os.path.join(ARTIFACTS_DIR, "clinical_scaler.pkl"))
    joblib.dump(le, os.path.join(ARTIFACTS_DIR, "clinical_label_encoder.pkl"))

    metadata = {
        "agent": "clinical_agent",
        "model_type": "GradientBoostingClassifier",
        "features": feature_cols,
        "classes": list(le.classes_),
        "test_accuracy": round(accuracy, 4),
        "test_f1_macro": round(f1, 4),
        "n_samples": len(df),
        "note": "Trained on synthetic data. Does not replace clinical validation.",
    }

    with open(os.path.join(ARTIFACTS_DIR, "clinical_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    print("Saved: clinical_model.pkl, clinical_scaler.pkl, clinical_label_encoder.pkl")
    print()


def train_speech_model():
    """Train speech agent model."""

    print("=" * 60)
    print("Training Speech Agent Model")
    print("=" * 60)

    df = pd.read_csv(os.path.join(DATASETS_DIR, "speech_agent_dataset.csv"))

    feature_cols = [
        "jitter",
        "shimmer",
        "hnr",
        "pitch_variation",
    ]

    X = df[feature_cols].values
    y = df["risk_level"].values

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=5,
        min_samples_leaf=10,
        random_state=42,
    )
    model.fit(X_train_scaled, y_train)

    # Calibrate probabilities
    calibrated_model = CalibratedClassifierCV(model, cv=5, method="isotonic")
    calibrated_model.fit(X_train_scaled, y_train)

    y_pred = calibrated_model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="macro")

    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"Test Macro F1: {f1:.4f}")
    print(f"Classes: {list(le.classes_)}")
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    # Save artifacts
    joblib.dump(calibrated_model, os.path.join(ARTIFACTS_DIR, "speech_model.pkl"))
    joblib.dump(scaler, os.path.join(ARTIFACTS_DIR, "speech_scaler.pkl"))
    joblib.dump(le, os.path.join(ARTIFACTS_DIR, "speech_label_encoder.pkl"))

    metadata = {
        "agent": "speech_agent",
        "model_type": "RandomForestClassifier",
        "features": feature_cols,
        "classes": list(le.classes_),
        "test_accuracy": round(accuracy, 4),
        "test_f1_macro": round(f1, 4),
        "n_samples": len(df),
        "note": "Trained on synthetic data. Does not replace clinical validation.",
    }

    with open(os.path.join(ARTIFACTS_DIR, "speech_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    print("Saved: speech_model.pkl, speech_scaler.pkl, speech_label_encoder.pkl")
    print()


def train_gait_model():
    """Train gait agent model."""

    print("=" * 60)
    print("Training Gait Agent Model")
    print("=" * 60)

    df = pd.read_csv(os.path.join(DATASETS_DIR, "gait_agent_dataset.csv"))

    feature_cols = [
        "walking_speed",
        "stride_variability",
        "freezing_index",
        "balance_score",
    ]

    X = df[feature_cols].values
    y = df["risk_level"].values

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = GradientBoostingClassifier(
        n_estimators=80,
        max_depth=3,
        learning_rate=0.1,
        min_samples_leaf=15,
        random_state=42,
    )
    model.fit(X_train_scaled, y_train)

    # Calibrate probabilities
    calibrated_model = CalibratedClassifierCV(model, cv=5, method="isotonic")
    calibrated_model.fit(X_train_scaled, y_train)

    y_pred = calibrated_model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="macro")

    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"Test Macro F1: {f1:.4f}")
    print(f"Classes: {list(le.classes_)}")
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    # Save artifacts
    joblib.dump(calibrated_model, os.path.join(ARTIFACTS_DIR, "gait_model.pkl"))
    joblib.dump(scaler, os.path.join(ARTIFACTS_DIR, "gait_scaler.pkl"))
    joblib.dump(le, os.path.join(ARTIFACTS_DIR, "gait_label_encoder.pkl"))

    metadata = {
        "agent": "gait_agent",
        "model_type": "GradientBoostingClassifier",
        "features": feature_cols,
        "classes": list(le.classes_),
        "test_accuracy": round(accuracy, 4),
        "test_f1_macro": round(f1, 4),
        "n_samples": len(df),
        "note": "Trained on synthetic data. Does not replace clinical validation.",
    }

    with open(os.path.join(ARTIFACTS_DIR, "gait_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    print("Saved: gait_model.pkl, gait_scaler.pkl, gait_label_encoder.pkl")
    print()


def train_progression_model():
    """Train progression agent model."""

    print("=" * 60)
    print("Training Progression Agent Model")
    print("=" * 60)

    df = pd.read_csv(os.path.join(DATASETS_DIR, "progression_agent_dataset.csv"))

    feature_cols = [
        "baseline_risk_score",
        "disease_duration_years",
        "medication_response_encoded",
        "clinical_risk_score",
        "gait_risk_score",
        "age",
    ]

    X = df[feature_cols].values
    y = df["progression_category"].values

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = GradientBoostingClassifier(
        n_estimators=80,
        max_depth=3,
        learning_rate=0.1,
        min_samples_leaf=15,
        random_state=42,
    )
    model.fit(X_train_scaled, y_train)

    # Calibrate probabilities
    calibrated_model = CalibratedClassifierCV(model, cv=5, method="isotonic")
    calibrated_model.fit(X_train_scaled, y_train)

    y_pred = calibrated_model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="macro")

    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"Test Macro F1: {f1:.4f}")
    print(f"Classes: {list(le.classes_)}")
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    # Save artifacts
    joblib.dump(calibrated_model, os.path.join(ARTIFACTS_DIR, "progression_model.pkl"))
    joblib.dump(scaler, os.path.join(ARTIFACTS_DIR, "progression_scaler.pkl"))
    joblib.dump(le, os.path.join(ARTIFACTS_DIR, "progression_label_encoder.pkl"))

    metadata = {
        "agent": "progression_agent",
        "model_type": "GradientBoostingClassifier",
        "features": feature_cols,
        "classes": list(le.classes_),
        "test_accuracy": round(accuracy, 4),
        "test_f1_macro": round(f1, 4),
        "n_samples": len(df),
        "note": "Trained on synthetic data. Does not replace clinical validation.",
    }

    with open(os.path.join(ARTIFACTS_DIR, "progression_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    print("Saved: progression_model.pkl, progression_scaler.pkl, progression_label_encoder.pkl")
    print()


if __name__ == "__main__":
    print()
    print("NeuroAgent-PD: Model Training Pipeline")
    print()

    train_clinical_model()
    train_speech_model()
    train_gait_model()
    train_progression_model()

    print("=" * 60)
    print("All models trained and saved to backend/artifacts/")
    print("=" * 60)
