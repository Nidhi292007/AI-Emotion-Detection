import os
import sys
import numpy as np
import pandas as pd
import joblib

from datasets import load_dataset
from scipy.optimize import minimize_scalar
from scipy.sparse import hstack
from sklearn.metrics import (
    accuracy_score,
    log_loss
)


# ============================================================
# EXPERIMENT 26
# POST-HOC PROBABILITY CALIBRATION
# ============================================================

print("=" * 70)
print("EXPERIMENT 26 - PROBABILITY CALIBRATION")
print("=" * 70)


# ============================================================
# 1. PATH SETUP
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)


MODEL_PATH = os.path.join(
    BASE_DIR,
    "experiment22_emotion_model.pkl"
)

WORD_VECTOR_PATH = os.path.join(
    BASE_DIR,
    "word_tfidf_vectorizer.pkl"
)

CHAR_VECTOR_PATH = os.path.join(
    BASE_DIR,
    "char_tfidf_vectorizer.pkl"
)

CHAR_WEIGHT_PATH = os.path.join(
    BASE_DIR,
    "char_weight.pkl"
)

RAW_CHAR_VECTOR_PATH = os.path.join(
    BASE_DIR,
    "raw_char_tfidf_vectorizer.pkl"
)

RAW_CHAR_WEIGHT_PATH = os.path.join(
    BASE_DIR,
    "raw_char_weight.pkl"
)

RAW_WORD_VECTOR_PATH = os.path.join(
    BASE_DIR,
    "experiment22_raw_word_vectorizer.pkl"
)

RAW_WORD_WEIGHT_PATH = os.path.join(
    BASE_DIR,
    "experiment22_raw_word_weight.pkl"
)


# ============================================================
# 2. IMPORT PREPROCESSING
# ============================================================

sys.path.append(
    os.path.join(BASE_DIR, "src")
)

from preprocessing import preprocess_text


# ============================================================
# 3. LOAD MODEL
# ============================================================

print("\nLoading Experiment 22 model...")

model = joblib.load(
    MODEL_PATH
)

word_vectorizer = joblib.load(
    WORD_VECTOR_PATH
)

char_vectorizer = joblib.load(
    CHAR_VECTOR_PATH
)

char_weight = joblib.load(
    CHAR_WEIGHT_PATH
)

raw_char_vectorizer = joblib.load(
    RAW_CHAR_VECTOR_PATH
)

raw_char_weight = joblib.load(
    RAW_CHAR_WEIGHT_PATH
)

raw_word_vectorizer = joblib.load(
    RAW_WORD_VECTOR_PATH
)

raw_word_weight = joblib.load(
    RAW_WORD_WEIGHT_PATH
)

print("Model and vectorizers loaded.")


# ============================================================
# 4. LOAD DATASET
# ============================================================

print("\nLoading Dair-AI Emotion dataset...")

dataset = load_dataset(
    "dair-ai/emotion"
)

train_data = dataset["train"]
validation_data = dataset["validation"]
test_data = dataset["test"]


validation_texts = validation_data["text"]
y_validation = np.array(
    validation_data["label"]
)

test_texts = test_data["text"]
y_test = np.array(
    test_data["label"]
)


print(
    "Validation samples:",
    len(validation_texts)
)

print(
    "Test samples:",
    len(test_texts)
)


# ============================================================
# 5. FEATURE CREATION FUNCTION
# ============================================================

def create_features(texts):

    print(
        f"Creating features for {len(texts)} samples..."
    )

    processed_texts = [
        preprocess_text(text)
        for text in texts
    ]

    # -------------------------
    # Word TF-IDF
    # -------------------------

    X_word = word_vectorizer.transform(
        processed_texts
    )

    # -------------------------
    # Processed character TF-IDF
    # -------------------------

    X_char = char_vectorizer.transform(
        processed_texts
    )

    X_char = X_char * char_weight

    # -------------------------
    # Raw character TF-IDF
    # -------------------------

    X_raw_char = raw_char_vectorizer.transform(
        texts
    )

    X_raw_char = X_raw_char * raw_char_weight

    # -------------------------
    # Raw word TF-IDF
    # -------------------------

    X_raw_word = raw_word_vectorizer.transform(
        texts
    )

    X_raw_word = X_raw_word * raw_word_weight

    # -------------------------
    # Combine
    # -------------------------

    X = hstack([
        X_word,
        X_char,
        X_raw_char,
        X_raw_word
    ])

    print(
        "Feature matrix:",
        X.shape
    )

    return X


# ============================================================
# 6. CREATE VALIDATION FEATURES
# ============================================================

print("\n" + "=" * 70)
print("VALIDATION FEATURES")
print("=" * 70)

X_validation = create_features(
    validation_texts
)


# ============================================================
# 7. CREATE TEST FEATURES
# ============================================================

print("\n" + "=" * 70)
print("TEST FEATURES")
print("=" * 70)

X_test = create_features(
    test_texts
)


# ============================================================
# 8. GET ORIGINAL PROBABILITIES
# ============================================================

print("\nGenerating probabilities...")

validation_probabilities = (
    model.predict_proba(
        X_validation
    )
)

test_probabilities = (
    model.predict_proba(
        X_test
    )
)


# ============================================================
# 9. BASELINE TEST ACCURACY
# ============================================================

baseline_predictions = np.argmax(
    test_probabilities,
    axis=1
)

baseline_accuracy = accuracy_score(
    y_test,
    baseline_predictions
)


print("\n" + "=" * 70)
print("BASELINE")
print("=" * 70)

print(
    f"Test Accuracy: "
    f"{baseline_accuracy * 100:.2f}%"
)


# ============================================================
# 10. ECE FUNCTION
# ============================================================

def calculate_ece(
    y_true,
    probabilities,
    bins=10
):

    predictions = np.argmax(
        probabilities,
        axis=1
    )

    confidence = np.max(
        probabilities,
        axis=1
    )

    correct = (
        predictions == y_true
    )

    bin_edges = np.linspace(
        0,
        1,
        bins + 1
    )

    ece = 0.0

    for i in range(bins):

        lower = bin_edges[i]
        upper = bin_edges[i + 1]

        if i == bins - 1:

            mask = (
                (confidence >= lower) &
                (confidence <= upper)
            )

        else:

            mask = (
                (confidence >= lower) &
                (confidence < upper)
            )

        count = np.sum(mask)

        if count == 0:
            continue

        avg_confidence = np.mean(
            confidence[mask]
        )

        accuracy = np.mean(
            correct[mask]
        )

        ece += (
            count / len(y_true)
        ) * abs(
            avg_confidence - accuracy
        )

    return ece


# ============================================================
# 11. BRIER SCORE
# ============================================================

def calculate_brier_score(
    y_true,
    probabilities
):

    one_hot = np.zeros_like(
        probabilities
    )

    for i, label in enumerate(y_true):

        one_hot[
            i,
            label
        ] = 1

    return np.mean(
        np.sum(
            (
                probabilities -
                one_hot
            ) ** 2,
            axis=1
        )
    )


# ============================================================
# 12. TEMPERATURE SCALING
# ============================================================

def temperature_scale(
    probabilities,
    temperature
):

    # Convert probabilities to log space
    log_probabilities = np.log(
        np.clip(
            probabilities,
            1e-12,
            1.0
        )
    )

    # Apply temperature
    scaled_logits = (
        log_probabilities /
        temperature
    )

    # Numerical stability
    scaled_logits -= np.max(
        scaled_logits,
        axis=1,
        keepdims=True
    )

    exp_values = np.exp(
        scaled_logits
    )

    calibrated = (
        exp_values /
        np.sum(
            exp_values,
            axis=1,
            keepdims=True
        )
    )

    return calibrated


# ============================================================
# 13. FIND BEST TEMPERATURE USING VALIDATION SET
# ============================================================

print("\n" + "=" * 70)
print("FINDING OPTIMAL TEMPERATURE")
print("=" * 70)

print(
    "Calibration is fitted ONLY on validation data."
)


def validation_loss(temperature):

    calibrated_probabilities = (
        temperature_scale(
            validation_probabilities,
            temperature
        )
    )

    return log_loss(
        y_validation,
        calibrated_probabilities
    )


result = minimize_scalar(
    validation_loss,
    bounds=(0.05, 5.0),
    method="bounded"
)


best_temperature = result.x


print(
    f"\nOptimal Temperature: "
    f"{best_temperature:.4f}"
)

print(
    f"Validation Log Loss: "
    f"{result.fun:.6f}"
)


# ============================================================
# 14. APPLY CALIBRATION TO TEST SET
# ============================================================

print("\nApplying calibration to untouched test set...")

calibrated_test_probabilities = (
    temperature_scale(
        test_probabilities,
        best_temperature
    )
)


# ============================================================
# 15. CALIBRATED PREDICTIONS
# ============================================================

calibrated_predictions = np.argmax(
    calibrated_test_probabilities,
    axis=1
)

calibrated_accuracy = accuracy_score(
    y_test,
    calibrated_predictions
)


# ============================================================
# 16. METRICS
# ============================================================

baseline_ece = calculate_ece(
    y_test,
    test_probabilities
)

calibrated_ece = calculate_ece(
    y_test,
    calibrated_test_probabilities
)


baseline_brier = calculate_brier_score(
    y_test,
    test_probabilities
)

calibrated_brier = calculate_brier_score(
    y_test,
    calibrated_test_probabilities
)


baseline_logloss = log_loss(
    y_test,
    test_probabilities
)

calibrated_logloss = log_loss(
    y_test,
    calibrated_test_probabilities
)


# ============================================================
# 17. CONFIDENCE
# ============================================================

baseline_confidence = np.max(
    test_probabilities,
    axis=1
)

calibrated_confidence = np.max(
    calibrated_test_probabilities,
    axis=1
)


# ============================================================
# 18. HIGH CONFIDENCE ERRORS
# ============================================================

baseline_high_conf_errors = np.sum(
    (
        baseline_predictions != y_test
    )
    &
    (
        baseline_confidence >= 0.80
    )
)

calibrated_high_conf_errors = np.sum(
    (
        calibrated_predictions != y_test
    )
    &
    (
        calibrated_confidence >= 0.80
    )
)


# ============================================================
# 19. DISPLAY RESULTS
# ============================================================

print("\n" + "=" * 70)
print("EXPERIMENT 26 RESULTS")
print("=" * 70)

print(
    f"\nBaseline Accuracy       : "
    f"{baseline_accuracy * 100:.2f}%"
)

print(
    f"Calibrated Accuracy     : "
    f"{calibrated_accuracy * 100:.2f}%"
)

print(
    f"\nBaseline ECE            : "
    f"{baseline_ece * 100:.2f}%"
)

print(
    f"Calibrated ECE          : "
    f"{calibrated_ece * 100:.2f}%"
)

print(
    f"\nBaseline Brier Score    : "
    f"{baseline_brier:.4f}"
)

print(
    f"Calibrated Brier Score  : "
    f"{calibrated_brier:.4f}"
)

print(
    f"\nBaseline Log Loss       : "
    f"{baseline_logloss:.4f}"
)

print(
    f"Calibrated Log Loss     : "
    f"{calibrated_logloss:.4f}"
)

print(
    f"\nBaseline Avg Confidence : "
    f"{np.mean(baseline_confidence) * 100:.2f}%"
)

print(
    f"Calibrated Avg Confidence: "
    f"{np.mean(calibrated_confidence) * 100:.2f}%"
)

print(
    f"\nBaseline High-Confidence Errors "
    f"(>=80%): "
    f"{baseline_high_conf_errors}"
)

print(
    f"Calibrated High-Confidence Errors "
    f"(>=80%): "
    f"{calibrated_high_conf_errors}"
)


# ============================================================
# 20. SAVE TEMPERATURE
# ============================================================

temperature_path = os.path.join(
    BASE_DIR,
    "experiment26_temperature.pkl"
)

joblib.dump(
    best_temperature,
    temperature_path
)


# ============================================================
# 21. SAVE RESULTS
# ============================================================

comparison_df = pd.DataFrame({

    "metric": [
        "Accuracy",
        "ECE",
        "Brier Score",
        "Log Loss",
        "Average Confidence",
        "High Confidence Errors >=80%"
    ],

    "baseline": [
        baseline_accuracy,
        baseline_ece,
        baseline_brier,
        baseline_logloss,
        np.mean(baseline_confidence),
        baseline_high_conf_errors
    ],

    "calibrated": [
        calibrated_accuracy,
        calibrated_ece,
        calibrated_brier,
        calibrated_logloss,
        np.mean(calibrated_confidence),
        calibrated_high_conf_errors
    ]
})


comparison_path = os.path.join(
    BASE_DIR,
    "experiment26_calibration_comparison.csv"
)

comparison_df.to_csv(
    comparison_path,
    index=False
)


# ============================================================
# 22. SAVE SAMPLE RESULTS
# ============================================================

sample_df = pd.DataFrame({

    "text": test_texts,

    "actual_label": y_test,

    "baseline_prediction":
        baseline_predictions,

    "baseline_confidence":
        baseline_confidence,

    "calibrated_prediction":
        calibrated_predictions,

    "calibrated_confidence":
        calibrated_confidence
})


sample_path = os.path.join(
    BASE_DIR,
    "experiment26_sample_calibration.csv"
)

sample_df.to_csv(
    sample_path,
    index=False
)


# ============================================================
# 23. FINAL CONCLUSION
# ============================================================

print("\n" + "=" * 70)
print("EXPERIMENT 26 CONCLUSION")
print("=" * 70)

if calibrated_ece < baseline_ece:

    print(
        "Calibration IMPROVED probability calibration."
    )

else:

    print(
        "Calibration did NOT improve ECE."
    )


if calibrated_brier < baseline_brier:

    print(
        "Brier score improved."
    )

else:

    print(
        "Brier score did NOT improve."
    )


print(
    "\nAccuracy should remain approximately the same "
    "because temperature scaling changes probabilities, "
    "not the underlying model."
)

print(
    "\nOptimal temperature:",
    round(best_temperature, 4)
)

print(
    "\nSaved:"
)

print(
    "1.",
    temperature_path
)

print(
    "2.",
    comparison_path
)

print(
    "3.",
    sample_path
)

print(
    "\nExperiment 26 completed successfully."
)

print("=" * 70)