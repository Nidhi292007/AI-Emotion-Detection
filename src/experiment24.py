import os
import sys
import numpy as np
import pandas as pd
import joblib

from datasets import load_dataset
from sklearn.metrics import accuracy_score, classification_report


# ============================================================
# EXPERIMENT 24
# Confidence-Aware Emotion Detection
# ============================================================

print("=" * 70)
print("EXPERIMENT 24 - CONFIDENCE-AWARE EMOTION DETECTION")
print("=" * 70)


# ============================================================
# 1. PATH SETUP
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = os.path.join(BASE_DIR, "experiment22_emotion_model.pkl")

WORD_VECTORIZER_PATH = os.path.join(
    BASE_DIR, "word_tfidf_vectorizer.pkl"
)

CHAR_VECTORIZER_PATH = os.path.join(
    BASE_DIR, "char_tfidf_vectorizer.pkl"
)

CHAR_WEIGHT_PATH = os.path.join(
    BASE_DIR, "char_weight.pkl"
)

RAW_CHAR_VECTORIZER_PATH = os.path.join(
    BASE_DIR, "raw_char_tfidf_vectorizer.pkl"
)

RAW_CHAR_WEIGHT_PATH = os.path.join(
    BASE_DIR, "raw_char_weight.pkl"
)

RAW_WORD_VECTORIZER_PATH = os.path.join(
    BASE_DIR, "experiment22_raw_word_vectorizer.pkl"
)

RAW_WORD_WEIGHT_PATH = os.path.join(
    BASE_DIR, "experiment22_raw_word_weight.pkl"
)


# ============================================================
# 2. IMPORT PREPROCESSING
# ============================================================

sys.path.append(os.path.join(BASE_DIR, "src"))

from preprocessing import preprocess_text


# ============================================================
# 3. LOAD MODEL AND VECTORIZERS
# ============================================================

print("\nLoading model and vectorizers...")

model = joblib.load(MODEL_PATH)

word_vectorizer = joblib.load(WORD_VECTORIZER_PATH)
char_vectorizer = joblib.load(CHAR_VECTORIZER_PATH)
char_weight = joblib.load(CHAR_WEIGHT_PATH)

raw_char_vectorizer = joblib.load(
    RAW_CHAR_VECTORIZER_PATH
)
raw_char_weight = joblib.load(
    RAW_CHAR_WEIGHT_PATH
)

raw_word_vectorizer = joblib.load(
    RAW_WORD_VECTORIZER_PATH
)
raw_word_weight = joblib.load(
    RAW_WORD_WEIGHT_PATH
)

print("Model loaded successfully.")

print("\nFeature weights:")
print("Processed character weight :", char_weight)
print("Raw character weight       :", raw_char_weight)
print("Raw word weight             :", raw_word_weight)


# ============================================================
# 4. LOAD DATASET
# ============================================================

print("\nLoading Dair-AI Emotion dataset...")

dataset = load_dataset("dair-ai/emotion")

test_data = dataset["test"]

texts = test_data["text"]
y_test = np.array(test_data["label"])

print("Test samples:", len(texts))


# ============================================================
# 5. PREPROCESS TEXT
# ============================================================

print("\nPreprocessing test data...")

processed_texts = [
    preprocess_text(text)
    for text in texts
]

print("Preprocessing completed.")


# ============================================================
# 6. CREATE FEATURE BRANCHES
# ============================================================

print("\nCreating feature representations...")

# ------------------------------------------------------------
# Word TF-IDF
# ------------------------------------------------------------

X_word = word_vectorizer.transform(
    processed_texts
)

print("Word features:", X_word.shape)


# ------------------------------------------------------------
# Processed character TF-IDF
# ------------------------------------------------------------

X_char = char_vectorizer.transform(
    processed_texts
)

X_char = X_char * char_weight

print("Processed char features:", X_char.shape)


# ------------------------------------------------------------
# Raw character TF-IDF
# ------------------------------------------------------------

X_raw_char = raw_char_vectorizer.transform(
    texts
)

X_raw_char = X_raw_char * raw_char_weight

print("Raw char features:", X_raw_char.shape)


# ------------------------------------------------------------
# Raw word TF-IDF
# ------------------------------------------------------------

X_raw_word = raw_word_vectorizer.transform(
    texts
)

X_raw_word = X_raw_word * raw_word_weight

print("Raw word features:", X_raw_word.shape)


# ============================================================
# 7. COMBINE FEATURES
# ============================================================

print("\nCombining feature branches...")

from scipy.sparse import hstack

X_test = hstack([
    X_word,
    X_char,
    X_raw_char,
    X_raw_word
])

print("Final feature matrix:", X_test.shape)


# ============================================================
# 8. PREDICT PROBABILITIES
# ============================================================

print("\nGenerating predictions...")

probabilities = model.predict_proba(X_test)

predictions = np.argmax(
    probabilities,
    axis=1
)


# ============================================================
# 9. EMOTION LABELS
# ============================================================

emotion_labels = [
    "Sadness",
    "Joy",
    "Love",
    "Anger",
    "Fear",
    "Surprise"
]


# ============================================================
# 10. TOP-1 / TOP-2 CONFIDENCE
# ============================================================

sorted_probabilities = np.sort(
    probabilities,
    axis=1
)

top1_confidence = sorted_probabilities[:, -1]

top2_confidence = sorted_probabilities[:, -2]

margin = (
    top1_confidence -
    top2_confidence
)


# ============================================================
# 11. BASIC PERFORMANCE
# ============================================================

accuracy = accuracy_score(
    y_test,
    predictions
)

print("\n" + "=" * 70)
print("BASELINE PERFORMANCE")
print("=" * 70)

print(
    f"Test Accuracy: {accuracy * 100:.2f}%"
)


# ============================================================
# 12. OVERALL CONFIDENCE STATISTICS
# ============================================================

print("\n" + "=" * 70)
print("CONFIDENCE ANALYSIS")
print("=" * 70)

print(
    f"Average Top-1 Confidence : "
    f"{np.mean(top1_confidence) * 100:.2f}%"
)

print(
    f"Average Top-2 Confidence : "
    f"{np.mean(top2_confidence) * 100:.2f}%"
)

print(
    f"Average Confidence Margin: "
    f"{np.mean(margin) * 100:.2f}%"
)


# ============================================================
# 13. CORRECT VS INCORRECT CONFIDENCE
# ============================================================

correct_mask = predictions == y_test
incorrect_mask = ~correct_mask

print("\nCorrect predictions:")
print(
    f"Count: {np.sum(correct_mask)}"
)

print(
    f"Average confidence: "
    f"{np.mean(top1_confidence[correct_mask]) * 100:.2f}%"
)

print("\nIncorrect predictions:")
print(
    f"Count: {np.sum(incorrect_mask)}"
)

print(
    f"Average confidence: "
    f"{np.mean(top1_confidence[incorrect_mask]) * 100:.2f}%"
)


# ============================================================
# 14. THRESHOLD ANALYSIS
# ============================================================

print("\n" + "=" * 70)
print("CONFIDENCE THRESHOLD ANALYSIS")
print("=" * 70)

thresholds = [
    0.20,
    0.25,
    0.30,
    0.35,
    0.40,
    0.45,
    0.50,
    0.55,
    0.60,
    0.65,
    0.70,
    0.75,
    0.80,
    0.85,
    0.90
]

threshold_results = []

for threshold in thresholds:

    accepted = top1_confidence >= threshold

    accepted_count = np.sum(accepted)

    rejected_count = len(y_test) - accepted_count

    coverage = accepted_count / len(y_test)

    if accepted_count > 0:

        selective_accuracy = accuracy_score(
            y_test[accepted],
            predictions[accepted]
        )

    else:

        selective_accuracy = 0

    if rejected_count > 0:

        rejected_error_rate = np.mean(
            incorrect_mask[~accepted]
        )

    else:

        rejected_error_rate = 0

    threshold_results.append({
        "threshold": threshold,
        "accepted_samples": accepted_count,
        "rejected_samples": rejected_count,
        "coverage": coverage,
        "selective_accuracy": selective_accuracy,
        "rejected_error_rate": rejected_error_rate
    })

    print(
        f"Threshold {threshold:.2f} | "
        f"Coverage {coverage * 100:6.2f}% | "
        f"Accuracy {selective_accuracy * 100:6.2f}% | "
        f"Rejected {rejected_count}"
    )


# ============================================================
# 15. CONFIDENCE BINS
# ============================================================

print("\n" + "=" * 70)
print("CONFIDENCE BINS")
print("=" * 70)

bins = [
    (0.00, 0.20),
    (0.20, 0.40),
    (0.40, 0.60),
    (0.60, 0.80),
    (0.80, 1.01)
]

confidence_bin_results = []

for lower, upper in bins:

    mask = (
        (top1_confidence >= lower) &
        (top1_confidence < upper)
    )

    count = np.sum(mask)

    if count > 0:

        bin_accuracy = accuracy_score(
            y_test[mask],
            predictions[mask]
        )

        avg_confidence = np.mean(
            top1_confidence[mask]
        )

    else:

        bin_accuracy = 0
        avg_confidence = 0

    confidence_bin_results.append({
        "confidence_range":
            f"{lower:.2f}-{min(upper, 1.0):.2f}",
        "sample_count": count,
        "average_confidence":
            avg_confidence,
        "accuracy":
            bin_accuracy
    })

    print(
        f"{lower:.2f}-{min(upper, 1.0):.2f} | "
        f"Samples: {count:4d} | "
        f"Avg Confidence: {avg_confidence * 100:6.2f}% | "
        f"Accuracy: {bin_accuracy * 100:6.2f}%"
    )


# ============================================================
# 16. CLASS-WISE CONFIDENCE
# ============================================================

print("\n" + "=" * 70)
print("CLASS-WISE CONFIDENCE")
print("=" * 70)

class_results = []

for class_id in range(len(emotion_labels)):

    mask = predictions == class_id

    count = np.sum(mask)

    if count > 0:

        avg_confidence = np.mean(
            top1_confidence[mask]
        )

    else:

        avg_confidence = 0

    class_results.append({
        "emotion": emotion_labels[class_id],
        "predicted_samples": count,
        "average_confidence":
            avg_confidence
    })

    print(
        f"{emotion_labels[class_id]:10s} | "
        f"Samples: {count:4d} | "
        f"Confidence: {avg_confidence * 100:.2f}%"
    )


# ============================================================
# 17. LOW-CONFIDENCE PREDICTIONS
# ============================================================

print("\n" + "=" * 70)
print("LOW-CONFIDENCE PREDICTIONS")
print("=" * 70)

LOW_CONFIDENCE_THRESHOLD = 0.40

low_confidence_indices = np.where(
    top1_confidence < LOW_CONFIDENCE_THRESHOLD
)[0]

print(
    "Low-confidence samples:",
    len(low_confidence_indices)
)

print("\nSample examples:\n")

for index in low_confidence_indices[:20]:

    predicted_label = emotion_labels[
        predictions[index]
    ]

    actual_label = emotion_labels[
        y_test[index]
    ]

    print(
        f"Text: {texts[index]}"
    )

    print(
        f"Actual: {actual_label} | "
        f"Predicted: {predicted_label} | "
        f"Confidence: "
        f"{top1_confidence[index] * 100:.2f}%"
    )

    print("-" * 70)


# ============================================================
# 18. SAVE THRESHOLD RESULTS
# ============================================================

threshold_df = pd.DataFrame(
    threshold_results
)

threshold_path = os.path.join(
    BASE_DIR,
    "experiment24_threshold_analysis.csv"
)

threshold_df.to_csv(
    threshold_path,
    index=False
)


# ============================================================
# 19. SAVE CONFIDENCE BIN RESULTS
# ============================================================

confidence_bin_df = pd.DataFrame(
    confidence_bin_results
)

confidence_bin_path = os.path.join(
    BASE_DIR,
    "experiment24_confidence_bins.csv"
)

confidence_bin_df.to_csv(
    confidence_bin_path,
    index=False
)


# ============================================================
# 20. SAVE CLASS CONFIDENCE
# ============================================================

class_df = pd.DataFrame(
    class_results
)

class_path = os.path.join(
    BASE_DIR,
    "experiment24_class_confidence.csv"
)

class_df.to_csv(
    class_path,
    index=False
)


# ============================================================
# 21. SAVE SAMPLE-LEVEL ANALYSIS
# ============================================================

sample_analysis = pd.DataFrame({
    "text": texts,
    "actual_label": [
        emotion_labels[i]
        for i in y_test
    ],
    "predicted_label": [
        emotion_labels[i]
        for i in predictions
    ],
    "top1_confidence": top1_confidence,
    "top2_confidence": top2_confidence,
    "confidence_margin": margin,
    "correct": correct_mask
})

sample_path = os.path.join(
    BASE_DIR,
    "experiment24_sample_confidence.csv"
)

sample_analysis.to_csv(
    sample_path,
    index=False
)


# ============================================================
# 22. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("EXPERIMENT 24 SUMMARY")
print("=" * 70)

print(
    f"Baseline test accuracy       : "
    f"{accuracy * 100:.2f}%"
)

print(
    f"Average top-1 confidence     : "
    f"{np.mean(top1_confidence) * 100:.2f}%"
)

print(
    f"Average top-2 confidence     : "
    f"{np.mean(top2_confidence) * 100:.2f}%"
)

print(
    f"Average confidence margin    : "
    f"{np.mean(margin) * 100:.2f}%"
)

print(
    f"Low-confidence (<40%) samples: "
    f"{len(low_confidence_indices)}"
)

print("\nFiles saved:")

print(
    "1.",
    threshold_path
)

print(
    "2.",
    confidence_bin_path
)

print(
    "3.",
    class_path
)

print(
    "4.",
    sample_path
)

print("\nExperiment 24 completed successfully.")
print("=" * 70)