import os
import sys
import numpy as np
import pandas as pd
import joblib

from datasets import load_dataset
from sklearn.metrics import accuracy_score, brier_score_loss


# ============================================================
# EXPERIMENT 25
# CONFIDENCE CALIBRATION & RELIABILITY ANALYSIS
# ============================================================

print("=" * 70)
print("EXPERIMENT 25 - CONFIDENCE CALIBRATION")
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

print("\nLoading model and vectorizers...")

model = joblib.load(MODEL_PATH)

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

print("All model components loaded successfully.")


# ============================================================
# 4. LOAD DATASET
# ============================================================

print("\nLoading Dair-AI Emotion dataset...")

dataset = load_dataset(
    "dair-ai/emotion"
)

test_data = dataset["test"]

texts = test_data["text"]

y_test = np.array(
    test_data["label"]
)

print(
    "Test samples:",
    len(texts)
)


# ============================================================
# 5. PREPROCESS
# ============================================================

print("\nPreprocessing text...")

processed_texts = [
    preprocess_text(text)
    for text in texts
]

print("Preprocessing completed.")


# ============================================================
# 6. CREATE FEATURES
# ============================================================

print("\nCreating feature representations...")


# Word TF-IDF
X_word = word_vectorizer.transform(
    processed_texts
)


# Processed character TF-IDF
X_char = char_vectorizer.transform(
    processed_texts
)

X_char = X_char * char_weight


# Raw character TF-IDF
X_raw_char = raw_char_vectorizer.transform(
    texts
)

X_raw_char = X_raw_char * raw_char_weight


# Raw word TF-IDF
X_raw_word = raw_word_vectorizer.transform(
    texts
)

X_raw_word = X_raw_word * raw_word_weight


print(
    "Word features:",
    X_word.shape
)

print(
    "Processed char features:",
    X_char.shape
)

print(
    "Raw char features:",
    X_raw_char.shape
)

print(
    "Raw word features:",
    X_raw_word.shape
)


# ============================================================
# 7. COMBINE FEATURES
# ============================================================

from scipy.sparse import hstack

X_test = hstack([
    X_word,
    X_char,
    X_raw_char,
    X_raw_word
])

print(
    "\nFinal feature matrix:",
    X_test.shape
)


# ============================================================
# 8. PREDICT PROBABILITIES
# ============================================================

print("\nGenerating probabilities...")

probabilities = model.predict_proba(
    X_test
)

predictions = np.argmax(
    probabilities,
    axis=1
)

print("Predictions generated.")


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
# 10. TOP-1 CONFIDENCE
# ============================================================

top1_confidence = np.max(
    probabilities,
    axis=1
)

correct = (
    predictions == y_test
)


# ============================================================
# 11. BASELINE ACCURACY
# ============================================================

accuracy = accuracy_score(
    y_test,
    predictions
)

print("\n" + "=" * 70)
print("BASELINE")
print("=" * 70)

print(
    f"Test Accuracy: {accuracy * 100:.2f}%"
)

print(
    f"Average Confidence: "
    f"{np.mean(top1_confidence) * 100:.2f}%"
)


# ============================================================
# 12. RELIABILITY / CALIBRATION BINS
# ============================================================

print("\n" + "=" * 70)
print("RELIABILITY ANALYSIS")
print("=" * 70)

# Ten confidence bins
bins = np.linspace(
    0.0,
    1.0,
    11
)

reliability_results = []

ece = 0.0

for i in range(
    len(bins) - 1
):

    lower = bins[i]
    upper = bins[i + 1]

    if i == len(bins) - 2:

        mask = (
            (top1_confidence >= lower) &
            (top1_confidence <= upper)
        )

    else:

        mask = (
            (top1_confidence >= lower) &
            (top1_confidence < upper)
        )

    count = np.sum(mask)

    if count > 0:

        avg_confidence = np.mean(
            top1_confidence[mask]
        )

        bin_accuracy = np.mean(
            correct[mask]
        )

        gap = abs(
            avg_confidence -
            bin_accuracy
        )

        ece += (
            count / len(y_test)
        ) * gap

    else:

        avg_confidence = 0.0
        bin_accuracy = 0.0
        gap = 0.0

    reliability_results.append({

        "confidence_range":
            f"{lower:.1f}-{upper:.1f}",

        "sample_count":
            count,

        "average_confidence":
            avg_confidence,

        "actual_accuracy":
            bin_accuracy,

        "calibration_gap":
            gap
    })

    print(
        f"{lower:.1f}-{upper:.1f} | "
        f"Samples: {count:4d} | "
        f"Confidence: "
        f"{avg_confidence * 100:6.2f}% | "
        f"Accuracy: "
        f"{bin_accuracy * 100:6.2f}% | "
        f"Gap: "
        f"{gap * 100:6.2f}%"
    )


# ============================================================
# 13. EXPECTED CALIBRATION ERROR
# ============================================================

print("\n" + "=" * 70)
print("EXPECTED CALIBRATION ERROR")
print("=" * 70)

print(
    f"ECE: {ece * 100:.2f}%"
)


# ============================================================
# 14. BRIER SCORE
# ============================================================

print("\n" + "=" * 70)
print("BRIER SCORE")
print("=" * 70)

# Convert labels to one-hot format
one_hot = np.zeros_like(
    probabilities
)

for i, label in enumerate(y_test):

    one_hot[
        i,
        label
    ] = 1


brier_score = np.mean(
    np.sum(
        (probabilities - one_hot) ** 2,
        axis=1
    )
)

print(
    f"Multiclass Brier Score: "
    f"{brier_score:.4f}"
)


# ============================================================
# 15. OVERCONFIDENT ERRORS
# ============================================================

print("\n" + "=" * 70)
print("HIGH-CONFIDENCE ERRORS")
print("=" * 70)

high_confidence_error_mask = (
    (~correct) &
    (top1_confidence >= 0.80)
)

high_confidence_error_count = np.sum(
    high_confidence_error_mask
)

print(
    "Incorrect predictions with "
    "confidence >= 80%:",
    high_confidence_error_count
)

if high_confidence_error_count > 0:

    indices = np.where(
        high_confidence_error_mask
    )[0]

    print("\nExamples:\n")

    for index in indices[:15]:

        print(
            f"Text: {texts[index]}"
        )

        print(
            f"Actual: "
            f"{emotion_labels[y_test[index]]}"
        )

        print(
            f"Predicted: "
            f"{emotion_labels[predictions[index]]}"
        )

        print(
            f"Confidence: "
            f"{top1_confidence[index] * 100:.2f}%"
        )

        print("-" * 70)


# ============================================================
# 16. CALIBRATION BY EMOTION
# ============================================================

print("\n" + "=" * 70)
print("CALIBRATION BY PREDICTED EMOTION")
print("=" * 70)

class_results = []

for class_id in range(
    len(emotion_labels)
):

    mask = (
        predictions == class_id
    )

    count = np.sum(mask)

    if count > 0:

        class_confidence = np.mean(
            top1_confidence[mask]
        )

        class_accuracy = np.mean(
            correct[mask]
        )

        class_gap = abs(
            class_confidence -
            class_accuracy
        )

    else:

        class_confidence = 0.0
        class_accuracy = 0.0
        class_gap = 0.0

    class_results.append({

        "emotion":
            emotion_labels[class_id],

        "predicted_samples":
            count,

        "average_confidence":
            class_confidence,

        "accuracy":
            class_accuracy,

        "calibration_gap":
            class_gap
    })

    print(
        f"{emotion_labels[class_id]:10s} | "
        f"Samples: {count:4d} | "
        f"Confidence: "
        f"{class_confidence * 100:6.2f}% | "
        f"Accuracy: "
        f"{class_accuracy * 100:6.2f}% | "
        f"Gap: "
        f"{class_gap * 100:6.2f}%"
    )


# ============================================================
# 17. CONFIDENCE LEVEL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("CONFIDENCE LEVEL SUMMARY")
print("=" * 70)

confidence_levels = [
    ("Low", 0.00, 0.40),
    ("Medium", 0.40, 0.70),
    ("High", 0.70, 1.01)
]

confidence_summary = []

for name, lower, upper in confidence_levels:

    mask = (
        (top1_confidence >= lower) &
        (top1_confidence < upper)
    )

    count = np.sum(mask)

    if count > 0:

        avg_confidence = np.mean(
            top1_confidence[mask]
        )

        level_accuracy = np.mean(
            correct[mask]
        )

    else:

        avg_confidence = 0.0
        level_accuracy = 0.0

    confidence_summary.append({

        "confidence_level": name,

        "sample_count": count,

        "average_confidence":
            avg_confidence,

        "accuracy":
            level_accuracy
    })

    print(
        f"{name:8s} | "
        f"Samples: {count:4d} | "
        f"Confidence: "
        f"{avg_confidence * 100:6.2f}% | "
        f"Accuracy: "
        f"{level_accuracy * 100:6.2f}%"
    )


# ============================================================
# 18. SAVE RELIABILITY RESULTS
# ============================================================

reliability_df = pd.DataFrame(
    reliability_results
)

reliability_path = os.path.join(
    BASE_DIR,
    "experiment25_reliability_analysis.csv"
)

reliability_df.to_csv(
    reliability_path,
    index=False
)


# ============================================================
# 19. SAVE CLASS CALIBRATION
# ============================================================

class_df = pd.DataFrame(
    class_results
)

class_path = os.path.join(
    BASE_DIR,
    "experiment25_class_calibration.csv"
)

class_df.to_csv(
    class_path,
    index=False
)


# ============================================================
# 20. SAVE CONFIDENCE SUMMARY
# ============================================================

confidence_df = pd.DataFrame(
    confidence_summary
)

confidence_path = os.path.join(
    BASE_DIR,
    "experiment25_confidence_levels.csv"
)

confidence_df.to_csv(
    confidence_path,
    index=False
)


# ============================================================
# 21. SAVE SAMPLE-LEVEL DATA
# ============================================================

sample_df = pd.DataFrame({

    "text": texts,

    "actual_emotion": [
        emotion_labels[i]
        for i in y_test
    ],

    "predicted_emotion": [
        emotion_labels[i]
        for i in predictions
    ],

    "confidence":
        top1_confidence,

    "correct":
        correct
})

sample_path = os.path.join(
    BASE_DIR,
    "experiment25_sample_calibration.csv"
)

sample_df.to_csv(
    sample_path,
    index=False
)


# ============================================================
# 22. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("EXPERIMENT 25 SUMMARY")
print("=" * 70)

print(
    f"Test Accuracy        : "
    f"{accuracy * 100:.2f}%"
)

print(
    f"Average Confidence   : "
    f"{np.mean(top1_confidence) * 100:.2f}%"
)

print(
    f"Expected Calibration Error (ECE): "
    f"{ece * 100:.2f}%"
)

print(
    f"Brier Score          : "
    f"{brier_score:.4f}"
)

print(
    f"High-confidence errors (>=80%): "
    f"{high_confidence_error_count}"
)

print("\nFiles saved:")

print(
    "1.",
    reliability_path
)

print(
    "2.",
    class_path
)

print(
    "3.",
    confidence_path
)

print(
    "4.",
    sample_path
)

print("\nExperiment 25 completed successfully.")

print("=" * 70)