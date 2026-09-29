# ================================================================
# EXPERIMENT 28 - FINAL ERROR ANALYSIS
# ================================================================

import os
import sys
import warnings

warnings.filterwarnings("ignore")

import joblib
import numpy as np
import pandas as pd

from datasets import load_dataset
from scipy.sparse import hstack
from sklearn.metrics import accuracy_score, confusion_matrix

# Add project root to path
sys.path.append(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)

from src.preprocessing import preprocess_text


print("=" * 70)
print("EXPERIMENT 28 - FINAL ERROR ANALYSIS")
print("=" * 70)


# ================================================================
# 1. LOAD EXPERIMENT 22 MODEL
# ================================================================

print("\nLoading Experiment 22 model...")

model = joblib.load(
    "experiment22_emotion_model.pkl"
)

word_vectorizer = joblib.load(
    "word_tfidf_vectorizer.pkl"
)

char_vectorizer = joblib.load(
    "char_tfidf_vectorizer.pkl"
)

char_weight = joblib.load(
    "char_weight.pkl"
)

raw_char_vectorizer = joblib.load(
    "raw_char_tfidf_vectorizer.pkl"
)

raw_char_weight = joblib.load(
    "raw_char_weight.pkl"
)

raw_word_vectorizer = joblib.load(
    "experiment22_raw_word_vectorizer.pkl"
)

raw_word_weight = joblib.load(
    "experiment22_raw_word_weight.pkl"
)

print("All components loaded.")


# ================================================================
# 2. LOAD TEST DATA
# ================================================================

print("\nLoading test dataset...")

dataset = load_dataset(
    "dair-ai/emotion"
)

test_data = dataset["test"]

texts = test_data["text"]
labels = np.array(
    test_data["label"]
)

# Use the same 2000 test samples
texts = texts[:2000]
labels = labels[:2000]

print(
    f"Test samples: {len(texts)}"
)


# ================================================================
# 3. EMOTION LABELS
# ================================================================

emotion_names = [
    "Sadness",
    "Joy",
    "Love",
    "Anger",
    "Fear",
    "Surprise"
]


# ================================================================
# 4. PREPROCESS TEXT
# ================================================================

print("\nPreprocessing text...")

processed_texts = [
    preprocess_text(text)
    for text in texts
]


# ================================================================
# 5. CREATE FEATURES
# ================================================================

print("\nCreating features...")


# ------------------------------------------------
# 5.1 Word TF-IDF
# ------------------------------------------------

X_word = word_vectorizer.transform(
    processed_texts
)


# ------------------------------------------------
# 5.2 Processed character TF-IDF
# ------------------------------------------------

X_char = char_vectorizer.transform(
    processed_texts
)


# ------------------------------------------------
# 5.3 Raw character TF-IDF
# ------------------------------------------------

X_raw_char = raw_char_vectorizer.transform(
    texts
)


# ------------------------------------------------
# 5.4 Raw word TF-IDF
# ------------------------------------------------

X_raw_word = raw_word_vectorizer.transform(
    texts
)


# ------------------------------------------------
# 5.5 Apply weights
# ------------------------------------------------

X_char = X_char.multiply(
    char_weight
)

X_raw_char = X_raw_char.multiply(
    raw_char_weight
)

X_raw_word = X_raw_word.multiply(
    raw_word_weight
)


# ------------------------------------------------
# 5.6 Combine sparse matrices
# ------------------------------------------------

X_test = hstack(
    [
        X_word,
        X_char,
        X_raw_char,
        X_raw_word
    ],
    format="csr"
)


print(
    f"Feature matrix: {X_test.shape}"
)

print(
    f"Feature matrix type: "
    f"{type(X_test)}"
)


# ================================================================
# 6. GENERATE PREDICTIONS
# ================================================================

print("\nGenerating predictions...")

probabilities = model.predict_proba(
    X_test
)

predictions = model.predict(
    X_test
)


# ================================================================
# 7. OVERALL PERFORMANCE
# ================================================================

accuracy = accuracy_score(
    labels,
    predictions
)

correct_mask = (
    predictions == labels
)

incorrect_mask = ~correct_mask

correct_count = int(
    np.sum(correct_mask)
)

incorrect_count = int(
    np.sum(incorrect_mask)
)


print("\n" + "=" * 70)
print("OVERALL PERFORMANCE")
print("=" * 70)

print(
    f"Accuracy       : "
    f"{accuracy * 100:.2f}%"
)

print(
    f"Correct        : "
    f"{correct_count}"
)

print(
    f"Incorrect      : "
    f"{incorrect_count}"
)


# ================================================================
# 8. TOP-1 / TOP-2 CONFIDENCE
# ================================================================

top2_indices = np.argsort(
    probabilities,
    axis=1
)[:, -2:]

top1_confidence = np.max(
    probabilities,
    axis=1
)

top2_confidence = np.sort(
    probabilities,
    axis=1
)[:, -2]

top1_predictions = np.argmax(
    probabilities,
    axis=1
)

margin = (
    top1_confidence -
    top2_confidence
)


# ================================================================
# 9. COMPLETE RESULTS DATAFRAME
# ================================================================

results = pd.DataFrame(
    {
        "text": texts,

        "true_label": labels,

        "true_emotion": [
            emotion_names[int(x)]
            for x in labels
        ],

        "predicted_label": predictions,

        "predicted_emotion": [
            emotion_names[int(x)]
            for x in predictions
        ],

        "confidence": top1_confidence,

        "second_best_confidence":
            top2_confidence,

        "margin": margin,

        "correct": correct_mask
    }
)


# ================================================================
# 10. MISCLASSIFIED SAMPLES
# ================================================================

errors = results[
    results["correct"] == False
].copy()

print("\n" + "=" * 70)
print("MISCLASSIFIED SAMPLES")
print("=" * 70)

print(
    f"Total errors: {len(errors)}"
)


# ================================================================
# 11. TOP CONFUSION PAIRS
# ================================================================

print("\n" + "=" * 70)
print("TOP CONFUSION PAIRS")
print("=" * 70)


confusion_pairs = (
    errors
    .groupby(
        [
            "true_emotion",
            "predicted_emotion"
        ]
    )
    .size()
    .reset_index(
        name="count"
    )
    .sort_values(
        "count",
        ascending=False
    )
)


# Do NOT use DataFrame.to_string()
# because it caused OSError [Errno 22]

for _, row in confusion_pairs.head(15).iterrows():

    true_emotion = str(
        row["true_emotion"]
    )

    predicted_emotion = str(
        row["predicted_emotion"]
    )

    count = int(
        row["count"]
    )

    print(
        f"{true_emotion:10s} -> "
        f"{predicted_emotion:10s} : "
        f"{count}"
    )


# ================================================================
# 12. PER-CLASS ERROR ANALYSIS
# ================================================================

print("\n" + "=" * 70)
print("PER-CLASS ERROR ANALYSIS")
print("=" * 70)


class_errors = []


for class_id, emotion in enumerate(
    emotion_names
):

    class_mask = (
        labels == class_id
    )

    total = int(
        np.sum(class_mask)
    )

    class_correct = int(
        np.sum(
            predictions[class_mask]
            == class_id
        )
    )

    class_incorrect = (
        total -
        class_correct
    )

    if total > 0:

        class_accuracy = (
            class_correct /
            total
        )

        error_rate = (
            class_incorrect /
            total
        )

    else:

        class_accuracy = 0

        error_rate = 0


    class_errors.append(
        {
            "emotion": emotion,

            "total_samples": total,

            "correct": class_correct,

            "errors": class_incorrect,

            "accuracy": class_accuracy,

            "error_rate": error_rate
        }
    )


class_errors_df = pd.DataFrame(
    class_errors
)


for _, row in class_errors_df.iterrows():

    print(
        f"{str(row['emotion']):10s} | "
        f"Total: "
        f"{int(row['total_samples']):4d} | "
        f"Correct: "
        f"{int(row['correct']):4d} | "
        f"Errors: "
        f"{int(row['errors']):4d} | "
        f"Accuracy: "
        f"{row['accuracy'] * 100:.2f}%"
    )


# ================================================================
# 13. LOW-CONFIDENCE ERRORS
# ================================================================

print("\n" + "=" * 70)
print("LOW-CONFIDENCE ERRORS (<40%)")
print("=" * 70)


low_confidence_errors = errors[
    errors["confidence"] < 0.40
].copy()


print(
    f"Low-confidence errors: "
    f"{len(low_confidence_errors)}"
)


# ================================================================
# 14. HIGH-CONFIDENCE ERRORS
# ================================================================

print("\n" + "=" * 70)
print("HIGH-CONFIDENCE ERRORS (>=80%)")
print("=" * 70)


high_confidence_errors = errors[
    errors["confidence"] >= 0.80
].copy()


print(
    f"High-confidence errors: "
    f"{len(high_confidence_errors)}"
)


# ================================================================
# 15. AMBIGUOUS ERRORS
# ================================================================

print("\n" + "=" * 70)
print("AMBIGUOUS ERRORS (MARGIN <20%)")
print("=" * 70)


ambiguous_errors = errors[
    errors["margin"] < 0.20
].copy()


print(
    f"Ambiguous errors: "
    f"{len(ambiguous_errors)}"
)


# ================================================================
# 16. SAMPLE MISCLASSIFICATIONS
# ================================================================

print("\n" + "=" * 70)
print("SAMPLE MISCLASSIFICATIONS")
print("=" * 70)


for i, (_, row) in enumerate(
    errors.head(20).iterrows()
):

    print("\n" + "-" * 70)

    print(
        f"Sample {i + 1}"
    )

    print(
        f"True emotion      : "
        f"{row['true_emotion']}"
    )

    print(
        f"Predicted emotion : "
        f"{row['predicted_emotion']}"
    )

    print(
        f"Confidence        : "
        f"{row['confidence'] * 100:.2f}%"
    )

    print(
        f"Margin            : "
        f"{row['margin'] * 100:.2f}%"
    )

    print(
        f"Text              : "
        f"{row['text']}"
    )


# ================================================================
# 17. CONFUSION MATRIX
# ================================================================

print("\n" + "=" * 70)
print("CONFUSION MATRIX")
print("=" * 70)


cm = confusion_matrix(
    labels,
    predictions
)


print(
    "              "
    "Sad Joy Love Anger Fear Surprise"
)


for i, emotion in enumerate(
    emotion_names
):

    row_values = " ".join(
        f"{int(x):4d}"
        for x in cm[i]
    )

    print(
        f"{emotion:10s}: "
        f"{row_values}"
    )


# ================================================================
# 18. SAVE ALL ANALYSIS FILES
# ================================================================

print("\n" + "=" * 70)
print("SAVING ANALYSIS FILES")
print("=" * 70)


results.to_csv(
    "experiment28_all_predictions.csv",
    index=False
)

errors.to_csv(
    "experiment28_misclassified_samples.csv",
    index=False
)

confusion_pairs.to_csv(
    "experiment28_confusion_pairs.csv",
    index=False
)

class_errors_df.to_csv(
    "experiment28_class_errors.csv",
    index=False
)

low_confidence_errors.to_csv(
    "experiment28_low_confidence_errors.csv",
    index=False
)

high_confidence_errors.to_csv(
    "experiment28_high_confidence_errors.csv",
    index=False
)

ambiguous_errors.to_csv(
    "experiment28_ambiguous_errors.csv",
    index=False
)


# ================================================================
# 19. FINAL SUMMARY
# ================================================================

print("\n" + "=" * 70)
print("EXPERIMENT 28 COMPLETE")
print("=" * 70)

print(
    f"Test Accuracy          : "
    f"{accuracy * 100:.2f}%"
)

print(
    f"Total Test Samples     : "
    f"{len(labels)}"
)

print(
    f"Correct Predictions    : "
    f"{correct_count}"
)

print(
    f"Incorrect Predictions  : "
    f"{incorrect_count}"
)

print(
    f"Misclassified Samples  : "
    f"{len(errors)}"
)

print(
    f"Low-confidence Errors  : "
    f"{len(low_confidence_errors)}"
)

print(
    f"High-confidence Errors : "
    f"{len(high_confidence_errors)}"
)

print(
    f"Ambiguous Errors       : "
    f"{len(ambiguous_errors)}"
)

print("\nSaved files:")

print(
    "  experiment28_all_predictions.csv"
)

print(
    "  experiment28_misclassified_samples.csv"
)

print(
    "  experiment28_confusion_pairs.csv"
)

print(
    "  experiment28_class_errors.csv"
)

print(
    "  experiment28_low_confidence_errors.csv"
)

print(
    "  experiment28_high_confidence_errors.csv"
)

print(
    "  experiment28_ambiguous_errors.csv"
)

print("\n" + "=" * 70)
print("DONE")
print("=" * 70)