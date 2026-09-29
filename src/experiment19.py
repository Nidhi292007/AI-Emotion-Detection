import os
import sys
import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets import load_dataset
from src.preprocessing import preprocess_text
import joblib


# =========================
# LOAD MODEL + VECTORIZERS
# =========================

model = joblib.load("emotion_model.pkl")

word_vectorizer = joblib.load("word_tfidf_vectorizer.pkl")
char_vectorizer = joblib.load("char_tfidf_vectorizer.pkl")
char_weight = joblib.load("char_weight.pkl")

raw_char_vectorizer = joblib.load("raw_char_tfidf_vectorizer.pkl")
raw_char_weight = joblib.load("raw_char_weight.pkl")


# =========================
# LOAD TEST DATA
# =========================

dataset = load_dataset("dair-ai/emotion")

test_texts = dataset["test"]["text"]
test_labels = np.array(dataset["test"]["label"])


# =========================
# PREPROCESS
# =========================

processed_texts = [
    preprocess_text(text)
    for text in test_texts
]


# =========================
# TRANSFORM
# =========================

X_word = word_vectorizer.transform(processed_texts)

X_char = char_vectorizer.transform(processed_texts)

X_raw_char = raw_char_vectorizer.transform(test_texts)


# =========================
# COMBINE FEATURES
# =========================

from scipy.sparse import hstack

X_test = hstack([
    X_word,
    X_char * char_weight,
    X_raw_char * raw_char_weight
])


# =========================
# PREDICTIONS
# =========================

predictions = model.predict(X_test)

probabilities = model.predict_proba(X_test)

top1_confidence = np.max(probabilities, axis=1)

sorted_probabilities = np.sort(probabilities, axis=1)

top2_confidence = sorted_probabilities[:, -2]

margin = top1_confidence - top2_confidence

correct = predictions == test_labels


# =========================
# BASIC RESULTS
# =========================

total = len(test_labels)
correct_count = np.sum(correct)
incorrect_count = total - correct_count

print("\n======================================")
print("EXPERIMENT 19")
print("CONFIDENCE-BASED ERROR ANALYSIS")
print("======================================")

print(f"\nTotal test samples : {total}")
print(f"Correct            : {correct_count}")
print(f"Incorrect          : {incorrect_count}")
print(f"Accuracy            : {correct_count / total * 100:.2f}%")

print(f"\nAverage confidence : {top1_confidence.mean() * 100:.2f}%")
print(f"Average margin     : {margin.mean() * 100:.2f}%")


# =========================
# THRESHOLD ANALYSIS
# =========================

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

results = []

print("\n======================================")
print("CONFIDENCE THRESHOLD ANALYSIS")
print("======================================")

print(
    f"{'Threshold':<12}"
    f"{'Coverage':<12}"
    f"{'Samples':<10}"
    f"{'Accuracy':<12}"
    f"{'Errors Rejected':<18}"
)

for threshold in thresholds:

    accepted = top1_confidence >= threshold

    accepted_count = np.sum(accepted)

    if accepted_count == 0:
        continue

    accepted_correct = np.sum(correct[accepted])

    accepted_accuracy = accepted_correct / accepted_count

    coverage = accepted_count / total

    rejected = ~accepted

    rejected_errors = np.sum((~correct) & rejected)

    error_rejection_rate = (
        rejected_errors / incorrect_count
        if incorrect_count > 0
        else 0
    )

    print(
        f"{threshold:<12.2f}"
        f"{coverage * 100:<12.2f}"
        f"{accepted_count:<10}"
        f"{accepted_accuracy * 100:<12.2f}"
        f"{error_rejection_rate * 100:<18.2f}"
    )

    results.append({
        "threshold": threshold,
        "coverage": coverage,
        "samples": accepted_count,
        "accuracy": accepted_accuracy,
        "errors_rejected": rejected_errors,
        "error_rejection_rate": error_rejection_rate
    })


# =========================
# CORRECT VS INCORRECT
# =========================

print("\n======================================")
print("CORRECT VS INCORRECT CONFIDENCE")
print("======================================")

print(
    f"Correct predictions   : "
    f"{top1_confidence[correct].mean() * 100:.2f}%"
)

print(
    f"Incorrect predictions : "
    f"{top1_confidence[~correct].mean() * 100:.2f}%"
)

print(
    f"\nCorrect margin   : "
    f"{margin[correct].mean() * 100:.2f}%"
)

print(
    f"Incorrect margin : "
    f"{margin[~correct].mean() * 100:.2f}%"
)


# =========================
# LOW CONFIDENCE ERRORS
# =========================

print("\n======================================")
print("LOW-CONFIDENCE ERRORS")
print("======================================")

error_indices = np.where(~correct)[0]

error_indices = sorted(
    error_indices,
    key=lambda i: top1_confidence[i]
)

label_names = [
    "Sadness",
    "Joy",
    "Love",
    "Anger",
    "Fear",
    "Surprise"
]

print("\n10 lowest-confidence errors:\n")

for i in error_indices[:10]:

    print("--------------------------------------")

    print(f"Text       : {test_texts[i]}")
    print(f"Actual     : {label_names[test_labels[i]]}")
    print(f"Predicted  : {label_names[predictions[i]]}")
    print(
        f"Confidence : "
        f"{top1_confidence[i] * 100:.2f}%"
    )

    print(
        f"Top-2     : "
        f"{top2_confidence[i] * 100:.2f}%"
    )

    print(
        f"Margin    : "
        f"{margin[i] * 100:.2f}%"
    )


# =========================
# SAVE ANALYSIS
# =========================

results_df = pd.DataFrame(results)

results_df.to_csv(
    "experiment19_confidence_analysis.csv",
    index=False
)

print("\n======================================")
print("Experiment 19 completed.")
print("No model files were modified.")
print("Saved: experiment19_confidence_analysis.csv")
print("======================================")