import os
import sys
import numpy as np
import pandas as pd
import joblib

from datasets import load_dataset
from scipy.sparse import hstack
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score
)
from sklearn.model_selection import StratifiedKFold


# ============================================================
# EXPERIMENT 27
# FINAL RIGOROUS VALIDATION
# ============================================================

print("=" * 70)
print("EXPERIMENT 27 - FINAL RIGOROUS VALIDATION")
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
# 3. LOAD MODEL + VECTORIZERS
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

print("All model components loaded.")


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


train_texts = train_data["text"]
y_train = np.array(
    train_data["label"]
)

validation_texts = validation_data["text"]
y_validation = np.array(
    validation_data["label"]
)

test_texts = test_data["text"]
y_test = np.array(
    test_data["label"]
)


print(
    f"Training samples    : {len(train_texts)}"
)

print(
    f"Validation samples  : {len(validation_texts)}"
)

print(
    f"Test samples        : {len(test_texts)}"
)


# ============================================================
# 5. FEATURE CREATION
# ============================================================

def create_features(texts):

    processed_texts = [
        preprocess_text(text)
        for text in texts
    ]

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

    # Combine all four branches
    X = hstack([
        X_word,
        X_char,
        X_raw_char,
        X_raw_word
    ])

    return X


# ============================================================
# 6. VALIDATION FEATURES
# ============================================================

print("\nCreating validation features...")

X_validation = create_features(
    validation_texts
)

print(
    "Validation feature matrix:",
    X_validation.shape
)


# ============================================================
# 7. TEST FEATURES
# ============================================================

print("\nCreating test features...")

X_test = create_features(
    test_texts
)

print(
    "Test feature matrix:",
    X_test.shape
)


# ============================================================
# 8. VALIDATION PERFORMANCE
# ============================================================

print("\n" + "=" * 70)
print("VALIDATION SET PERFORMANCE")
print("=" * 70)

validation_predictions = model.predict(
    X_validation
)

validation_accuracy = accuracy_score(
    y_validation,
    validation_predictions
)

validation_macro_f1 = f1_score(
    y_validation,
    validation_predictions,
    average="macro"
)

validation_weighted_f1 = f1_score(
    y_validation,
    validation_predictions,
    average="weighted"
)


print(
    f"Validation Accuracy : "
    f"{validation_accuracy * 100:.2f}%"
)

print(
    f"Validation Macro F1 : "
    f"{validation_macro_f1 * 100:.2f}%"
)

print(
    f"Validation Weighted F1 : "
    f"{validation_weighted_f1 * 100:.2f}%"
)


# ============================================================
# 9. FINAL TEST EVALUATION
# ============================================================

print("\n" + "=" * 70)
print("FINAL TEST EVALUATION")
print("=" * 70)

test_predictions = model.predict(
    X_test
)

test_accuracy = accuracy_score(
    y_test,
    test_predictions
)

test_macro_f1 = f1_score(
    y_test,
    test_predictions,
    average="macro"
)

test_weighted_f1 = f1_score(
    y_test,
    test_predictions,
    average="weighted"
)


print(
    f"Test Accuracy : "
    f"{test_accuracy * 100:.2f}%"
)

print(
    f"Test Macro F1 : "
    f"{test_macro_f1 * 100:.2f}%"
)

print(
    f"Test Weighted F1 : "
    f"{test_weighted_f1 * 100:.2f}%"
)


# ============================================================
# 10. CLASSIFICATION REPORT
# ============================================================

emotion_names = [
    "Sadness",
    "Joy",
    "Love",
    "Anger",
    "Fear",
    "Surprise"
]


print("\n" + "=" * 70)
print("CLASSIFICATION REPORT")
print("=" * 70)

report = classification_report(
    y_test,
    test_predictions,
    target_names=emotion_names,
    digits=4
)

print(report)


# ============================================================
# 11. SAVE CLASSIFICATION REPORT
# ============================================================

report_dict = classification_report(
    y_test,
    test_predictions,
    target_names=emotion_names,
    output_dict=True
)

report_df = pd.DataFrame(
    report_dict
).transpose()

report_path = os.path.join(
    BASE_DIR,
    "experiment27_classification_report.csv"
)

report_df.to_csv(
    report_path
)


# ============================================================
# 12. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_test,
    test_predictions
)

print("\n" + "=" * 70)
print("CONFUSION MATRIX")
print("=" * 70)

print(cm)


# ============================================================
# 13. SAVE CONFUSION MATRIX
# ============================================================

cm_df = pd.DataFrame(
    cm,
    index=emotion_names,
    columns=emotion_names
)

cm_path = os.path.join(
    BASE_DIR,
    "experiment27_confusion_matrix.csv"
)

cm_df.to_csv(
    cm_path
)


# ============================================================
# 14. PER-CLASS ACCURACY
# ============================================================

print("\n" + "=" * 70)
print("PER-CLASS RESULTS")
print("=" * 70)

class_results = []

for i, emotion in enumerate(
    emotion_names
):

    total = np.sum(
        y_test == i
    )

    correct = cm[i, i]

    class_accuracy = (
        correct / total
        if total > 0
        else 0
    )

    print(
        f"{emotion:10s} : "
        f"{class_accuracy * 100:.2f}% "
        f"({correct}/{total})"
    )

    class_results.append({

        "emotion": emotion,

        "samples": total,

        "correct": correct,

        "accuracy": class_accuracy
    })


class_df = pd.DataFrame(
    class_results
)

class_path = os.path.join(
    BASE_DIR,
    "experiment27_class_results.csv"
)

class_df.to_csv(
    class_path,
    index=False
)


# ============================================================
# 15. COMPARISON WITH EXPERIMENT 22
# ============================================================

previous_accuracy = 0.9040

difference = (
    test_accuracy -
    previous_accuracy
)


print("\n" + "=" * 70)
print("COMPARISON WITH EXPERIMENT 22")
print("=" * 70)

print(
    f"Previous Experiment 22 : "
    f"{previous_accuracy * 100:.2f}%"
)

print(
    f"Experiment 27          : "
    f"{test_accuracy * 100:.2f}%"
)

print(
    f"Difference              : "
    f"{difference * 100:+.2f} percentage points"
)


# ============================================================
# 16. FINAL INTERPRETATION
# ============================================================

print("\n" + "=" * 70)
print("FINAL INTERPRETATION")
print("=" * 70)

if abs(difference) <= 0.005:

    print(
        "The Experiment 22 result is reproduced "
        "within a small tolerance."
    )

elif difference > 0:

    print(
        "Experiment 27 produced a higher test score."
    )

else:

    print(
        "Experiment 27 produced a lower test score."
    )


print(
    "\nThe test set was NOT used for model selection "
    "or hyperparameter tuning in this experiment."
)

print(
    "Experiment 22 remains the current selected model."
)


# ============================================================
# 17. SAVE SUMMARY
# ============================================================

summary_df = pd.DataFrame({

    "metric": [
        "Validation Accuracy",
        "Validation Macro F1",
        "Validation Weighted F1",
        "Test Accuracy",
        "Test Macro F1",
        "Test Weighted F1"
    ],

    "value": [
        validation_accuracy,
        validation_macro_f1,
        validation_weighted_f1,
        test_accuracy,
        test_macro_f1,
        test_weighted_f1
    ]
})


summary_path = os.path.join(
    BASE_DIR,
    "experiment27_final_validation_summary.csv"
)

summary_df.to_csv(
    summary_path,
    index=False
)


# ============================================================
# 18. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("EXPERIMENT 27 COMPLETE")
print("=" * 70)

print(
    f"Validation Accuracy : "
    f"{validation_accuracy * 100:.2f}%"
)

print(
    f"Final Test Accuracy : "
    f"{test_accuracy * 100:.2f}%"
)

print(
    f"Test Macro F1       : "
    f"{test_macro_f1 * 100:.2f}%"
)

print(
    f"Test Weighted F1    : "
    f"{test_weighted_f1 * 100:.2f}%"
)

print("\nFiles saved:")

print(
    "1.",
    report_path
)

print(
    "2.",
    cm_path
)

print(
    "3.",
    class_path
)

print(
    "4.",
    summary_path
)

print(
    "\nNo model files were modified."
)

print("=" * 70)