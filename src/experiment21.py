import os
import sys
import numpy as np
import joblib

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

from datasets import load_dataset
from scipy.sparse import hstack
from sklearn.naive_bayes import ComplementNB
from sklearn.model_selection import StratifiedKFold


# ============================================================
# LOAD CURRENT MODEL + VECTORIZERS
# ============================================================

main_model = joblib.load("emotion_model.pkl")

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


# ============================================================
# LABELS
# ============================================================

label_names = [
    "Sadness",
    "Joy",
    "Love",
    "Anger",
    "Fear",
    "Surprise"
]

JOY = 1
LOVE = 2


# ============================================================
# LOAD DATA
# ============================================================

dataset = load_dataset("dair-ai/emotion")

train_texts = dataset["train"]["text"]
train_labels = np.array(
    dataset["train"]["label"]
)

test_texts = dataset["test"]["text"]
test_labels = np.array(
    dataset["test"]["label"]
)


# ============================================================
# PREPROCESS
# ============================================================

from src.preprocessing import preprocess_text

processed_train = [
    preprocess_text(text)
    for text in train_texts
]

processed_test = [
    preprocess_text(text)
    for text in test_texts
]


# ============================================================
# CREATE FEATURES
# ============================================================

print("\nCreating training features...")

X_train_word = word_vectorizer.transform(
    processed_train
)

X_train_char = char_vectorizer.transform(
    processed_train
)

X_train_raw_char = raw_char_vectorizer.transform(
    train_texts
)

X_train = hstack([
    X_train_word,
    X_train_char * char_weight,
    X_train_raw_char * raw_char_weight
])


print("Creating test features...")

X_test_word = word_vectorizer.transform(
    processed_test
)

X_test_char = char_vectorizer.transform(
    processed_test
)

X_test_raw_char = raw_char_vectorizer.transform(
    test_texts
)

X_test = hstack([
    X_test_word,
    X_test_char * char_weight,
    X_test_raw_char * raw_char_weight
])


# ============================================================
# CURRENT BASELINE
# ============================================================

base_predictions = main_model.predict(
    X_test
)

baseline_accuracy = np.mean(
    base_predictions == test_labels
)

print("\n==============================================")
print("EXPERIMENT 21")
print("LOVE <-> JOY PAIRWISE NAIVE BAYES")
print("==============================================")

print(
    f"\nCurrent baseline accuracy : "
    f"{baseline_accuracy * 100:.2f}%"
)


# ============================================================
# SELECT ONLY JOY + LOVE TRAINING DATA
# ============================================================

pair_train_mask = (
    (train_labels == JOY)
    |
    (train_labels == LOVE)
)

X_pair = X_train[pair_train_mask]

y_pair_original = train_labels[pair_train_mask]


# Convert:
# Joy  -> 0
# Love -> 1

y_pair = np.where(
    y_pair_original == LOVE,
    1,
    0
)


print(
    f"\nLove/Joy training samples : "
    f"{len(y_pair)}"
)


# ============================================================
# 5-FOLD CV FOR PAIRWISE MODEL
# ============================================================

print("\n==============================================")
print("5-FOLD CROSS-VALIDATION")
print("==============================================")


skf = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)


thresholds = [
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


threshold_scores = {
    threshold: []
    for threshold in thresholds
}


for fold, (train_idx, val_idx) in enumerate(
    skf.split(X_pair, y_pair),
    start=1
):

    print(f"Processing fold {fold}/5...")

    X_fold_train = X_pair[train_idx]
    y_fold_train = y_pair[train_idx]

    X_fold_val = X_pair[val_idx]
    y_fold_val = y_pair[val_idx]

    pair_model = ComplementNB(
        alpha=0.35
    )

    pair_model.fit(
        X_fold_train,
        y_fold_train
    )

    val_probabilities = pair_model.predict_proba(
        X_fold_val
    )[:, 1]

    for threshold in thresholds:

        val_predictions = (
            val_probabilities >= threshold
        ).astype(int)

        accuracy = np.mean(
            val_predictions == y_fold_val
        )

        threshold_scores[
            threshold
        ].append(accuracy)


# ============================================================
# SELECT BEST THRESHOLD USING CV ONLY
# ============================================================

print("\n==============================================")
print("CV THRESHOLD RESULTS")
print("==============================================")


best_threshold = None
best_cv_accuracy = -1


for threshold in thresholds:

    mean_accuracy = np.mean(
        threshold_scores[threshold]
    )

    print(
        f"Threshold {threshold:.2f} "
        f"-> CV accuracy "
        f"{mean_accuracy * 100:.2f}%"
    )

    if mean_accuracy > best_cv_accuracy:

        best_cv_accuracy = mean_accuracy
        best_threshold = threshold


print(
    f"\nBest threshold : "
    f"{best_threshold:.2f}"
)

print(
    f"Best pairwise CV accuracy : "
    f"{best_cv_accuracy * 100:.2f}%"
)


# ============================================================
# TRAIN FINAL PAIRWISE MODEL
# ============================================================

print("\nTraining final Love/Joy pairwise model...")


pair_model = ComplementNB(
    alpha=0.35
)

pair_model.fit(
    X_pair,
    y_pair
)


# ============================================================
# APPLY PAIRWISE MODEL TO TEST SET
# ============================================================

final_predictions = base_predictions.copy()


# Only correct predictions currently classified
# as Joy or Love.

candidate_mask = (
    (base_predictions == JOY)
    |
    (base_predictions == LOVE)
)

candidate_indices = np.where(
    candidate_mask
)[0]


pair_probabilities = pair_model.predict_proba(
    X_test[candidate_indices]
)[:, 1]


# ============================================================
# PAIRWISE CORRECTION
# ============================================================

for position, test_index in enumerate(
    candidate_indices
):

    love_probability = (
        pair_probabilities[position]
    )

    if love_probability >= best_threshold:

        final_predictions[test_index] = LOVE

    else:

        final_predictions[test_index] = JOY


# ============================================================
# FINAL ACCURACY
# ============================================================

final_accuracy = np.mean(
    final_predictions == test_labels
)


print("\n==============================================")
print("FINAL RESULT")
print("==============================================")


print(
    f"\nBaseline accuracy : "
    f"{baseline_accuracy * 100:.2f}%"
)

print(
    f"Pairwise accuracy : "
    f"{final_accuracy * 100:.2f}%"
)

print(
    f"Difference        : "
    f"{(final_accuracy - baseline_accuracy) * 100:+.2f} percentage points"
)


# ============================================================
# LOVE / JOY CONFUSION
# ============================================================

print("\n==============================================")
print("LOVE / JOY COMPARISON")
print("==============================================")


baseline_love_joy_errors = np.sum(
    (
        ((test_labels == LOVE) &
         (base_predictions == JOY))
        |
        ((test_labels == JOY) &
         (base_predictions == LOVE))
    )
)


final_love_joy_errors = np.sum(
    (
        ((test_labels == LOVE) &
         (final_predictions == JOY))
        |
        ((test_labels == JOY) &
         (final_predictions == LOVE))
    )
)


print(
    f"Baseline Love/Joy errors : "
    f"{baseline_love_joy_errors}"
)

print(
    f"Final Love/Joy errors    : "
    f"{final_love_joy_errors}"
)

print(
    f"Errors corrected         : "
    f"{baseline_love_joy_errors - final_love_joy_errors}"
)


# ============================================================
# TOTAL CHANGED PREDICTIONS
# ============================================================

changed = np.sum(
    final_predictions != base_predictions
)

print(
    f"\nTotal predictions changed : "
    f"{changed}"
)


# ============================================================
# SAFETY CHECK
# ============================================================

print("\n==============================================")
print("MODEL SAFETY CHECK")
print("==============================================")

print(
    "emotion_model.pkl was NOT modified."
)

print(
    "All existing vectorizers were NOT modified."
)

print(
    "Experiment 21 does NOT save or overwrite the model."
)

print("\n==============================================")
print("EXPERIMENT 21 COMPLETED")
print("==============================================")