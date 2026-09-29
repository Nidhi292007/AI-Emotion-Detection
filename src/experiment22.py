import os
import sys
import numpy as np
import joblib

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

from datasets import load_dataset
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import ComplementNB
from sklearn.model_selection import StratifiedKFold


# ============================================================
# LOAD CURRENT VECTORIZERS + MODEL
# ============================================================

main_model = joblib.load(
    "emotion_model.pkl"
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


# ============================================================
# LOAD DATASET
# ============================================================

dataset = load_dataset(
    "dair-ai/emotion"
)

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

print("\nPreprocessing training data...")

processed_train = [
    preprocess_text(text)
    for text in train_texts
]

print("Preprocessing test data...")

processed_test = [
    preprocess_text(text)
    for text in test_texts
]


# ============================================================
# EXISTING FEATURES
# ============================================================

print("\nCreating existing word features...")

X_train_word = word_vectorizer.transform(
    processed_train
)

X_test_word = word_vectorizer.transform(
    processed_test
)


print("Creating existing processed-char features...")

X_train_char = char_vectorizer.transform(
    processed_train
)

X_test_char = char_vectorizer.transform(
    processed_test
)


print("Creating existing raw-char features...")

X_train_raw_char = raw_char_vectorizer.transform(
    train_texts
)

X_test_raw_char = raw_char_vectorizer.transform(
    test_texts
)


# ============================================================
# EXISTING FEATURE BASE
# ============================================================

X_train_base = hstack([
    X_train_word,
    X_train_char * char_weight,
    X_train_raw_char * raw_char_weight
])

X_test_base = hstack([
    X_test_word,
    X_test_char * char_weight,
    X_test_raw_char * raw_char_weight
])


# ============================================================
# NEW RAW-WORD TF-IDF
# ============================================================

print("\nCreating NEW raw-word TF-IDF features...")

raw_word_vectorizer = TfidfVectorizer(
    ngram_range=(1, 2),
    max_features=30000,
    min_df=2,
    max_df=0.95,
    sublinear_tf=True
)

X_train_raw_word = raw_word_vectorizer.fit_transform(
    train_texts
)

X_test_raw_word = raw_word_vectorizer.transform(
    test_texts
)


print(
    f"Raw-word features : "
    f"{X_train_raw_word.shape[1]}"
)


# ============================================================
# CURRENT BASELINE
# ============================================================

base_predictions = main_model.predict(
    X_test_base
)

baseline_accuracy = np.mean(
    base_predictions == test_labels
)


print("\n==============================================")
print("EXPERIMENT 22")
print("RAW-WORD TF-IDF FEATURE FUSION")
print("==============================================")

print(
    f"\nCurrent baseline accuracy : "
    f"{baseline_accuracy * 100:.2f}%"
)


# ============================================================
# CROSS VALIDATION
# ============================================================

print("\n==============================================")
print("5-FOLD CROSS-VALIDATION")
print("==============================================")


skf = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)


# Raw-word weights to test
raw_word_weights = [
    0.10,
    0.20,
    0.30,
    0.40,
    0.50,
    0.60,
    0.70,
    0.80,
    1.00
]


# ComplementNB alpha values
alphas = [
    0.25,
    0.30,
    0.35,
    0.40,
    0.45,
    0.50
]


best_cv_accuracy = -1
best_weight = None
best_alpha = None


for weight in raw_word_weights:

    print(
        f"\nTesting raw-word weight = "
        f"{weight:.2f}"
    )

    X_train_combined = hstack([
        X_train_base,
        X_train_raw_word * weight
    ])

    fold_scores = []

    for fold, (train_idx, val_idx) in enumerate(
        skf.split(
            X_train_combined,
            train_labels
        ),
        start=1
    ):

        X_fold_train = X_train_combined[
            train_idx
        ]

        X_fold_val = X_train_combined[
            val_idx
        ]

        y_fold_train = train_labels[
            train_idx
        ]

        y_fold_val = train_labels[
            val_idx
        ]

        best_fold_score = -1

        for alpha in alphas:

            model = ComplementNB(
                alpha=alpha
            )

            model.fit(
                X_fold_train,
                y_fold_train
            )

            predictions = model.predict(
                X_fold_val
            )

            accuracy = np.mean(
                predictions == y_fold_val
            )

            if accuracy > best_fold_score:
                best_fold_score = accuracy

        fold_scores.append(
            best_fold_score
        )

        print(
            f"  Fold {fold}: "
            f"{best_fold_score * 100:.2f}%"
        )

    mean_cv = np.mean(
        fold_scores
    )

    print(
        f"  Mean CV: "
        f"{mean_cv * 100:.2f}%"
    )

    if mean_cv > best_cv_accuracy:

        best_cv_accuracy = mean_cv
        best_weight = weight


# ============================================================
# IMPORTANT:
# REFINE ALPHA FOR BEST WEIGHT
# ============================================================

print("\n==============================================")
print("ALPHA REFINEMENT")
print("==============================================")


X_train_best_weight = hstack([
    X_train_base,
    X_train_raw_word * best_weight
])


alpha_scores = {}


for alpha in alphas:

    fold_scores = []

    for train_idx, val_idx in skf.split(
        X_train_best_weight,
        train_labels
    ):

        model = ComplementNB(
            alpha=alpha
        )

        model.fit(
            X_train_best_weight[train_idx],
            train_labels[train_idx]
        )

        predictions = model.predict(
            X_train_best_weight[val_idx]
        )

        accuracy = np.mean(
            predictions ==
            train_labels[val_idx]
        )

        fold_scores.append(
            accuracy
        )

    mean_cv = np.mean(
        fold_scores
    )

    alpha_scores[alpha] = mean_cv

    print(
        f"Alpha {alpha:.2f} -> "
        f"CV {mean_cv * 100:.2f}%"
    )


best_alpha = max(
    alpha_scores,
    key=alpha_scores.get
)

best_cv_accuracy = alpha_scores[
    best_alpha
]


# ============================================================
# CV RESULT
# ============================================================

print("\n==============================================")
print("BEST CV CONFIGURATION")
print("==============================================")

print(
    f"Raw-word weight : "
    f"{best_weight:.2f}"
)

print(
    f"Alpha           : "
    f"{best_alpha:.2f}"
)

print(
    f"CV accuracy     : "
    f"{best_cv_accuracy * 100:.2f}%"
)


# ============================================================
# TRAIN FINAL EXPERIMENTAL MODEL
# ============================================================

print(
    "\nTraining final experimental model..."
)


experimental_model = ComplementNB(
    alpha=best_alpha
)

experimental_model.fit(
    X_train_best_weight,
    train_labels
)


# ============================================================
# TEST
# ============================================================

X_test_combined = hstack([
    X_test_base,
    X_test_raw_word * best_weight
])


experimental_predictions = (
    experimental_model.predict(
        X_test_combined
    )
)


experimental_accuracy = np.mean(
    experimental_predictions ==
    test_labels
)


# ============================================================
# FINAL RESULT
# ============================================================

print("\n==============================================")
print("FINAL RESULT")
print("==============================================")


print(
    f"\nBaseline accuracy    : "
    f"{baseline_accuracy * 100:.2f}%"
)

print(
    f"Experiment 22        : "
    f"{experimental_accuracy * 100:.2f}%"
)

print(
    f"Difference           : "
    f"{(experimental_accuracy - baseline_accuracy) * 100:+.2f} percentage points"
)


# ============================================================
# LOVE / JOY ANALYSIS
# ============================================================

baseline_love_joy = np.sum(
    (
        ((test_labels == 2) &
         (base_predictions == 1))
        |
        ((test_labels == 1) &
         (base_predictions == 2))
    )
)


experimental_love_joy = np.sum(
    (
        ((test_labels == 2) &
         (experimental_predictions == 1))
        |
        ((test_labels == 1) &
         (experimental_predictions == 2))
    )
)


print("\n==============================================")
print("LOVE / JOY ANALYSIS")
print("==============================================")


print(
    f"Baseline Love/Joy errors : "
    f"{baseline_love_joy}"
)

print(
    f"Experiment 22 errors     : "
    f"{experimental_love_joy}"
)

print(
    f"Change                   : "
    f"{experimental_love_joy - baseline_love_joy:+d}"
)


# ============================================================
# SAVE ONLY EXPERIMENTAL FILES
# ============================================================

joblib.dump(
    raw_word_vectorizer,
    "experiment22_raw_word_vectorizer.pkl"
)

joblib.dump(
    experimental_model,
    "experiment22_emotion_model.pkl"
)

joblib.dump(
    best_weight,
    "experiment22_raw_word_weight.pkl"
)

print("\n==============================================")
print("EXPERIMENT 22 COMPLETED")
print("==============================================")

print(
    "\nOriginal emotion_model.pkl was NOT modified."
)

print(
    "Original vectorizers were NOT modified."
)

print(
    "Experimental files were saved separately."
)

print("==============================================")