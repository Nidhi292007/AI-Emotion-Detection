import os
import sys
import numpy as np
import joblib

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

from datasets import load_dataset
from scipy.sparse import hstack
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import ComplementNB
from sklearn.metrics import accuracy_score


# ============================================================
# EXPERIMENT 23
# ERROR-DRIVEN CLASS WEIGHTING
# ============================================================

print("=" * 60)
print("EXPERIMENT 23")
print("ERROR-DRIVEN CLASS WEIGHTING")
print("=" * 60)


# ============================================================
# LABELS
# ============================================================

labels = [
    "sadness",
    "joy",
    "love",
    "anger",
    "fear",
    "surprise"
]


# ============================================================
# LOAD EXPERIMENT 22 COMPONENTS
# ============================================================

print("\nLoading Experiment 22 components...")


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


print(f"Processed-char weight : {char_weight}")
print(f"Raw-char weight       : {raw_char_weight}")
print("Raw-word weight       : 0.60")


# ============================================================
# LOAD DATASET
# ============================================================

print("\nLoading dataset...")

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
# CREATE EXISTING FEATURES
# ============================================================

print("\nCreating processed-word features...")

X_train_word = word_vectorizer.transform(
    processed_train
)

X_test_word = word_vectorizer.transform(
    processed_test
)


print("Creating processed-char features...")

X_train_char = char_vectorizer.transform(
    processed_train
)

X_test_char = char_vectorizer.transform(
    processed_test
)


print("Creating raw-char features...")

X_train_raw_char = raw_char_vectorizer.transform(
    train_texts
)

X_test_raw_char = raw_char_vectorizer.transform(
    test_texts
)


# ============================================================
# CREATE RAW-WORD FEATURES
# ============================================================

print("Creating raw-word features...")

X_train_raw_word = raw_word_vectorizer.transform(
    train_texts
)

X_test_raw_word = raw_word_vectorizer.transform(
    test_texts
)


# ============================================================
# EXPERIMENT 22 ARCHITECTURE
# ============================================================

RAW_WORD_WEIGHT = 0.60

print("\nBuilding Experiment 22 feature architecture...")


X_train = hstack([
    X_train_word,
    X_train_char * char_weight,
    X_train_raw_char * raw_char_weight,
    X_train_raw_word * RAW_WORD_WEIGHT
]).tocsr()


X_test = hstack([
    X_test_word,
    X_test_char * char_weight,
    X_test_raw_char * raw_char_weight,
    X_test_raw_word * RAW_WORD_WEIGHT
]).tocsr()


print(
    f"Total features : {X_train.shape[1]}"
)


# ============================================================
# EXPERIMENT 22 BASELINE
# ============================================================

BASE_ALPHA = 0.30

baseline_model = ComplementNB(
    alpha=BASE_ALPHA,
    norm=False
)

baseline_model.fit(
    X_train,
    train_labels
)

baseline_predictions = baseline_model.predict(
    X_test
)

baseline_accuracy = accuracy_score(
    test_labels,
    baseline_predictions
)


print("\n" + "=" * 60)
print("EXPERIMENT 22 BASELINE")
print("=" * 60)

print(
    f"Baseline test accuracy : "
    f"{baseline_accuracy * 100:.2f}%"
)


# ============================================================
# CLASS WEIGHT CONFIGURATIONS
# ============================================================

configs = {

    "baseline": {
        0: 1.00,
        1: 1.00,
        2: 1.00,
        3: 1.00,
        4: 1.00,
        5: 1.00
    },

    "love_1.05": {
        0: 1.00,
        1: 1.00,
        2: 1.05,
        3: 1.00,
        4: 1.00,
        5: 1.00
    },

    "love_1.10": {
        0: 1.00,
        1: 1.00,
        2: 1.10,
        3: 1.00,
        4: 1.00,
        5: 1.00
    },

    "anger_1.05": {
        0: 1.00,
        1: 1.00,
        2: 1.00,
        3: 1.05,
        4: 1.00,
        5: 1.00
    },

    "anger_1.10": {
        0: 1.00,
        1: 1.00,
        2: 1.00,
        3: 1.10,
        4: 1.00,
        5: 1.00
    },

    "sadness_1.05": {
        0: 1.05,
        1: 1.00,
        2: 1.00,
        3: 1.00,
        4: 1.00,
        5: 1.00
    },

    "fear_1.05": {
        0: 1.00,
        1: 1.00,
        2: 1.00,
        3: 1.00,
        4: 1.05,
        5: 1.00
    },

    "surprise_1.05": {
        0: 1.00,
        1: 1.00,
        2: 1.00,
        3: 1.00,
        4: 1.00,
        5: 1.05
    },

    "love_anger": {
        0: 1.00,
        1: 1.00,
        2: 1.08,
        3: 1.08,
        4: 1.00,
        5: 1.00
    },

    "anger_sadness": {
        0: 1.08,
        1: 1.00,
        2: 1.00,
        3: 1.08,
        4: 1.00,
        5: 1.00
    },

    "fear_surprise": {
        0: 1.00,
        1: 1.00,
        2: 1.00,
        3: 1.00,
        4: 1.08,
        5: 1.08
    },

    "targeted_all": {
        0: 1.05,
        1: 1.00,
        2: 1.08,
        3: 1.08,
        4: 1.05,
        5: 1.05
    }
}


# ============================================================
# 5-FOLD CROSS-VALIDATION
# ============================================================

print("\n" + "=" * 60)
print("5-FOLD CROSS-VALIDATION")
print("=" * 60)


skf = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)


best_cv_accuracy = -1
best_name = None
best_weights = None


for name, class_weights in configs.items():

    print(
        f"\nTesting configuration: "
        f"{name}"
    )

    fold_scores = []

    for fold, (train_idx, val_idx) in enumerate(
        skf.split(X_train, train_labels),
        start=1
    ):

        X_fold_train = X_train[
            train_idx
        ]

        X_fold_val = X_train[
            val_idx
        ]

        y_fold_train = train_labels[
            train_idx
        ]

        y_fold_val = train_labels[
            val_idx
        ]

        # Create sample weights
        sample_weights = np.array([
            class_weights[int(y)]
            for y in y_fold_train
        ])

        model = ComplementNB(
            alpha=BASE_ALPHA,
            norm=False
        )

        model.fit(
            X_fold_train,
            y_fold_train,
            sample_weight=sample_weights
        )

        predictions = model.predict(
            X_fold_val
        )

        accuracy = accuracy_score(
            y_fold_val,
            predictions
        )

        fold_scores.append(
            accuracy
        )

        print(
            f"  Fold {fold}: "
            f"{accuracy * 100:.2f}%"
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
        best_name = name
        best_weights = class_weights


# ============================================================
# BEST CONFIGURATION
# ============================================================

print("\n" + "=" * 60)
print("BEST CV CONFIGURATION")
print("=" * 60)

print(
    f"Configuration : "
    f"{best_name}"
)

print(
    f"CV accuracy   : "
    f"{best_cv_accuracy * 100:.2f}%"
)

print("\nClass weights:")

for i in range(6):

    print(
        f"  {labels[i]:9s} : "
        f"{best_weights[i]:.2f}"
    )


# ============================================================
# TRAIN FINAL MODEL
# ============================================================

print(
    "\nTraining final Experiment 23 model..."
)

final_sample_weights = np.array([
    best_weights[int(y)]
    for y in train_labels
])


final_model = ComplementNB(
    alpha=BASE_ALPHA,
    norm=False
)

final_model.fit(
    X_train,
    train_labels,
    sample_weight=final_sample_weights
)


# ============================================================
# TEST
# ============================================================

final_predictions = final_model.predict(
    X_test
)

final_accuracy = accuracy_score(
    test_labels,
    final_predictions
)


# ============================================================
# FINAL RESULT
# ============================================================

print("\n" + "=" * 60)
print("FINAL RESULT")
print("=" * 60)

print(
    f"Experiment 22 baseline : "
    f"{baseline_accuracy * 100:.2f}%"
)

print(
    f"Experiment 23          : "
    f"{final_accuracy * 100:.2f}%"
)

difference = (
    final_accuracy -
    baseline_accuracy
) * 100

print(
    f"Difference              : "
    f"{difference:+.2f} percentage points"
)


# ============================================================
# LOVE / JOY ANALYSIS
# ============================================================

baseline_love_joy = np.sum(
    (
        ((test_labels == 2) &
         (baseline_predictions == 1))
        |
        ((test_labels == 1) &
         (baseline_predictions == 2))
    )
)


experiment_love_joy = np.sum(
    (
        ((test_labels == 2) &
         (final_predictions == 1))
        |
        ((test_labels == 1) &
         (final_predictions == 2))
    )
)


print("\n" + "=" * 60)
print("LOVE / JOY ANALYSIS")
print("=" * 60)

print(
    f"Baseline Love/Joy errors : "
    f"{baseline_love_joy}"
)

print(
    f"Experiment 23 errors     : "
    f"{experiment_love_joy}"
)

print(
    f"Change                   : "
    f"{experiment_love_joy - baseline_love_joy:+d}"
)


# ============================================================
# SAVE ONLY IF CV IMPROVES
# ============================================================

# Experiment 22 CV was 89.73%
experiment22_cv = 0.8973

if best_cv_accuracy > experiment22_cv:

    joblib.dump(
        final_model,
        "experiment23_emotion_model.pkl"
    )

    joblib.dump(
        best_weights,
        "experiment23_class_weights.pkl"
    )

    print(
        "\nExperiment 23 model saved:"
    )

    print(
        "  experiment23_emotion_model.pkl"
    )

    print(
        "  experiment23_class_weights.pkl"
    )

else:

    print(
        "\nExperiment 23 did NOT improve "
        "Experiment 22 CV."
    )

    print(
        "No Experiment 23 model will be "
        "treated as the new champion."
    )


# ============================================================
# SAFETY CHECK
# ============================================================

print("\n" + "=" * 60)
print("EXPERIMENT 23 COMPLETED")
print("=" * 60)

print(
    "emotion_model.pkl was NOT modified."
)

print(
    "Experiment 22 files were NOT modified."
)

print("=" * 60)