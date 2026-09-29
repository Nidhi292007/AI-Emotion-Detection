import os
import sys
import numpy as np
import pandas as pd
import joblib

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

from datasets import load_dataset
from scipy.sparse import hstack
from src.preprocessing import preprocess_text


# ============================================================
# LOAD CURRENT MODEL
# ============================================================

model = joblib.load("emotion_model.pkl")

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


# ============================================================
# LOAD TEST DATA
# ============================================================

dataset = load_dataset("dair-ai/emotion")

test_texts = dataset["test"]["text"]
test_labels = np.array(
    dataset["test"]["label"]
)


# ============================================================
# PREPROCESS
# ============================================================

processed_texts = [
    preprocess_text(text)
    for text in test_texts
]


# ============================================================
# CREATE FEATURES
# ============================================================

X_word = word_vectorizer.transform(
    processed_texts
)

X_char = char_vectorizer.transform(
    processed_texts
)

X_raw_char = raw_char_vectorizer.transform(
    test_texts
)


# ============================================================
# COMBINE FEATURES
# ============================================================

X_test = hstack([
    X_word,
    X_char * char_weight,
    X_raw_char * raw_char_weight
])


# ============================================================
# PREDICTIONS + PROBABILITIES
# ============================================================

predictions = model.predict(X_test)

probabilities = model.predict_proba(X_test)


# ============================================================
# BASELINE
# ============================================================

baseline_accuracy = np.mean(
    predictions == test_labels
)

print("\n==============================================")
print("EXPERIMENT 20")
print("TARGETED CLASS-PAIR BOUNDARY ANALYSIS")
print("==============================================")

print(
    f"\nCurrent baseline accuracy : "
    f"{baseline_accuracy * 100:.2f}%"
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

confusion = np.zeros(
    (6, 6),
    dtype=int
)

for actual, predicted in zip(
    test_labels,
    predictions
):
    confusion[actual][predicted] += 1


print("\n==============================================")
print("CONFUSION MATRIX")
print("==============================================")

print(
    f"{'Actual':<12}"
    + "".join(
        f"{name:<12}"
        for name in label_names
    )
)

for i in range(6):

    print(
        f"{label_names[i]:<12}"
        + "".join(
            f"{confusion[i][j]:<12}"
            for j in range(6)
        )
    )


# ============================================================
# TARGET PAIRS
# ============================================================

pairs = [
    (2, 1),  # Love -> Joy
    (1, 2),  # Joy -> Love

    (4, 5),  # Fear -> Surprise
    (5, 4),  # Surprise -> Fear

    (3, 0),  # Anger -> Sadness
    (0, 3),  # Sadness -> Anger

    (0, 4),  # Sadness -> Fear
    (4, 0)   # Fear -> Sadness
]


# ============================================================
# ANALYZE EACH PAIR
# ============================================================

print("\n==============================================")
print("TARGETED CLASS-PAIR ANALYSIS")
print("==============================================")


for actual_class, predicted_class in pairs:

    mask = (
        (test_labels == actual_class)
        &
        (predictions == predicted_class)
    )

    count = np.sum(mask)

    print("\n----------------------------------------------")

    print(
        f"Actual      : "
        f"{label_names[actual_class]}"
    )

    print(
        f"Predicted   : "
        f"{label_names[predicted_class]}"
    )

    print(
        f"Error count : {count}"
    )

    if count == 0:
        continue

    indices = np.where(mask)[0]

    print("\nExamples:")

    for index in indices[:5]:

        probs = probabilities[index]

        top_indices = np.argsort(
            probs
        )[-3:][::-1]

        print("\nText:")
        print(test_texts[index])

        print(
            f"Actual    : "
            f"{label_names[test_labels[index]]}"
        )

        print(
            f"Predicted : "
            f"{label_names[predictions[index]]}"
        )

        print("Top probabilities:")

        for class_index in top_indices:

            print(
                f"  {label_names[class_index]:<10}"
                f"{probs[class_index] * 100:.2f}%"
            )


# ============================================================
# PAIRWISE PROBABILITY ANALYSIS
# ============================================================

print("\n==============================================")
print("PAIRWISE PROBABILITY ANALYSIS")
print("==============================================")


pair_results = []


for class_a, class_b in [
    (2, 1),  # Love/Joy
    (4, 5),  # Fear/Surprise
    (3, 0),  # Anger/Sadness
    (0, 4)   # Sadness/Fear
]:

    mask = (
        ((test_labels == class_a) &
         (predictions == class_b))
        |
        ((test_labels == class_b) &
         (predictions == class_a))
    )

    indices = np.where(mask)[0]

    if len(indices) == 0:
        continue

    print("\n----------------------------------------------")

    print(
        f"{label_names[class_a]} "
        f"<-> "
        f"{label_names[class_b]}"
    )

    print(
        f"Total pair errors : "
        f"{len(indices)}"
    )

    for index in indices:

        prob_a = probabilities[
            index,
            class_a
        ]

        prob_b = probabilities[
            index,
            class_b
        ]

        pair_total = prob_a + prob_b

        if pair_total == 0:
            continue

        pair_a_ratio = (
            prob_a / pair_total
        )

        pair_b_ratio = (
            prob_b / pair_total
        )

        pair_results.append({

            "actual":
                label_names[
                    test_labels[index]
                ],

            "predicted":
                label_names[
                    predictions[index]
                ],

            "class_a":
                label_names[class_a],

            "class_b":
                label_names[class_b],

            "prob_a":
                prob_a,

            "prob_b":
                prob_b,

            "pair_a_ratio":
                pair_a_ratio,

            "pair_b_ratio":
                pair_b_ratio,

            "text":
                test_texts[index]
        })


# ============================================================
# SUMMARY
# ============================================================

print("\n==============================================")
print("EXPERIMENT 20 SUMMARY")
print("==============================================")

print(
    "\nLargest targeted error pairs:"
)

for actual_class, predicted_class in pairs:

    count = confusion[
        actual_class,
        predicted_class
    ]

    if count > 0:

        print(
            f"{label_names[actual_class]}"
            f" -> "
            f"{label_names[predicted_class]}"
            f" : {count}"
        )


# ============================================================
# SAVE ANALYSIS
# ============================================================

if pair_results:

    df = pd.DataFrame(
        pair_results
    )

    df.to_csv(
        "experiment20_pair_analysis.csv",
        index=False
    )

print("\n==============================================")
print("Experiment 20 completed.")
print("No model files were modified.")
print("==============================================")