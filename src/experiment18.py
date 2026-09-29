import joblib
import numpy as np

from datasets import load_dataset
from sklearn.metrics import accuracy_score
from scipy.sparse import hstack

from preprocessing import preprocess_text


# ============================================================
# LOAD DATA
# ============================================================

print("Loading dataset...")

dataset = load_dataset("dair-ai/emotion")

train_texts = dataset["train"]["text"]
test_texts = dataset["test"]["text"]

y_train = np.array(dataset["train"]["label"])
y_test = np.array(dataset["test"]["label"])

print("Training samples:", len(train_texts))
print("Test samples:", len(test_texts))


# ============================================================
# PREPROCESS
# ============================================================

print("\nPreprocessing...")

train_processed = [
    preprocess_text(text)
    for text in train_texts
]

test_processed = [
    preprocess_text(text)
    for text in test_texts
]


# ============================================================
# LOAD MODEL + VECTORIZERS
# ============================================================

print("\nLoading saved model...")

model = joblib.load("emotion_model.pkl")

word_vectorizer = joblib.load(
    "word_tfidf_vectorizer.pkl"
)

char_vectorizer = joblib.load(
    "char_tfidf_vectorizer.pkl"
)

raw_char_vectorizer = joblib.load(
    "raw_char_tfidf_vectorizer.pkl"
)

char_weight = joblib.load(
    "char_weight.pkl"
)

raw_char_weight = joblib.load(
    "raw_char_weight.pkl"
)


# ============================================================
# CREATE FEATURES
# ============================================================

print("\nCreating features...")

X_word_test = word_vectorizer.transform(
    test_processed
)

X_char_test = char_vectorizer.transform(
    test_processed
)

X_raw_test = raw_char_vectorizer.transform(
    test_texts
)

X_test = hstack([
    X_word_test,
    X_char_test * char_weight,
    X_raw_test * raw_char_weight
]).tocsr()


# ============================================================
# TRAIN MODEL
# ============================================================

print("\nTraining model...")

X_word_train = word_vectorizer.transform(
    train_processed
)

X_char_train = char_vectorizer.transform(
    train_processed
)

X_raw_train = raw_char_vectorizer.transform(
    train_texts
)

X_train = hstack([
    X_word_train,
    X_char_train * char_weight,
    X_raw_train * raw_char_weight
]).tocsr()

model.fit(
    X_train,
    y_train
)


# ============================================================
# PROBABILITIES
# ============================================================

print("\nGenerating probabilities...")

probabilities = model.predict_proba(
    X_test
)

predictions = np.argmax(
    probabilities,
    axis=1
)

accuracy = accuracy_score(
    y_test,
    predictions
)

print("\n" + "=" * 70)
print("BASE MODEL")
print("=" * 70)

print(
    f"Accuracy: {accuracy * 100:.2f}%"
)


# ============================================================
# LABELS
# ============================================================

labels = [
    "Sadness",
    "Joy",
    "Love",
    "Anger",
    "Fear",
    "Surprise"
]


# ============================================================
# CONFIDENCE ANALYSIS
# ============================================================

print("\n" + "=" * 70)
print("CONFIDENCE ANALYSIS")
print("=" * 70)


top1 = np.max(
    probabilities,
    axis=1
)

sorted_probabilities = np.sort(
    probabilities,
    axis=1
)

top2 = sorted_probabilities[:, -2]

margin = top1 - top2


print(
    f"Average top-1 confidence: "
    f"{np.mean(top1) * 100:.2f}%"
)

print(
    f"Average top-2 confidence: "
    f"{np.mean(top2) * 100:.2f}%"
)

print(
    f"Average confidence margin: "
    f"{np.mean(margin) * 100:.2f}%"
)


# ============================================================
# CORRECT VS INCORRECT CONFIDENCE
# ============================================================

correct = predictions == y_test

print("\n" + "=" * 70)
print("CORRECT VS INCORRECT")
print("=" * 70)

print(
    f"Correct predictions: "
    f"{np.sum(correct)}"
)

print(
    f"Incorrect predictions: "
    f"{np.sum(~correct)}"
)

print(
    f"Average confidence - correct: "
    f"{np.mean(top1[correct]) * 100:.2f}%"
)

print(
    f"Average confidence - incorrect: "
    f"{np.mean(top1[~correct]) * 100:.2f}%"
)

print(
    f"Average margin - correct: "
    f"{np.mean(margin[correct]) * 100:.2f}%"
)

print(
    f"Average margin - incorrect: "
    f"{np.mean(margin[~correct]) * 100:.2f}%"
)


# ============================================================
# LOW CONFIDENCE ERRORS
# ============================================================

print("\n" + "=" * 70)
print("LOW-CONFIDENCE ERRORS")
print("=" * 70)

error_indices = np.where(~correct)[0]

error_indices = sorted(
    error_indices,
    key=lambda i: margin[i]
)

for i in error_indices[:20]:

    actual = labels[y_test[i]]
    predicted = labels[predictions[i]]

    print("\nText:")
    print(test_texts[i])

    print(
        f"Actual: {actual}"
    )

    print(
        f"Predicted: {predicted}"
    )

    print(
        f"Top confidence: "
        f"{top1[i] * 100:.2f}%"
    )

    print(
        f"Second confidence: "
        f"{top2[i] * 100:.2f}%"
    )

    print(
        f"Margin: "
        f"{margin[i] * 100:.2f}%"
    )


# ============================================================
# LOVE ↔ JOY ANALYSIS
# ============================================================

print("\n" + "=" * 70)
print("LOVE ↔ JOY ANALYSIS")
print("=" * 70)

love_joy_indices = []

for i in range(len(y_test)):

    actual = y_test[i]
    predicted = predictions[i]

    if (
        (actual == 1 and predicted == 2)
        or
        (actual == 2 and predicted == 1)
    ):
        love_joy_indices.append(i)


print(
    "Love/Joy confusion count:",
    len(love_joy_indices)
)


for i in love_joy_indices[:20]:

    print("\nText:")
    print(test_texts[i])

    print(
        "Actual:",
        labels[y_test[i]]
    )

    print(
        "Predicted:",
        labels[predictions[i]]
    )

    print(
        "Joy probability:",
        f"{probabilities[i][1] * 100:.2f}%"
    )

    print(
        "Love probability:",
        f"{probabilities[i][2] * 100:.2f}%"
    )


# ============================================================
# FEAR ↔ SURPRISE ANALYSIS
# ============================================================

print("\n" + "=" * 70)
print("FEAR ↔ SURPRISE ANALYSIS")
print("=" * 70)

fear_surprise_indices = []

for i in range(len(y_test)):

    actual = y_test[i]
    predicted = predictions[i]

    if (
        (actual == 4 and predicted == 5)
        or
        (actual == 5 and predicted == 4)
    ):
        fear_surprise_indices.append(i)


print(
    "Fear/Surprise confusion count:",
    len(fear_surprise_indices)
)


for i in fear_surprise_indices[:20]:

    print("\nText:")
    print(test_texts[i])

    print(
        "Actual:",
        labels[y_test[i]]
    )

    print(
        "Predicted:",
        labels[predictions[i]]
    )

    print(
        "Fear probability:",
        f"{probabilities[i][4] * 100:.2f}%"
    )

    print(
        "Surprise probability:",
        f"{probabilities[i][5] * 100:.2f}%"
    )


print("\n" + "=" * 70)
print("EXPERIMENT 18 COMPLETE")
print("=" * 70)

print("No model files were modified.")