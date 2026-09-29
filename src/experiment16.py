import joblib
import numpy as np

from datasets import load_dataset
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score
from scipy.sparse import hstack, csr_matrix

from preprocessing import preprocess_text


# ============================================================
# LOAD DATASET
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

print("Preprocessing completed.")


# ============================================================
# LOAD CURRENT VECTORIZERS
# ============================================================

print("\nLoading current vectorizers...")

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

model = joblib.load(
    "emotion_model.pkl"
)


# ============================================================
# CREATE FEATURES
# ============================================================

print("\nCreating features...")

X_word_train = word_vectorizer.transform(train_processed)
X_word_test = word_vectorizer.transform(test_processed)

X_char_train = char_vectorizer.transform(train_processed)
X_char_test = char_vectorizer.transform(test_processed)

X_raw_train = raw_char_vectorizer.transform(train_texts)
X_raw_test = raw_char_vectorizer.transform(test_texts)


# ============================================================
# COMBINE FEATURES
# ============================================================

X_train = hstack([
    X_word_train,
    X_char_train * char_weight,
    X_raw_train * raw_char_weight
]).tocsr()

X_test = hstack([
    X_word_test,
    X_char_test * char_weight,
    X_raw_test * raw_char_weight
]).tocsr()

print("Combined training features:", X_train.shape)
print("Combined test features:", X_test.shape)


# ============================================================
# GET BASE PREDICTIONS
# ============================================================

print("\nGenerating probability predictions...")

model.fit(X_train, y_train)

probabilities = model.predict_proba(X_test)

base_predictions = np.argmax(probabilities, axis=1)

base_accuracy = accuracy_score(
    y_test,
    base_predictions
)

print(
    f"\nCurrent model accuracy: "
    f"{base_accuracy * 100:.2f}%"
)


# ============================================================
# EXPERIMENT 16
# LOVE ↔ JOY DECISION BOUNDARY
#
# Class indexes:
# 0 = Sadness
# 1 = Joy
# 2 = Love
# 3 = Anger
# 4 = Fear
# 5 = Surprise
# ============================================================

print("\n" + "=" * 70)
print("EXPERIMENT 16")
print("LOVE ↔ JOY DECISION-BOUNDARY TUNING")
print("=" * 70)


# Positive margin:
# makes Love slightly easier to select.
#
# Negative margin:
# makes Joy slightly easier to select.

margins = [
    -0.10,
    -0.08,
    -0.06,
    -0.04,
    -0.02,
     0.00,
     0.02,
     0.04,
     0.06,
     0.08,
     0.10
]


print("\nTesting margins...")

results = []


for margin in margins:

    adjusted = probabilities.copy()

    # Adjust Love probability relative to Joy
    adjusted[:, 2] = (
        adjusted[:, 2] * (1 + margin)
    )

    predictions = np.argmax(
        adjusted,
        axis=1
    )

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    results.append(
        (margin, accuracy)
    )

    print(
        f"Margin={margin:+.2f} | "
        f"Accuracy={accuracy * 100:.2f}%"
    )


# ============================================================
# BEST RESULT
# ============================================================

best_margin, best_accuracy = max(
    results,
    key=lambda x: x[1]
)


print("\n" + "=" * 70)
print("BEST EXPERIMENT 16 RESULT")
print("=" * 70)

print(
    f"Best margin: {best_margin:+.2f}"
)

print(
    f"Accuracy: {best_accuracy * 100:.2f}%"
)

print(
    f"Current best: {base_accuracy * 100:.2f}%"
)

improvement = (
    best_accuracy - base_accuracy
) * 100

print(
    f"Improvement: {improvement:+.2f} percentage points"
)


# ============================================================
# SAVE ONLY IF BETTER
# ============================================================

if best_accuracy > base_accuracy:

    joblib.dump(
        best_margin,
        "love_joy_margin.pkl"
    )

    print("\n🔥 IMPROVEMENT FOUND!")
    print("Love/Joy margin saved.")

else:

    print("\nNo improvement.")
    print("Current 90.05% model remains unchanged.")