import joblib
import numpy as np
from datasets import load_dataset
from sklearn.metrics import classification_report, confusion_matrix

from preprocessing import preprocess_text


# ============================================================
# LOAD MODEL
# ============================================================

print("Loading dataset...")

dataset = load_dataset("dair-ai/emotion")

test_texts = dataset["test"]["text"]
y_test = np.array(dataset["test"]["label"])

print("Test samples:", len(test_texts))


# ============================================================
# PREPROCESS
# ============================================================

print("\nPreprocessing test data...")

processed_texts = [
    preprocess_text(text)
    for text in test_texts
]


# ============================================================
# LOAD SAVED MODEL
# ============================================================

print("\nLoading saved model...")

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
# TRANSFORM FEATURES
# ============================================================

print("\nCreating features...")

# Word features
X_word = word_vectorizer.transform(
    processed_texts
)

# Processed character features
X_char = char_vectorizer.transform(
    processed_texts
)

# Raw character features
X_raw_char = raw_char_vectorizer.transform(
    test_texts
)


# ============================================================
# COMBINE
# ============================================================

X_combined = np.hstack([
    X_word.toarray(),
    (X_char * char_weight).toarray(),
    (X_raw_char * raw_char_weight).toarray()
])


# ============================================================
# PREDICTION
# ============================================================

print("\nPredicting...")

y_pred = model.predict(X_combined)


# ============================================================
# RESULTS
# ============================================================

labels = [
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

print(
    classification_report(
        y_test,
        y_pred,
        target_names=labels,
        digits=4
    )
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_test,
    y_pred
)

print("\n" + "=" * 70)
print("CONFUSION MATRIX")
print("=" * 70)

print(
    "              " +
    " ".join(f"{x:>10}" for x in labels)
)

for i, row in enumerate(cm):
    print(
        f"{labels[i]:<12}" +
        " ".join(f"{x:>10}" for x in row)
    )


# ============================================================
# TOP CONFUSIONS
# ============================================================

print("\n" + "=" * 70)
print("TOP CONFUSIONS")
print("=" * 70)

confusions = []

for i in range(len(labels)):
    for j in range(len(labels)):
        if i != j:
            confusions.append(
                (
                    cm[i][j],
                    labels[i],
                    labels[j]
                )
            )

confusions.sort(reverse=True)

for count, actual, predicted in confusions[:15]:
    print(
        f"{actual} -> {predicted}: {count}"
    )