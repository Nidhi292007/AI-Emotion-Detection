import joblib
import numpy as np

from datasets import load_dataset
from sklearn.metrics import accuracy_score
from scipy.sparse import hstack

from preprocessing import preprocess_text


# ============================================================
# 1. LOAD DATA
# ============================================================

print("Loading dataset...")

dataset = load_dataset("dair-ai/emotion")

test_texts = dataset["test"]["text"]
test_labels = np.array(dataset["test"]["label"])

print("Test samples:", len(test_texts))


# ============================================================
# 2. PREPROCESS
# ============================================================

print("\nPreprocessing...")

test_processed = [
    preprocess_text(text)
    for text in test_texts
]


# ============================================================
# 3. LOAD CURRENT MODEL
# ============================================================

print("\nLoading current 89.95% model...")

model = joblib.load("emotion_model.pkl")
word_vectorizer = joblib.load("word_tfidf_vectorizer.pkl")
char_vectorizer = joblib.load("char_tfidf_vectorizer.pkl")
char_weight = joblib.load("char_weight.pkl")


# ============================================================
# 4. TRANSFORM TEST DATA
# ============================================================

X_word = word_vectorizer.transform(test_processed)

X_char = char_vectorizer.transform(test_processed)
X_char = X_char * char_weight

X_test = hstack([
    X_word,
    X_char
])


# ============================================================
# 5. PREDICT
# ============================================================

predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)

accuracy = accuracy_score(
    test_labels,
    predictions
)

print("\nCurrent accuracy:", round(accuracy * 100, 2), "%")


# ============================================================
# 6. LABEL NAMES
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
# 7. FUNCTION TO SHOW CONFUSIONS
# ============================================================

def show_confusions(actual_class, predicted_class, max_examples=15):

    print("\n")
    print("=" * 80)
    print(
        f"ACTUAL: {labels[actual_class]} "
        f"--> PREDICTED: {labels[predicted_class]}"
    )
    print("=" * 80)

    count = 0

    for i in range(len(test_texts)):

        if (
            test_labels[i] == actual_class
            and predictions[i] == predicted_class
        ):

            print(f"\nExample {count + 1}")
            print("-" * 60)

            print("Original text:")
            print(test_texts[i])

            print("\nPreprocessed:")
            print(test_processed[i])

            print("\nModel probabilities:")

            sorted_indices = np.argsort(
                probabilities[i]
            )[::-1]

            for idx in sorted_indices[:3]:

                print(
                    f"  {labels[idx]:<10}: "
                    f"{probabilities[i][idx] * 100:.2f}%"
                )

            count += 1

            if count >= max_examples:
                break

    print(
        f"\nShowing {count} examples."
    )


# ============================================================
# 8. IMPORTANT CONFUSION PAIRS
# ============================================================

# Love -> Joy
show_confusions(
    actual_class=2,
    predicted_class=1,
    max_examples=15
)

# Joy -> Love
show_confusions(
    actual_class=1,
    predicted_class=2,
    max_examples=15
)

# Fear -> Surprise
show_confusions(
    actual_class=4,
    predicted_class=5,
    max_examples=10
)

# Surprise -> Fear
show_confusions(
    actual_class=5,
    predicted_class=4,
    max_examples=10
)

# Anger -> Sadness
show_confusions(
    actual_class=3,
    predicted_class=0,
    max_examples=10
)

# Fear -> Sadness
show_confusions(
    actual_class=4,
    predicted_class=0,
    max_examples=10
)