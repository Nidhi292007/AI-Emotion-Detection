import joblib
import numpy as np

from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import ComplementNB
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from scipy.sparse import hstack

from preprocessing import preprocess_text


# ============================================================
# 1. LOAD DATA
# ============================================================

print("Loading dataset...")

dataset = load_dataset("dair-ai/emotion")

train_texts = dataset["train"]["text"]
train_labels = np.array(dataset["train"]["label"])

test_texts = dataset["test"]["text"]
test_labels = np.array(dataset["test"]["label"])

print("Training samples:", len(train_texts))
print("Test samples:", len(test_texts))


# ============================================================
# 2. PREPROCESS
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
# 3. WORD TF-IDF
# ============================================================

print("\nCreating word TF-IDF...")

word_vectorizer = TfidfVectorizer(
    ngram_range=(1, 2),
    max_features=30000,
    min_df=2,
    max_df=0.95,
    binary=True,
    use_idf=True
)

X_train_word = word_vectorizer.fit_transform(train_processed)
X_test_word = word_vectorizer.transform(test_processed)

print("Word features:", X_train_word.shape[1])


# ============================================================
# 4. CHARACTER TF-IDF
# ============================================================

print("\nCreating character TF-IDF...")

char_vectorizer = TfidfVectorizer(
    analyzer="char",
    ngram_range=(3, 6),
    max_features=75000,
    min_df=2,
    max_df=0.95,
    sublinear_tf=True
)

X_train_char = char_vectorizer.fit_transform(train_processed)
X_test_char = char_vectorizer.transform(test_processed)

print("Character features:", X_train_char.shape[1])


# ============================================================
# 5. CHARACTER WEIGHT
# ============================================================

char_weight = 0.6

X_train_char = X_train_char * char_weight
X_test_char = X_test_char * char_weight


# ============================================================
# 6. COMBINE FEATURES
# ============================================================

print("\nCombining features...")

X_train = hstack([
    X_train_word,
    X_train_char
])

X_test = hstack([
    X_test_word,
    X_test_char
])

print("Final feature shape:", X_train.shape)


# ============================================================
# 7. EXPERIMENT 14
# ============================================================

print("\n" + "=" * 70)
print("EXPERIMENT 14")
print("TARGETED CLASS REBALANCING")
print("=" * 70)

# Class labels:
# 0 = Sadness
# 1 = Joy
# 2 = Love
# 3 = Anger
# 4 = Fear
# 5 = Surprise


love_weights = [1.0, 1.1, 1.2, 1.3]
fear_weights = [1.0, 1.05, 1.1]
surprise_weights = [1.0, 1.1, 1.2]

alpha = 0.45


# ============================================================
# 8. BASELINE
# ============================================================

print("\nBaseline:")
print("Love = 1.0")
print("Fear = 1.0")
print("Surprise = 1.0")
print("Alpha = 0.45")


# ============================================================
# 9. CROSS-VALIDATION
# ============================================================

skf = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)


results = []

best_cv = -1
best_weights = None


for love_w in love_weights:

    for fear_w in fear_weights:

        for surprise_w in surprise_weights:

            class_weights = np.array([
                1.0,        # Sadness
                1.0,        # Joy
                love_w,     # Love
                1.0,        # Anger
                fear_w,     # Fear
                surprise_w  # Surprise
            ])

            fold_scores = []

            for train_idx, val_idx in skf.split(
                X_train,
                train_labels
            ):

                X_tr = X_train[train_idx]
                X_val = X_train[val_idx]

                y_tr = train_labels[train_idx]
                y_val = train_labels[val_idx]

                sample_weights = class_weights[y_tr]

                model = ComplementNB(
                    alpha=alpha,
                    norm=False
                )

                model.fit(
                    X_tr,
                    y_tr,
                    sample_weight=sample_weights
                )

                predictions = model.predict(X_val)

                score = accuracy_score(
                    y_val,
                    predictions
                )

                fold_scores.append(score)

            cv_score = np.mean(fold_scores)

            results.append(
                (
                    cv_score,
                    love_w,
                    fear_w,
                    surprise_w
                )
            )

            print(
                f"Love={love_w:.2f} | "
                f"Fear={fear_w:.2f} | "
                f"Surprise={surprise_w:.2f} | "
                f"CV={cv_score * 100:.2f}%"
            )

            if cv_score > best_cv:

                best_cv = cv_score

                best_weights = (
                    love_w,
                    fear_w,
                    surprise_w
                )


# ============================================================
# 10. BEST CV RESULT
# ============================================================

best_love = best_weights[0]
best_fear = best_weights[1]
best_surprise = best_weights[2]

print("\n" + "=" * 70)
print("BEST CROSS-VALIDATION RESULT")
print("=" * 70)

print("Love weight:", best_love)
print("Fear weight:", best_fear)
print("Surprise weight:", best_surprise)
print("CV accuracy:", round(best_cv * 100, 2), "%")


# ============================================================
# 11. TRAIN FINAL MODEL
# ============================================================

final_class_weights = np.array([
    1.0,
    1.0,
    best_love,
    1.0,
    best_fear,
    best_surprise
])

final_sample_weights = final_class_weights[train_labels]

final_model = ComplementNB(
    alpha=alpha,
    norm=False
)

final_model.fit(
    X_train,
    train_labels,
    sample_weight=final_sample_weights
)


# ============================================================
# 12. TEST SET
# ============================================================

test_predictions = final_model.predict(X_test)

test_accuracy = accuracy_score(
    test_labels,
    test_predictions
)

print("\n" + "=" * 70)
print("FINAL TEST RESULT")
print("=" * 70)

print(
    "Test accuracy:",
    round(test_accuracy * 100, 2),
    "%"
)


# ============================================================
# 13. COMPARE WITH CURRENT MODEL
# ============================================================

CURRENT_BEST = 0.8995

print("\nCurrent model:", CURRENT_BEST * 100, "%")
print(
    "New model:",
    round(test_accuracy * 100, 2),
    "%"
)

improvement = (
    test_accuracy - CURRENT_BEST
) * 100

print(
    "Improvement:",
    round(improvement, 2),
    "percentage points"
)


# ============================================================
# 14. SAVE ONLY IF BETTER
# ============================================================

if test_accuracy > CURRENT_BEST:

    print("\n" + "=" * 70)
    print("🔥 NEW BEST MODEL!")
    print("=" * 70)

    joblib.dump(
        final_model,
        "emotion_model.pkl"
    )

    joblib.dump(
        word_vectorizer,
        "word_tfidf_vectorizer.pkl"
    )

    joblib.dump(
        char_vectorizer,
        "char_tfidf_vectorizer.pkl"
    )

    joblib.dump(
        char_weight,
        "char_weight.pkl"
    )

    print("Model saved successfully.")

else:

    print("\n" + "=" * 70)
    print("No improvement.")
    print("=" * 70)

    print(
        "Existing 89.95% model was NOT overwritten."
    )