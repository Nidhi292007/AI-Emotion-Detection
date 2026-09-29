import joblib
import numpy as np

from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import ComplementNB
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
from scipy.sparse import hstack

from preprocessing import preprocess_text


# ============================================================
# 1. LOAD DATASET
# ============================================================

print("Loading dataset...")

dataset = load_dataset("dair-ai/emotion")

train_texts = dataset["train"]["text"]
train_labels = dataset["train"]["label"]

test_texts = dataset["test"]["text"]
test_labels = dataset["test"]["label"]

print("Training samples:", len(train_texts))
print("Test samples:", len(test_texts))


# ============================================================
# 2. PREPROCESSING
# ============================================================

print("\nPreprocessing text...")

train_processed = [preprocess_text(text) for text in train_texts]
test_processed = [preprocess_text(text) for text in test_texts]

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
# 6. COMBINE WORD + CHARACTER FEATURES
# ============================================================

print("\nCombining word + character features...")

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
# 7. EXPERIMENT 13
#    ComplementNB norm=False vs norm=True
# ============================================================

print("\n" + "=" * 60)
print("EXPERIMENT 13")
print("Testing ComplementNB norm parameter")
print("=" * 60)

alpha_values = [
    0.20,
    0.25,
    0.30,
    0.35,
    0.40,
    0.45,
    0.50,
    0.55,
    0.60,
    0.70,
    0.80
]

results = []

best_cv_score = -1
best_model = None
best_alpha = None
best_norm = None


for norm_value in [False, True]:

    print("\n----------------------------------------")
    print("Testing norm =", norm_value)
    print("----------------------------------------")

    model = ComplementNB(norm=norm_value)

    grid = GridSearchCV(
        model,
        {
            "alpha": alpha_values
        },
        cv=5,
        scoring="accuracy",
        n_jobs=-1
    )

    grid.fit(X_train, train_labels)

    cv_score = grid.best_score_
    alpha = grid.best_params_["alpha"]

    test_predictions = grid.best_estimator_.predict(X_test)
    test_accuracy = accuracy_score(
        test_labels,
        test_predictions
    )

    print("Best alpha:", alpha)
    print("CV accuracy:", round(cv_score * 100, 2), "%")
    print("Test accuracy:", round(test_accuracy * 100, 2), "%")

    results.append({
        "norm": norm_value,
        "alpha": alpha,
        "cv": cv_score,
        "test": test_accuracy
    })

    # Select based on CV performance
    if cv_score > best_cv_score:
        best_cv_score = cv_score
        best_model = grid.best_estimator_
        best_alpha = alpha
        best_norm = norm_value


# ============================================================
# 8. DISPLAY ALL RESULTS
# ============================================================

print("\n" + "=" * 60)
print("EXPERIMENT 13 RESULTS")
print("=" * 60)

for result in results:

    print(
        "norm =", result["norm"],
        "| alpha =", result["alpha"],
        "| CV =",
        round(result["cv"] * 100, 2),
        "%",
        "| Test =",
        round(result["test"] * 100, 2),
        "%"
    )


# ============================================================
# 9. FINAL EVALUATION
# ============================================================

final_predictions = best_model.predict(X_test)

final_accuracy = accuracy_score(
    test_labels,
    final_predictions
)

print("\n" + "=" * 60)
print("BEST MODEL")
print("=" * 60)

print("norm:", best_norm)
print("alpha:", best_alpha)
print("CV accuracy:", round(best_cv_score * 100, 2), "%")
print("Test accuracy:", round(final_accuracy * 100, 2), "%")

print("\nClassification Report:\n")

print(
    classification_report(
        test_labels,
        final_predictions,
        target_names=[
            "Sadness",
            "Joy",
            "Love",
            "Anger",
            "Fear",
            "Surprise"
        ]
    )
)


# ============================================================
# 10. SAVE ONLY IF >= 90%
# ============================================================

CURRENT_BEST = 0.8995

if final_accuracy >= 0.90:

    print("\n" + "=" * 60)
    print("🎉 90%+ ACHIEVED!")
    print("=" * 60)

    print(
        "New test accuracy:",
        round(final_accuracy * 100, 2),
        "%"
    )

    print("Saving new model...")

    joblib.dump(
        best_model,
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

    print("\nModel saved successfully.")

else:

    print("\n" + "=" * 60)
    print("90% NOT REACHED")
    print("=" * 60)

    print(
        "Best test accuracy:",
        round(final_accuracy * 100, 2),
        "%"
    )

    print(
        "Existing 89.95% model was NOT overwritten."
    )