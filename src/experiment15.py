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
# 3. EXISTING WORD TF-IDF
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

X_train_word = word_vectorizer.fit_transform(
    train_processed
)

X_test_word = word_vectorizer.transform(
    test_processed
)

print(
    "Word features:",
    X_train_word.shape[1]
)


# ============================================================
# 4. EXISTING PROCESSED CHARACTER TF-IDF
# ============================================================

print("\nCreating processed character TF-IDF...")

char_vectorizer = TfidfVectorizer(
    analyzer="char",
    ngram_range=(3, 6),
    max_features=75000,
    min_df=2,
    max_df=0.95,
    sublinear_tf=True
)

X_train_char = char_vectorizer.fit_transform(
    train_processed
)

X_test_char = char_vectorizer.transform(
    test_processed
)

print(
    "Processed character features:",
    X_train_char.shape[1]
)


# ============================================================
# 5. EXISTING CHARACTER WEIGHT
# ============================================================

existing_char_weight = 0.6

X_train_char = (
    X_train_char * existing_char_weight
)

X_test_char = (
    X_test_char * existing_char_weight
)


# ============================================================
# 6. NEW RAW-TEXT CHARACTER TF-IDF
# ============================================================

print("\nCreating RAW character TF-IDF...")

raw_char_vectorizer = TfidfVectorizer(
    analyzer="char",
    ngram_range=(3, 6),
    max_features=50000,
    min_df=2,
    max_df=0.95,
    sublinear_tf=True
)

X_train_raw_char = raw_char_vectorizer.fit_transform(
    train_texts
)

X_test_raw_char = raw_char_vectorizer.transform(
    test_texts
)

print(
    "Raw character features:",
    X_train_raw_char.shape[1]
)


# ============================================================
# 7. COMBINE BASE FEATURES
# ============================================================

X_train_base = hstack([
    X_train_word,
    X_train_char
])

X_test_base = hstack([
    X_test_word,
    X_test_char
])


# ============================================================
# 8. EXPERIMENT 15
# ============================================================

print("\n" + "=" * 70)
print("EXPERIMENT 15")
print("ADDING RAW-TEXT CHARACTER FEATURES")
print("=" * 70)

raw_weights = [
    0.05,
    0.10,
    0.15,
    0.20,
    0.25,
    0.30,
    0.40
]

alpha_values = [
    0.30,
    0.35,
    0.40,
    0.45,
    0.50,
    0.55,
    0.60
]


# ============================================================
# 9. CROSS VALIDATION
# ============================================================

skf = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

best_cv = -1
best_weight = None
best_alpha = None

results = []


for raw_weight in raw_weights:

    print("\n" + "-" * 60)
    print(
        "Testing raw character weight:",
        raw_weight
    )
    print("-" * 60)

    X_train = hstack([
        X_train_base,
        X_train_raw_char * raw_weight
    ])

    X_test = hstack([
        X_test_base,
        X_test_raw_char * raw_weight
    ])


    for alpha in alpha_values:

        fold_scores = []

        for train_idx, val_idx in skf.split(
            X_train,
            train_labels
        ):

            X_tr = X_train[train_idx]
            X_val = X_train[val_idx]

            y_tr = train_labels[train_idx]
            y_val = train_labels[val_idx]

            model = ComplementNB(
                alpha=alpha,
                norm=False
            )

            model.fit(
                X_tr,
                y_tr
            )

            predictions = model.predict(
                X_val
            )

            score = accuracy_score(
                y_val,
                predictions
            )

            fold_scores.append(score)


        cv_score = np.mean(
            fold_scores
        )

        results.append(
            (
                cv_score,
                raw_weight,
                alpha
            )
        )

        print(
            f"Weight={raw_weight:.2f} | "
            f"Alpha={alpha:.2f} | "
            f"CV={cv_score * 100:.2f}%"
        )


        if cv_score > best_cv:

            best_cv = cv_score
            best_weight = raw_weight
            best_alpha = alpha


# ============================================================
# 10. BEST CV RESULT
# ============================================================

print("\n" + "=" * 70)
print("BEST CV RESULT")
print("=" * 70)

print(
    "Raw character weight:",
    best_weight
)

print(
    "Alpha:",
    best_alpha
)

print(
    "CV accuracy:",
    round(best_cv * 100, 2),
    "%"
)


# ============================================================
# 11. FINAL MODEL
# ============================================================

X_train_final = hstack([
    X_train_base,
    X_train_raw_char * best_weight
])

X_test_final = hstack([
    X_test_base,
    X_test_raw_char * best_weight
])

final_model = ComplementNB(
    alpha=best_alpha,
    norm=False
)

final_model.fit(
    X_train_final,
    train_labels
)


# ============================================================
# 12. TEST ACCURACY
# ============================================================

test_predictions = final_model.predict(
    X_test_final
)

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

print(
    "Current best:",
    "89.95%"
)

print(
    "Improvement:",
    round(
        (test_accuracy - 0.8995) * 100,
        2
    ),
    "percentage points"
)


# ============================================================
# 13. SAVE ONLY IF BETTER
# ============================================================

if test_accuracy > 0.8995:

    print("\n" + "=" * 70)
    print("🔥 NEW BEST MODEL")
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
        raw_char_vectorizer,
        "raw_char_tfidf_vectorizer.pkl"
    )

    joblib.dump(
        existing_char_weight,
        "char_weight.pkl"
    )

    joblib.dump(
        best_weight,
        "raw_char_weight.pkl"
    )

    print("New model saved.")

else:

    print("\n" + "=" * 70)
    print("NO IMPROVEMENT")
    print("=" * 70)

    print(
        "Existing 89.95% model was NOT overwritten."
    )