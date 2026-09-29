import joblib
import numpy as np

from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import ComplementNB
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score
from scipy.sparse import hstack


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

from preprocessing import preprocess_text

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
# LOAD FIXED CHARACTER BRANCHES
# ============================================================

print("\nLoading existing character vectorizers...")

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

print("Processed char weight:", char_weight)
print("Raw char weight:", raw_char_weight)


# ============================================================
# CHARACTER FEATURES
# ============================================================

print("\nCreating fixed character features...")

X_char_train = char_vectorizer.transform(
    train_processed
)

X_char_test = char_vectorizer.transform(
    test_processed
)

X_raw_train = raw_char_vectorizer.transform(
    train_texts
)

X_raw_test = raw_char_vectorizer.transform(
    test_texts
)

X_char_train = X_char_train * char_weight
X_char_test = X_char_test * char_weight

X_raw_train = X_raw_train * raw_char_weight
X_raw_test = X_raw_test * raw_char_weight

print(
    "Processed character features:",
    X_char_train.shape[1]
)

print(
    "Raw character features:",
    X_raw_train.shape[1]
)


# ============================================================
# EXPERIMENT 17
# WORD TF-IDF CONFIGURATION
# ============================================================

print("\n" + "=" * 70)
print("EXPERIMENT 17")
print("WORD TF-IDF REPRESENTATION SEARCH")
print("=" * 70)


# ------------------------------------------------------------
# Configurations
# ------------------------------------------------------------

configs = [

    {
        "name": "Baseline binary (1,2)",
        "ngram_range": (1, 2),
        "binary": True,
        "sublinear_tf": False
    },

    {
        "name": "Standard TF-IDF (1,2)",
        "ngram_range": (1, 2),
        "binary": False,
        "sublinear_tf": True
    },

    {
        "name": "Binary + sublinear (1,2)",
        "ngram_range": (1, 2),
        "binary": True,
        "sublinear_tf": True
    },

    {
        "name": "Binary unigrams only",
        "ngram_range": (1, 1),
        "binary": True,
        "sublinear_tf": False
    },

    {
        "name": "Binary bigrams only",
        "ngram_range": (2, 2),
        "binary": True,
        "sublinear_tf": False
    }
]


# ============================================================
# CROSS VALIDATION
# ============================================================

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

results = []


for config in configs:

    print("\n" + "-" * 60)
    print(config["name"])
    print("-" * 60)

    vectorizer = TfidfVectorizer(
        ngram_range=config["ngram_range"],
        max_features=30000,
        min_df=2,
        max_df=0.95,
        binary=config["binary"],
        use_idf=True,
        sublinear_tf=config["sublinear_tf"]
    )

    X_word_train = vectorizer.fit_transform(
        train_processed
    )

    print(
        "Word features:",
        X_word_train.shape[1]
    )

    # Combine with fixed character branches
    X_combined_train = hstack([
        X_word_train,
        X_char_train,
        X_raw_train
    ]).tocsr()

    model = ComplementNB(
        alpha=0.35,
        norm=False
    )

    scores = cross_val_score(
        model,
        X_combined_train,
        y_train,
        cv=cv,
        scoring="accuracy",
        n_jobs=-1
    )

    cv_accuracy = scores.mean()

    print(
        f"CV accuracy: {cv_accuracy * 100:.2f}%"
    )

    results.append(
        (
            cv_accuracy,
            config,
            vectorizer
        )
    )


# ============================================================
# SELECT BEST BY CV
# ============================================================

results.sort(
    key=lambda x: x[0],
    reverse=True
)

best_cv, best_config, best_vectorizer = results[0]


print("\n" + "=" * 70)
print("EXPERIMENT 17 CV RESULTS")
print("=" * 70)

for score, config, _ in results:

    print(
        f"{config['name']:<35} "
        f"{score * 100:.2f}%"
    )


# ============================================================
# FINAL TEST EVALUATION
# ============================================================

print("\n" + "=" * 70)
print("BEST CONFIGURATION")
print("=" * 70)

print(
    "Configuration:",
    best_config["name"]
)

print(
    "CV accuracy:",
    f"{best_cv * 100:.2f}%"
)


X_word_train = best_vectorizer.transform(
    train_processed
)

X_word_test = best_vectorizer.transform(
    test_processed
)

X_train_final = hstack([
    X_word_train,
    X_char_train,
    X_raw_train
]).tocsr()

X_test_final = hstack([
    X_word_test,
    X_char_test,
    X_raw_test
]).tocsr()


final_model = ComplementNB(
    alpha=0.35,
    norm=False
)

final_model.fit(
    X_train_final,
    y_train
)

y_pred = final_model.predict(
    X_test_final
)

test_accuracy = accuracy_score(
    y_test,
    y_pred
)


# ============================================================
# RESULT
# ============================================================

print("\n" + "=" * 70)
print("FINAL TEST RESULT")
print("=" * 70)

print(
    f"Test accuracy: {test_accuracy * 100:.2f}%"
)

print(
    "Current best: 90.05%"
)

improvement = (
    test_accuracy - 0.9005
) * 100

print(
    f"Improvement: {improvement:+.2f} percentage points"
)


# ============================================================
# SAVE ONLY IF BETTER
# ============================================================

if test_accuracy > 0.9005:

    print("\n🔥 NEW BEST MODEL!")

    joblib.dump(
        final_model,
        "emotion_model.pkl"
    )

    joblib.dump(
        best_vectorizer,
        "word_tfidf_vectorizer.pkl"
    )

    print("New model saved.")

else:

    print("\nNo improvement.")
    print(
        "Existing 90.05% model remains unchanged."
    )