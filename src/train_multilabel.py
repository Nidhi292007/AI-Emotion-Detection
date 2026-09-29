from datasets import load_dataset
from preprocessing import preprocess_text

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import MultinomialNB

from sklearn.metrics import classification_report
from sklearn.metrics import hamming_loss
from sklearn.metrics import accuracy_score

import joblib

print("Loading GoEmotions...")

dataset = load_dataset(
    "google-research-datasets/go_emotions",
    "simplified"
)

train = dataset["train"]
test = dataset["test"]

print("Preprocessing...")

X_train_text = [preprocess_text(x["text"]) for x in train]
X_test_text = [preprocess_text(x["text"]) for x in test]

print("Creating TF-IDF...")

vectorizer = TfidfVectorizer(
    max_features=10000,
    ngram_range=(1,2),
    min_df=2,
    max_df=0.95
)

X_train = vectorizer.fit_transform(X_train_text)
X_test = vectorizer.transform(X_test_text)

print("Preparing labels...")

mlb = MultiLabelBinarizer()

y_train = mlb.fit_transform([x["labels"] for x in train])
y_test = mlb.transform([x["labels"] for x in test])

print("Training model...")

model = OneVsRestClassifier(MultinomialNB())

model.fit(X_train, y_train)

print("Predicting...")

pred = model.predict(X_test)

print("\nSubset Accuracy")

print(accuracy_score(y_test,pred))

print("\nHamming Loss")

print(hamming_loss(y_test,pred))

print("\nClassification Report")

print(classification_report(
    y_test,
    pred,
    zero_division=0
))

print("\nSaving model...")

joblib.dump(model,"models/emotion_model.pkl")
joblib.dump(vectorizer,"models/tfidf.pkl")
joblib.dump(mlb,"models/mlb.pkl")

print("\nFinished!")