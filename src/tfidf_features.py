from datasets import load_dataset
from preprocessing import preprocess_text
from sklearn.feature_extraction.text import TfidfVectorizer

print("Loading dataset...")

dataset = load_dataset(
    "google-research-datasets/go_emotions",
    "simplified"
)

train = dataset["train"]

print("Preprocessing text...")

texts = [preprocess_text(sample["text"]) for sample in train]

print("Creating TF-IDF vectors...")

vectorizer = TfidfVectorizer(
    max_features=30000,
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.95,
    sublinear_tf=True
)

X = vectorizer.fit_transform(texts)

print("\nTF-IDF Matrix Shape:")

print(X.shape)

print("\nNumber of Features:")

print(len(vectorizer.get_feature_names_out()))

print("\nFirst 20 Features:")

print(vectorizer.get_feature_names_out()[:20])