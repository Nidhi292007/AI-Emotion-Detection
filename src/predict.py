import joblib

from preprocessing import preprocess_text

model = joblib.load("emotion_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

emotion_names = [
    "Sadness",
    "Joy",
    "Love",
    "Anger",
    "Fear",
    "Surprise"
]

print("="*40)
print("Emotion Detection")
print("="*40)

while True:

    sentence = input("\nEnter Sentence : ")

    if sentence.lower()=="exit":
        break

    processed = preprocess_text(sentence)

    vector = vectorizer.transform([processed])

    prediction = model.predict(vector)[0]

    probability = model.predict_proba(vector)[0]

    print("\nPredicted Emotion :",emotion_names[prediction])

    print("Confidence :",round(max(probability)*100,2),"%")