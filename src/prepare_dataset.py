from datasets import load_dataset
from preprocessing import preprocess_text

print("Loading GoEmotions Dataset...")

dataset = load_dataset("google-research-datasets/go_emotions", "simplified")

train = dataset["train"]

print("Preprocessing training data...")

processed_texts = []

for sample in train:
    cleaned = preprocess_text(sample["text"])
    processed_texts.append(cleaned)

print("\nFinished preprocessing!")

print("\nOriginal:")
print(train[0]["text"])

print("\nProcessed:")
print(processed_texts[0])

print("\nTotal processed samples:", len(processed_texts))