from datasets import load_dataset

dataset = load_dataset("dair-ai/emotion")

labels = [
    "Sadness",
    "Joy",
    "Love",
    "Anger",
    "Fear",
    "Surprise"
]

print("Emotion Labels\n")

for i, emotion in enumerate(labels):
    print(f"{i} --> {emotion}")