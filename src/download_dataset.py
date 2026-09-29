from datasets import load_dataset

print("Downloading Emotion Dataset...")

dataset = load_dataset("dair-ai/emotion")

print(dataset)

print("\nTraining Samples:", len(dataset["train"]))
print("Validation Samples:", len(dataset["validation"]))
print("Testing Samples:", len(dataset["test"]))

print("\nFirst Sample:")
print(dataset["train"][0])