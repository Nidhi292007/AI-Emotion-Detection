from preprocessing import preprocess_text

sentence = "My favourite food is anything I didn't have to cook myself."

print("Original Sentence:\n")
print(sentence)

print("\nProcessed Sentence:\n")
print(preprocess_text(sentence))