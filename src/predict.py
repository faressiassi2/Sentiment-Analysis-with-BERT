from transformers import pipeline

classifier = pipeline(
    "sentiment-analysis",
    model="model",
    tokenizer="model"
)

texts = [
    "I love this product!",
    "This is okay, nothing special.",
    "Worst experience ever."
]

for t in texts:
    print(t, "→", classifier(t))
