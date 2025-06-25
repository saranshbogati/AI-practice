from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

model_path = "./models/hospital-sentiment-model"
tokenizer_path = "./tokenizers/hospital-sentiment-model"
tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

texts = [
    "The doctor was incredibly patient and explained everything clearly.",
    "I waited for over two hours and nobody even apologized.",
    "Great service overall, though the parking situation was terrible.",
    "The emergency room was chaotic and understaffed.",
    "Clean facilities, friendly staff, and quick check-in process.",
    "The receptionist was very rude, but the nurse was kind.",
    "Everything was fine — nothing stood out either way.",
    "My surgery went smoothly and the recovery room was comfortable.",
    "The food was awful and the bed was uncomfortable.",
    "They lost my paperwork twice. Completely unacceptable experience.",
]

sentiment_pipe = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

results = sentiment_pipe(texts)
for text, r in zip(texts, results):
    label = "Positive" if r["label"] == "LABEL_1" else "Negative"
    print(f"{label} ({r['score']:.2f}): {text}")
