# Sentiment Analysis on Customer Conversations
# Applied NLP models to classify customer survey comments from
# medical facilities as positive, negative, or neutral—helping surface insights for
# service quality evaluation and improvement


# Baseline using hugging face general model

import pandas as pd
from sklearn.metrics import classification_report, accuracy_score

# huggingface sentiment model
from transformers import pipeline

df = pd.read_csv("data/hospital.csv")

# preprocess
df.drop(columns=["Unnamed: 3"], errors="ignore")
df.dropna(subset=["Feedback"])

# print(df.head())
# print(df.columns)

sentiment_pipeline = pipeline("sentiment-analysis")
results = sentiment_pipeline(df["Feedback"].tolist())
# print(results)

df["predicted_label"] = [1 if r["label"] == "POSITIVE" else 0 for r in results]
df["confidence"] = [r["score"] for r in results]

print("Accuracy: ", accuracy_score(df["Sentiment Label"], df["predicted_label"]))
print(
    "Classification report\n",
    classification_report(df["Sentiment Label"], df["predicted_label"]),
)
