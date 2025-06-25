# Sentiment Analysis on Customer Conversations
# Applied NLP models to classify customer survey comments from
# medical facilities as positive, negative, or neutral—helping surface insights for
# service quality evaluation and improvement


# Baseline using hugging face general model

import pandas as pd
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    precision_recall_fscore_support,
)

# huggingface sentiment model
from sklearn.model_selection import train_test_split
from transformers import pipeline
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)
from loader import get_clean_hospital_data


df = get_clean_hospital_data()


def process_baseline():
    sentiment_pipeline = pipeline("sentiment-analysis")
    results = sentiment_pipeline(df["Feedback"].tolist())
    # print(results)

    df["predicted_label"] = [1 if r["label"] == "POSITIVE" else 0 for r in results]
    df["confidence"] = [r["score"] for r in results]

    print("Baseline on pretrained model\n")
    print("Accuracy: ", accuracy_score(df["Sentiment Label"], df["predicted_label"]))
    print(
        "Classification report\n",
        classification_report(df["Sentiment Label"], df["predicted_label"]),
    )


# fine tuning model to dataset
df = df.rename(columns={"Feedback": "text", "Sentiment Label": "label"})
train_texts, val_texts, train_labels, val_labels = train_test_split(
    df["text"].tolist(),
    df["label"].tolist(),
    test_size=0.2,
    random_state=42,
    stratify=df["label"],
)

train_dataset = Dataset.from_dict({"text": train_texts, "label": train_labels})
val_dataset = Dataset.from_dict({"text": val_texts, "label": val_labels})

model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)


def tokenize(batch):
    return tokenizer(batch["text"], padding=True, truncation=True, max_length=128)


def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="binary"
    )
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}


train_dataset = train_dataset.map(tokenize, batched=True)
val_dataset = val_dataset.map(tokenize, batched=True)

train_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])
val_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])

training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    num_train_epochs=4,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    save_total_limit=2,
    seed=42,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
)

trainer.train()
eval_results = trainer.evaluate()

print("Evaluation results:", eval_results)

trainer.save_model("./models/hospital-sentiment-model")
tokenizer.save_pretrained("./tokenizers/hospital-sentiment-model")

print("Complete!!!!")
