import spacy
import re
import pandas as pd

nlp = spacy.load("en_core_web_trf")


def redact_text(text):
    doc = nlp(text)

    redacted = text
    for ent in doc.ents:
        if ent.label_ in ["PERSON", "GPE", "ORG"]:
            redacted = redacted.replace(ent.text, "[REDACTED]")

    redacted = re.sub(
        r"\b\d{3}[-.\s]??\d{2}[-.\s]??\d{4}\b", "[REDACTED]", redacted
    )  # SSN
    redacted = re.sub(r"\b\d{10,}\b", "[REDACTED]", redacted)
    redacted = re.sub(r"\S+@\S+\.\S+", "[REDACTED]", redacted)

    return redacted


df = pd.read_csv("data/hospital.csv")

df["Redacted_Feedback"] = df["Feedback"].apply(redact_text)


for i, text in enumerate(df["Feedback"].dropna()):
    doc = nlp(text)
    print(f"\nFeedback {i+1}: {text}")
    for ent in doc.ents:
        print(f"  - {ent.text} ({ent.label_})")
    if i == 50:
        break

print(df[["Feedback", "Redacted_Feedback"]].head(50))
