# Practice Tasks for AI

A collection of AI mini-projects covering NLP, ML, LLMs, and generative AI — each task is in its own folder and built as a Jupyter Notebook for clear demonstration and reproducibility.

---

## 1. Sentiment Analysis

**Task**  
Applied NLP models to classify customer survey comments from medical facilities as **positive**, **negative**, or **neutral** — helping surface insights for service quality evaluation and improvement.

**Details**
- Dataset: `hospital_review` from Kaggle.
- Baseline using Hugging Face `pipeline("sentiment-analysis")`.
- Saved predictions with confidence scores for each comment.

---

## 2. Topic Modeling & Clustering

**Task**  
Trained topic modeling to group customer feedback into themes like wait time, discharge process, facilities, and doctor experience — helping identify key areas of concern and emerging service patterns.

**Details**
- Same `hospital_review` dataset.
- Used `BERTopic` with `all-MiniLM-L6-v2` embeddings.
- Cleaned and preprocessed text.
- Generated topic clusters and auto-labeled using top keywords.
- Created final DataFrame with:
  - `Feedback`
  - `Topic ID`
  - `Topic Probabilities`
  - `Topic Label`
- Sorted, filtered, and handled outliers as `Other / Uncategorized`.

---

## 3. Named Entity Recognition (NER)

**Task**  
Used NER to identify sensitive information such as names, locations, and organizations from customer feedback — supporting privacy compliance.

**Details**
- Model: `spacy` `en_core_web_trf`.
- Input: same `hospital_review` dataset.
- Printed detected entities by type (e.g., PERSON, ORG) for each comment.

---

## 4. Information Extraction from Unstructured Data

**Task**  
Built an LLM-based system to process multiple litigation PDF files — extracting key metadata, organizing it, and preparing it for fast search or review.

**Details**
- Extracted raw text using `PyPDF2`.
- Used Hugging Face QA pipeline with `deepset/roberta-base-squad2`.
- Asked 10 custom questions (case number, parties, court, date, summary, etc.).
- Stored results as `.json` per file.
- Combined all results into a single `.csv` file.
- Added execution timing to track performance.

---

## 5. Speech-to-Text (STT)

**Task**  
Integrated an open-source STT model to transcribe audio files — supporting multilingual or accessible use cases.

**Details**
- Model: `openai/whisper-base` via Hugging Face `pipeline("automatic-speech-recognition")`.
- Input: `.wav` file (`harvard.wav` sample).
- Output: printed full transcript.
- No live mic input — only file-based transcription.

---

## 6. Text-to-Speech (TTS)

**Task**  
Generated audio from text prompts for accessible content or multilingual support.

**Details**
- Model: `suno/bark-small` from Hugging Face.
- Text converted to speech and saved as `.wav` output file.
- Demonstrates more natural speech compared to local TTS engines.

---

## 7. Image Generation from Text

**Task**  
Generated stylized visual assets based on text prompts.

**Details**
- Model: Stable Diffusion `v1.5` via `diffusers`.
- Prompt: Generated an orc character holding an axe.
- Output saved as image file.

---

## 8. Predictive Modeling for User Conversion

**Task**  
Built a predictive ML model to forecast user behavior — e.g., whether an appointment will be a show or no-show.

**Details**
- Dataset: appointment show/no-show data.
- Model: `RandomForestClassifier` from `scikit-learn`.
- Input: features like date, time, patient data.
- Output: predicted class — show or no-show.
- Evaluated using accuracy score.

---

## 9. AI Chatbot with RAG

**Task**  
Built a Retrieval-Augmented Generation (RAG) pipeline to answer questions grounded in external documents.

**Details**
- Data: Premier League 24/25 season stats.
- Chunked rows, embedded using `sentence-transformers` `all-MiniLM-L6-v2`.
- Stored embeddings in `ChromaDB`.
- Queried using OpenAI LLM to generate answers backed by retrieved chunks.

---

## Dataset Links (Kaggle)
- [Hospital Reviews](https://www.kaggle.com/datasets/junaid6731/hospital-reviews-dataset)
- [Appointment No show](https://www.kaggle.com/datasets/joniarroba/noshowappointments)
- [Premier League](https://www.kaggle.com/datasets/aesika/english-premier-league-player-stats-2425)

