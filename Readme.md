# Practice tasks for AI

## Sentiment Analysis

### Task
**Sentiment Analysis on Customer Conversations**
>Applied NLP models to classify customer survey comments from medical facilities as positive, negative, or neutral—helping surface insights for service quality evaluation and improvement

Baseline done using **Huggingface**'s *sentiment-analysis* on *hospital_review* dataset from kaggle.


---

## Topic Modeling & Clustering

### Task
**Topic Modeling on Customer Survey Comments**  
> Trained topic modeling to group customer feedback into themes like wait time, discharge process, facilities, and doctor experience — helping identify key areas of concern and emerging service patterns.

### What’s Done
- Used the same `hospital_review` dataset from Kaggle.
- Applied `BERTopic` with `all-MiniLM-L6-v2` embedding model.
- Cleaned and preprocessed feedback data.
- Generated topic clusters using top keywords and document embeddings.
- Automatically assigned human-readable topic labels (e.g., “hospital / facilities / staff”).
- Created final DataFrame with:
- `Feedback`
- `Topics` (ID)
- `Topic Probabilities`
- `Topic Label`
- Sorted and filtered results.
- Handled outlier or uncategorized comments as `Other / Uncategorized`.

---