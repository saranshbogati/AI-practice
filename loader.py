import pandas as pd


def get_clean_hospital_data():
    df = pd.read_csv("data/hospital.csv")
    # preprocess
    df.drop(columns=["Unnamed: 3"], errors="ignore")
    df.dropna(subset=["Feedback", "Sentiment Label"])

    return df
