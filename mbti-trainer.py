import pandas as pd
import joblib
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

df = pd.read_csv("mbti_1.csv")
df = df.dropna(subset=["type", "posts"])

X = df["posts"]
y_EI = df["type"].str[0]
y_SN = df["type"].str[1]
y_TF = df["type"].str[2]
y_JP = df["type"].str[3]

def make_model():
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=8000,
            stop_words="english",
            ngram_range=(1,2),
            min_df=2
        )),
        ("clf", LogisticRegression(
            max_iter=2000,
            class_weight="balanced"
        ))
    ])

models = {
    "EI": make_model().fit(X, y_EI),
    "SN": make_model().fit(X, y_SN),
    "TF": make_model().fit(X, y_TF),
    "JP": make_model().fit(X, y_JP),
}

joblib.dump(models, "mbti_models.joblib")

print("Saved to mbti_models.joblib")