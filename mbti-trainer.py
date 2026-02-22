import pandas as pd
import joblib

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# -----------------------
# Load data
# -----------------------
df = pd.read_csv("mbti_1.csv")
df = df.dropna(subset=["type", "posts"])

X = df["posts"].astype(str)

y_EI = df["type"].str[0]
y_SN = df["type"].str[1]
y_TF = df["type"].str[2]
y_JP = df["type"].str[3]

# -----------------------
# Models
# -----------------------
def make_model_general():
    # your original-ish baseline for S/N, T/F, J/P
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=8000,
            stop_words="english",
            ngram_range=(1, 2),
            min_df=2
        )),
        ("clf", LogisticRegression(
            max_iter=2000,
            class_weight="balanced"
        ))
    ])

def make_model_ei_push_to_E(e_boost=20.0):
    """
    Quick/dirty: push hard toward E by:
      - explicitly weighting E heavier than I
      - adding char n-grams (style cues)
      - slightly more features + sublinear TF
    Increase e_boost (e.g., 20, 30) if it still underpredicts E.
    """
    word = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        max_features=15000,
        min_df=1,
        sublinear_tf=True
    )
    char = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=2,
        sublinear_tf=True
    )

    return Pipeline([
        ("features", FeatureUnion([
            ("word", word),
            ("char", char),
        ])),
        ("clf", LogisticRegression(
            max_iter=5000,
            solver="liblinear",
            C=3.0,
            class_weight={"I": 1.0, "E": float(e_boost)}
        ))
    ])

# -----------------------
# Train
# -----------------------
models = {
    "EI": make_model_ei_push_to_E(e_boost=12.0).fit(X, y_EI),
    "SN": make_model_general().fit(X, y_SN),
    "TF": make_model_general().fit(X, y_TF),
    "JP": make_model_general().fit(X, y_JP),
}

# -----------------------
# Save
# -----------------------
joblib.dump(models, "mbti_models.joblib")
print("Saved to mbti_models.joblib")

# Optional: print class counts so you can sanity-check each head
print("\nClass balance:")
print("EI:", y_EI.value_counts().to_dict())
print("SN:", y_SN.value_counts().to_dict())
print("TF:", y_TF.value_counts().to_dict())
print("JP:", y_JP.value_counts().to_dict())