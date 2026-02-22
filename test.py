import joblib
import pandas as pd
import numpy as np

models = joblib.load("mbti_models.joblib")
ei = models["EI"]

print("Loaded EI model")
print("EI classes_:", ei.named_steps["clf"].classes_)

# ---------
# Check prediction distribution on dataset (dirty sanity check)
# ---------
df = pd.read_csv("mbti_1.csv").dropna(subset=["posts"])
X = df["posts"].astype(str)

pred = ei.predict(X)
print("\nEI predicted distribution:")
print(pd.Series(pred).value_counts(normalize=True).round(3).to_dict())

# ---------
# Correct feature inspection for binary LogisticRegression
# coef_[0] corresponds to positive class = classes_[1]
# ---------
clf = ei.named_steps["clf"]
features = ei.named_steps["features"]

word_names = features.transformer_list[0][1].get_feature_names_out()
char_names = features.transformer_list[1][1].get_feature_names_out()
feature_names = np.concatenate([word_names, char_names])

coef = clf.coef_[0]  # only row
pos_class = clf.classes_[1]
neg_class = clf.classes_[0]

# Top features for pos_class have largest positive coef
top_pos = np.argsort(coef)[-25:][::-1]
top_neg = np.argsort(coef)[:25]

print(f"\nTop features for class '{pos_class}' (positive):")
for i in top_pos:
    print(feature_names[i], float(coef[i]))

print(f"\nTop features for class '{neg_class}' (negative):")
for i in top_neg:
    print(feature_names[i], float(coef[i]))

# ---------
# Push to E WITHOUT retraining: lower threshold for predicting E
# ---------
def predict_ei(texts, threshold=0.25):
    proba = ei.predict_proba(texts)
    e_idx = list(clf.classes_).index("E")
    pE = proba[:, e_idx]
    out = ["E" if p >= threshold else "I" for p in pE]
    return out, pE

print("\n--- Threshold test ---")
sample = ["I love meeting people and going out every weekend with friends!!!"]
out, pE = predict_ei(sample, threshold=0.25)
print("Prediction:", out[0], "P(E):", round(float(pE[0]), 3))