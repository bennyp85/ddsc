import streamlit as st
import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# ---------------------------------
# Config
# ---------------------------------
st.set_page_config(page_title="Live MBTI Predictor", layout="wide")
MIN_CHARS = 220  # helps reduce “majority class” collapse on short inputs

# ---------------------------------
# Load data
# ---------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("mbti_1.csv")
    # Basic cleanup
    df = df.dropna(subset=["type", "posts"]).copy()
    df["type"] = df["type"].astype(str).str.strip().str.upper()
    df["posts"] = df["posts"].astype(str)
    return df

df = load_data()

# ---------------------------------
# Train 4 binary classifiers (EI, SN, TF, JP)
# ---------------------------------
import joblib
import os

@st.cache_resource
def load_models():
    if not os.path.exists("mbti_models.joblib"):
        st.error("Model file not found. Run training script first.")
        st.stop()
    return joblib.load("mbti_models.joblib")

models = load_models()

# ---------------------------------
# Prediction helpers
# ---------------------------------
DIM_LETTERS = {
    "EI": ("E", "I"),
    "SN": ("S", "N"),
    "TF": ("T", "F"),
    "JP": ("J", "P"),
}

ALL_TYPES = [a + b + c + d
             for a in DIM_LETTERS["EI"]
             for b in DIM_LETTERS["SN"]
             for c in DIM_LETTERS["TF"]
             for d in DIM_LETTERS["JP"]]

def predict_dimensions(text: str):
    """Returns chosen letters and per-dimension probability dicts."""
    chosen = {}
    probs = {}

    for dim, model in models.items():
        clf = model.named_steps["clf"]
        classes = list(clf.classes_)  # e.g. ["E","I"] (order may vary)

        p = model.predict_proba([text])[0]
        p_map = {classes[i]: float(p[i]) for i in range(len(classes))}
        probs[dim] = p_map

        pred = model.predict([text])[0]
        chosen[dim] = pred

    mbti = chosen["EI"] + chosen["SN"] + chosen["TF"] + chosen["JP"]
    return mbti, probs

def type_distribution_from_dim_probs(dim_probs: dict):
    """Compute 16-type probabilities by multiplying dimension probabilities, then normalize."""
    scores = []
    for t in ALL_TYPES:
        s = (
            dim_probs["EI"].get(t[0], 0.0)
            * dim_probs["SN"].get(t[1], 0.0)
            * dim_probs["TF"].get(t[2], 0.0)
            * dim_probs["JP"].get(t[3], 0.0)
        )
        scores.append(s)

    scores = np.array(scores, dtype=float)
    if scores.sum() > 0:
        scores = scores / scores.sum()
    return pd.DataFrame({"type": ALL_TYPES, "prob": scores}).sort_values("prob", ascending=False)

def pct_dataset(t: str) -> float:
    return float((df["type"] == t).mean() * 100)

# ---------------------------------
# UI
# ---------------------------------
st.title("Live MBTI Prediction 🔮📊")
st.caption("Tiny NLP model trained on 8,675 users. You type text → it predicts your MBTI (as 4 binary decisions).")

with st.expander("Quick notes (for presenting) 🙂", expanded=False):
    st.write(
        "This is a fun demo, not a psychological assessment. "
        "Predictions improve with longer text because the training data is long forum-style writing."
    )

st.markdown("---")

left, right = st.columns([1, 1], vertical_alignment="top")

with left:
    st.subheader("1) Write something")
    st.write("A short paragraph about how you think, work, argue, plan, or make decisions. More text = better signal. ✍️")
    user_text = st.text_area(
        "Your text",
        height=220,
        placeholder="Example: I like to plan early, break problems into steps, and test assumptions...",
    )

    c1, c2 = st.columns([1, 1])
    with c1:
        go = st.button("Predict 🚀", use_container_width=True)
    with c2:
        sample = st.button("Use a sample text 🎲", use_container_width=True)

    if sample:
        user_text = (
            "I like to explore possibilities and connect ideas across topics. "
            "I usually start with a big-picture model, then iterate with quick experiments. "
            "I care about whether solutions are fair and how they affect people, "
            "but I also want the logic to be consistent. Deadlines help me focus, "
            "though I prefer flexibility early on."
        )
        st.session_state["__sample_injected__"] = True
        st.rerun()

    st.markdown("---")
    st.subheader("Dataset vibe")
    type_counts = df["type"].value_counts()
    st.bar_chart(type_counts)

with right:
    st.subheader("2) Prediction + confidence")

    if not go:
        st.info("Hit **Predict 🚀** when you’re ready.")
    else:
        txt = (user_text or "").strip()
        if len(txt) < MIN_CHARS:
            st.warning(f"Give it a bit more text (aim for {MIN_CHARS}+ characters). Current: {len(txt)} 🙂")
            st.stop()

        mbti, dim_probs = predict_dimensions(txt)
        dist16 = type_distribution_from_dim_probs(dim_probs)

        st.markdown(f"## Predicted type: **{mbti}**")

        # Show per-dimension confidence as small bars
        st.write("Confidence by dimension (probabilities):")
        dim_rows = []
        for dim, (a, b) in DIM_LETTERS.items():
            pa = dim_probs[dim].get(a, 0.0)
            pb = dim_probs[dim].get(b, 0.0)
            pick = a if pa >= pb else b
            dim_rows.append(
                {
                    "Dimension": dim,
                    "Pick": pick,
                    a: pa,
                    b: pb,
                    "Confidence (max)": max(pa, pb),
                }
            )
        dim_df = pd.DataFrame(dim_rows)
        st.dataframe(
            dim_df[["Dimension", "Pick", "Confidence (max)"]].style.format({"Confidence (max)": "{:.2f}"}),
            use_container_width=True,
            hide_index=True,
        )

        st.markdown("---")

        # Top-k MBTI types (derived from 4 probabilities)
        st.write("Top predicted types (derived from the 4 dimension probabilities):")
        topk = dist16.head(6).copy()
        topk["prob"] = topk["prob"].astype(float)
        st.bar_chart(topk.set_index("type")["prob"])

        st.markdown("---")

        p_dataset = pct_dataset(mbti)
        st.metric("How common this type is in the dataset", f"{p_dataset:.2f}%")

        st.write("Fun prompt for the room 🗣️: does that feel right? If not, what would you type to ‘steer’ it? 😄")