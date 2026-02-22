import streamlit as st
import pandas as pd

st.set_page_config(page_title="Mini MBTI Quiz → Profile Text", layout="wide")

# ---------------------------------
# Data (optional chart)
# ---------------------------------
@st.cache_data
def load_mbti_data():
    df = pd.read_csv("mbti_1.csv")
    df = df.dropna(subset=["type", "posts"]).copy()
    df["type"] = df["type"].astype(str).str.strip().str.upper()
    return df

# ---------------------------------
# Load text blocks from file
# ---------------------------------
@st.cache_data
def load_text_blocks(filepath="quiz_profile_text_blocks.txt"):
    """
    File format:

    [E]
    text...

    [I]
    text...

    (blank line between blocks)
    """
    with open(filepath, "r", encoding="utf-8") as f:
        raw = f.read().strip()

    blocks = {}
    for chunk in raw.split("\n\n"):
        lines = [ln.strip() for ln in chunk.splitlines() if ln.strip()]
        if not lines:
            continue

        header = lines[0]
        if not (header.startswith("[") and header.endswith("]") and len(header) == 3):
            continue

        key = header[1]  # single letter like E/I/S/N/T/F/J/P
        text = " ".join(lines[1:]).strip()
        if text:
            blocks[key] = text

    required = set("EISNTFJP")
    missing = sorted(required - set(blocks.keys()))
    if missing:
        raise ValueError(
            f"Missing blocks in {filepath}: {', '.join(missing)}. "
            "Need one block each for [E][I][S][N][T][F][J][P]."
        )

    return blocks

def compute_type(q1, q2, q3, q4, q5, q6, q7, q8):
    # E / I
    E = sum([q1 == "Talk to everyone", q2 == "Energised"])
    # S / N
    S = sum([q3 == "Practical examples", q4 == "Details"])
    # T / F
    T = sum([q5 == "Logic", q6 == "Direct & objective"])
    # J / P
    J = sum([q7 == "Clear plans", q8 == "Finish early"])

    type_code = ""
    type_code += "E" if E >= 1 else "I"
    type_code += "S" if S >= 1 else "N"
    type_code += "T" if T >= 1 else "F"
    type_code += "J" if J >= 1 else "P"
    return type_code

def build_profile_text(type_code: str, blocks: dict, repeat_ei: int = 2) -> str:
    """
    repeat_ei = number of times to repeat the EI block to strengthen signal.
    For demos, 2 usually helps.
    """
    parts = []

    # Repeat the first letter block (E or I) to increase EI signal
    for _ in range(max(1, repeat_ei)):
        parts.append(blocks[type_code[0]])

    # Append S/N, T/F, J/P blocks once each
    for ch in type_code[1:]:
        parts.append(blocks[ch])

    return " ".join(parts).strip()

# ---------------------------------
# App UI
# ---------------------------------
st.title("Mini MBTI Quiz 🔮 → Profile Text Generator")
st.caption("Answer 8 quick questions → get your MBTI → generate a profile paragraph to copy into your NLP model.")

st.markdown("---")

# Load resources
try:
    BLOCKS = load_text_blocks("quiz_profile_text_blocks.txt")
except Exception as e:
    st.error(str(e))
    st.stop()

df = load_mbti_data()

# Sidebar quiz
st.sidebar.header("Quick Quiz (30 seconds)")

def question(label, a, b):
    return st.sidebar.radio(label, [a, b], horizontal=True)

# E / I
q1 = question("At hackathons you:", "Talk to everyone", "Stick with 1–2 people")
q2 = question("After a big social event you feel:", "Energised", "Drained")

# S / N
q3 = question("You prefer:", "Practical examples", "Big ideas & theory")
q4 = question("You focus more on:", "Details", "Patterns")

# T / F
q5 = question("In debates you prioritise:", "Logic", "People impact")
q6 = question("Feedback style:", "Direct & objective", "Considerate & supportive")

# J / P
q7 = question("You like:", "Clear plans", "Keeping options open")
q8 = question("Deadlines:", "Finish early", "Work in bursts")

st.sidebar.markdown("---")
repeat_ei = st.sidebar.slider("Boost E/I signal (repeat block)", min_value=1, max_value=4, value=2)
go = st.sidebar.button("Generate My Profile Text 🚀", use_container_width=True)

left, right = st.columns([1, 1], vertical_alignment="top")

with left:
    st.subheader("Your Result 🎯")

    if not go:
        st.info("Answer the quiz in the sidebar, then press **Generate My Profile Text 🚀**.")
    else:
        user_type = compute_type(q1, q2, q3, q4, q5, q6, q7, q8)
        st.markdown(f"## {user_type}")

        profile_text = build_profile_text(user_type, BLOCKS, repeat_ei=repeat_ei)

        st.subheader("Generated profile text (copy/paste) 🧾")
        st.text_area("", value=profile_text, height=260)
        st.caption("Click inside → Ctrl+A → Ctrl+C. Paste into your NLP model tab.")

        with st.expander("Show the blocks used"):
            st.write(f"Repeated EI block: {repeat_ei}×")
            for ch in user_type:
                st.write(f"[{ch}] {BLOCKS[ch]}")

with right:
    st.subheader("Dataset Visual (Optional) 📊")
    st.caption("Keeps it lively; safe to delete for a pure quiz-only page.")

    if not go:
        st.info("Generates after you produce your profile text.")
    else:
        type_counts = df["type"].value_counts()
        st.bar_chart(type_counts)

        pct = (df["type"] == user_type).mean() * 100
        st.metric(f"People in dataset with {user_type}", f"{pct:.2f}%")