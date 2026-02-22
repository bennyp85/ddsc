import streamlit as st
import pandas as pd

st.set_page_config(page_title="Mini MBTI for Data Science", layout="wide")

# -----------------------------
# Load MBTI dataset
# -----------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("mbti_1.csv")
    return df

df = load_data()

st.title("Mini MBTI Quiz 🔮 (Data Science Edition)")
st.caption("8 quick questions → get your type → compare with real MBTI data.")

st.markdown("---")

# -----------------------------
# Quiz Questions
# -----------------------------
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

go = st.sidebar.button("Reveal My Type 🚀", use_container_width=True)

# -----------------------------
# Compute Type
# -----------------------------
def compute_type():
    E = sum([q1 == "Talk to everyone", q2 == "Energised"])
    S = sum([q3 == "Practical examples", q4 == "Details"])
    T = sum([q5 == "Logic", q6 == "Direct & objective"])
    J = sum([q7 == "Clear plans", q8 == "Finish early"])

    type_code = ""
    type_code += "E" if E >= 1 else "I"
    type_code += "S" if S >= 1 else "N"
    type_code += "T" if T >= 1 else "F"
    type_code += "J" if J >= 1 else "P"

    return type_code

# -----------------------------
# Layout
# -----------------------------
left, right = st.columns([1, 1])

with left:
    st.subheader("Your Result 🎯")

    if not go:
        st.info("Answer the questions in the sidebar and press **Reveal My Type 🚀**")
    else:
        user_type = compute_type()
        st.markdown(f"## {user_type}")

        st.write("Fun interpretation:")
        interpretations = {
            "INTJ": "Strategic data architect 🧠",
            "ENTJ": "Vision-driven tech lead 🚀",
            "INFP": "Ethical AI advocate 🌿",
            "ENFP": "Creative product innovator 💡",
        }

        st.write(interpretations.get(user_type, "Future data wizard ✨"))

# -----------------------------
# Data Visualisations
# -----------------------------
with right:
    st.subheader("MBTI Distribution (Online Dataset) 📊")

    if go:
        type_counts = df["type"].value_counts()
        st.bar_chart(type_counts)

        st.markdown("---")

        user_type = compute_type()
        pct = (df["type"] == user_type).mean() * 100

        st.metric(
            label=f"People online with your type ({user_type})",
            value=f"{pct:.2f}%"
        )

        st.write("Are you rare… or very online? 😄")

    else:
        st.info("Your comparison appears after you reveal your type.")