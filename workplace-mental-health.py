import streamlit as st
import pandas as pd

st.set_page_config(page_title="Workplace Wellbeing: Mini Check-in", layout="wide")

# -----------------------------
# Helpers
# -----------------------------
FREQ_ORDER = [
    "Never",
    "A few times per year",
    "Once a month",
    "A few times per month",
    "Once a week",
    "A few times per week",
    "Every day",
]

WB_ORDER = [
    "At no time",
    "Some of the time",
    "Less than half of the time",
    "More than half of the time",
    "Most of the time",
    "All of the time",
]

def pct(cond: pd.Series) -> float:
    return float(cond.mean() * 100)

def safe_cat(series: pd.Series, order: list[str]) -> pd.Categorical:
    # Make a categorical with a fixed order, keeping unknown values at end
    existing = [v for v in order if v in set(series.dropna().unique())]
    return pd.Categorical(series, categories=existing, ordered=True)

def label_from_scores(burnout_score: int, wellbeing_score: int, support: str) -> tuple[str, str]:
    # Simple, friendly label mapping
    if burnout_score >= 5 and wellbeing_score <= 2:
        return ("Redline Mode 🔥", "High strain + low wellbeing. Worth prioritising supports + boundaries.")
    if burnout_score >= 5 and wellbeing_score >= 3:
        return ("Grinding but Coping 🏃‍♂️", "High strain, but wellbeing signals are holding up. Watch sustainability.")
    if burnout_score <= 2 and wellbeing_score >= 4:
        return ("Cruise Control 😎", "Low strain + strong wellbeing. Great foundation for long-term growth.")
    if burnout_score <= 2 and wellbeing_score <= 2:
        return ("Quiet Drift 🌫️", "Low strain but low wellbeing. Might be missing connection, meaning, or support.")
    return ("Steady State 🌿", "Mixed signals (which is normal). The goal is trend + habits, not perfection.")

# -----------------------------
# Load Data
# -----------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("stress_data.csv")
    return df

df = load_data()

# Columns we rely on
REQUIRED = {"Industry", "WorkLocation", "AccessMH", "EE3", "WB3"}
missing = REQUIRED - set(df.columns)
if missing:
    st.error(f"Dataset missing required columns: {sorted(missing)}")
    st.stop()

# -----------------------------
# Header
# -----------------------------
st.title("Workplace Mental Health: Mini Check-in ➜ Then the Data 📊")
st.caption("A quick interactive demo for orientation week. Not medical advice — just a conversation starter.")

st.markdown("---")

# -----------------------------
# Mini Questionnaire (sidebar)
# -----------------------------
st.sidebar.header("Mini Check-in (30 seconds) 🧩")
st.sidebar.write("Answer as a *future-you* in a first job/internship.")

age = st.sidebar.selectbox("Age bracket (just for demo)", sorted([x for x in df["Age"].dropna().unique()]))
industry = st.sidebar.selectbox("Preferred industry", sorted([x for x in df["Industry"].dropna().unique()]))
work_location = st.sidebar.selectbox("Preferred work setup", sorted([x for x in df["WorkLocation"].dropna().unique()]))

exhaustion = st.sidebar.select_slider("How often do you feel emotionally exhausted?", options=FREQ_ORDER, value="Once a week")
wellbeing = st.sidebar.select_slider("How often do you feel you’re doing well overall?", options=WB_ORDER, value="Most of the time")
support = st.sidebar.radio("Would you want mental health support available at work?", ["Yes", "No"], horizontal=True)

# Convert to simple scores for a fun profile
burnout_score = FREQ_ORDER.index(exhaustion)  # 0..6
wellbeing_score = WB_ORDER.index(wellbeing)   # 0..5
profile_title, profile_blurb = label_from_scores(burnout_score, wellbeing_score, support)

st.sidebar.markdown("---")
go = st.sidebar.button("Reveal the data 🚀", use_container_width=True)

# -----------------------------
# Main Layout
# -----------------------------
left, right = st.columns([1, 1])

with left:
    st.subheader("Your mini “profile” 🎭")
    st.markdown(f"### {profile_title}")
    st.write(profile_blurb)

    st.write("")
    st.write("Your inputs (demo):")
    st.write(f"• Industry: **{industry}**")
    st.write(f"• Work setup: **{work_location}**")
    st.write(f"• Exhaustion: **{exhaustion}**")
    st.write(f"• Wellbeing: **{wellbeing}**")
    st.write(f"• Support wanted: **{support}**")


# -----------------------------
# “Boom” section
# -----------------------------
with right:
    st.subheader("Boom! What the dataset says 💥")

    if not go:
        st.info("Answer the mini check-in on the left, then press **Reveal the data 🚀**.")
    else:
        # Filters
        subgroup = df.copy()
        subgroup = subgroup[subgroup["Industry"].eq(industry)]
        subgroup = subgroup[subgroup["WorkLocation"].eq(work_location)]

        use_subgroup = len(subgroup) >= 50
        view = subgroup if use_subgroup else df
        scope_label = f"{industry} + {work_location}" if use_subgroup else "All respondents"

        daily_burnout = pct(view["EE3"].eq("Every day"))
        high_wellbeing = pct(view["WB3"].eq("All of the time"))
        access_support = pct(view["AccessMH"].eq("Yes"))

        c1, c2, c3 = st.columns(3)
        c1.metric("Daily exhaustion", f"{daily_burnout:.1f}%")
        c2.metric("Peak wellbeing", f"{high_wellbeing:.1f}%")
        c3.metric("Workplace support access", f"{access_support:.1f}%")

        st.caption(f"Scope: {scope_label} | n = {len(view):,}")

        st.markdown("---")

        # Plot 1
        st.write("Emotional exhaustion distribution")
        ee_counts = view["EE3"].value_counts().reindex(
            [x for x in FREQ_ORDER if x in view["EE3"].unique()]
        )
        st.bar_chart(ee_counts)

        # Plot 2
        st.write("Wellbeing distribution")
        wb_counts = view["WB3"].value_counts().reindex(
            [x for x in WB_ORDER if x in view["WB3"].unique()]
        )
        st.bar_chart(wb_counts)

        # Plot 3
        st.write("Support access by industry")
        industry_support = (
            df.groupby("Industry")["AccessMH"]
            .apply(lambda s: (s == "Yes").mean() * 100)
            .sort_values(ascending=False)
        )
        st.bar_chart(industry_support)

