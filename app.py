import streamlit as st
import numpy as np
from model import load_and_train_model

st.set_page_config(
    page_title="Football Match Predictor",
    layout="centered"
)

@st.cache_resource
def load_model():
    return load_and_train_model("merged_seasons_12_files.csv")

model, team_stats = load_model()

st.title("⚽ Football Match Predictor")
st.caption(
    "Predictions based on historical form and Elo ratings. "
    "For analysis only — not betting advice."
)

teams = sorted(team_stats.index.tolist())

home_team = st.selectbox("Home Team", teams)
away_team = st.selectbox("Away Team", teams)

if home_team == away_team:
    st.warning("Please select two different teams.")
else:
    if st.button("Predict"):
        home = team_stats.loc[home_team]
        away = team_stats.loc[away_team]

        X_input = np.array([[
            home["GF_avg"], home["GA_avg"], home["Pts_avg"],
            away["GF_avg"], away["GA_avg"], away["Pts_avg"],
            home["Elo"], away["Elo"]
        ]])

        probabilities = model.predict_proba(X_input)[0]
        classes = model.classes_

        results = dict(zip(classes, probabilities))

        st.subheader("Prediction Probabilities")
        st.metric("🏠 Home Win", f"{results['H']*100:.1f}%")
        st.metric("🤝 Draw", f"{results['D']*100:.1f}%")
        st.metric("🚗 Away Win", f"{results['A']*100:.1f}%")
