import streamlit as st
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
import os
import random
import joblib
import gspread
from google.oauth2.service_account import Credentials
import json

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Poké-Profiler", page_icon="🔮", layout="centered")

# --- MODEL AND DATA FILENAMES ---
MODEL_PATH = "trained_pokemon_model.joblib"
DATA_PATH = "pokemon_data.csv"

# --- GOOGLE SHEETS CONNECTION (MANUAL, ROBUST METHOD) ---
@st.cache_resource
def connect_to_gsheets():
    """Establishes a connection to Google Sheets using gspread and st.secrets."""
    try:
        scopes = ["https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/drive"]
        creds = Credentials.from_service_account_info(st.secrets, scopes=scopes)
        client = gspread.authorize(creds)
        return client
    except Exception as e:
        st.error(f"Failed to connect to Google Sheets. Check your secrets. Details: {e}")
        st.stop()

def log_feedback_to_sheet(feedback_data):
    """Logs a new row of feedback data to the specified Google Sheet."""
    try:
        client = connect_to_gsheets()
        # IMPORTANT: Replace "PokeProfilerFeedback" with the exact name of your Google Sheet.
        sheet = client.open("PokeProfilerFeedback").sheet1
        sheet.append_row(list(feedback_data.values()))
        return True
    except Exception as e:
        st.error(f"Failed to write to Google Sheet. Details: {e}")
        return False

# --- DATA & MODEL LOADING ---
@st.cache_resource
def load_model():
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    else:
        st.error(f"Fatal Error: `{MODEL_PATH}` not found. Please pre-train and upload the model."); st.stop()

@st.cache_data
def load_pokemon_data():
    if os.path.exists(DATA_PATH):
        return pd.read_csv(DATA_PATH)
    else:
        st.error(f"Fatal Error: `{DATA_PATH}` not found."); st.stop()

# --- INITIALIZATION ---
pipeline = load_model()
pokemon_data_df = load_pokemon_data()
POKEMON_INFO = pokemon_data_df.set_index('pokemon_name').to_dict('index')

# --- MAIN APP LAYOUT ---
st.title("Poké-Profiler 🔮")
st.markdown("Answer the call of the wild! Describe yourself to discover your true Pokémon partner.")
st.write("---")

with st.form("profiler_form"):
    st.subheader("Tell us about yourself...")
    environment = st.selectbox("Which environment do you feel most at home in?", ('Forests & Jungles', 'Oceans & Lakes', 'Mountains & Caves', 'Cities & Plains', 'Mysterious Places'))
    personality = st.radio("Which best describes your personality?", ['Bold & Competitive', 'Calm & Loyal', 'Mysterious & Cunning', 'Energetic & Free-Spirited', 'Adaptable & Friendly'])
    core_strength = st.selectbox("What do you value most in a partner?", ('Raw Power', 'Resilience', 'Speed & Evasion', 'Versatility'))
    battle_style = st.radio("How do you approach challenges?", ['Physical & Head-on', 'Strategic & Long-Range', 'Quick & Agile', 'Balanced & Versatile'])
    destiny_checked = st.checkbox("Do you feel a touch of destiny?", help="Checking this may lead to a legendary encounter...")
    submitted = st.form_submit_button("Discover My Partner!")

if submitted:
    input_data = pd.DataFrame([[environment, battle_style, core_strength, personality]],
                              columns=['environment', 'battle_style', 'core_strength', 'personality'])
    
    prediction = pipeline.predict(input_data)[0]
    is_legendary_encounter = False
    if destiny_checked and personality == 'Mysterious & Cunning' and core_strength == 'Raw Power':
        if random.randint(1, 20) == 1:
            legendary_pool = pokemon_data_df[pokemon_data_df['is_legendary'] | pokemon_data_df['is_mythical']]
            if not legendary_pool.empty:
                prediction = legendary_pool.sample(n=1)['pokemon_name'].iloc[0]
                is_legendary_encounter = True

    is_shiny = (random.randint(1, 100) == 1)
    pokemon_info = POKEMON_INFO.get(prediction)
    img_to_display = pokemon_info['shiny_img_url'] if is_shiny else pokemon_info['img_url']
    
    if is_legendary_encounter: st.success("A legendary force answers your call...", icon="🌟")
    if is_shiny: st.success("Whoa! A rare Shiny partner appeared!", icon="✨"); st.balloons()
    
    st.subheader("Your Pokémon Partner is...")
    col1, col2 = st.columns([1, 2])
    with col1: st.image(img_to_display, width=200)
    with col2:
        title = f"{prediction} ✨" if is_shiny else prediction
        st.markdown(f"## {title}")
        st.markdown(f"**Type:** `{pokemon_info['type1'].capitalize()}`")
        st.write("**Base Stats:**")
        stat_cols = st.columns(3)
        stat_cols[0].metric("HP", pokemon_info['hp'])
        stat_cols[1].metric("Attack", pokemon_info['attack'])
        stat_cols[2].metric("Defense", pokemon_info['defense'])
    st.info(f"**Pokédex Entry:** *{pokemon_info['pokedex_entry']}*")
    
    st.write("---")
    st.write("**Is this your perfect partner?** Your feedback helps the Profiler get smarter!")
    st.session_state.last_input = input_data.to_dict('records')[0]
    st.session_state.last_prediction = prediction
    feedback_cols = st.columns(3)
    if feedback_cols[0].button("✅ It's a perfect match!", use_container_width=True): st.session_state.feedback_given = True
    if feedback_cols[1].button("🤔 It's pretty close", use_container_width=True): st.session_state.feedback_given = True
    if feedback_cols[2].button("❌ Not quite right", use_container_width=True): st.session_state.feedback_given = True

# --- LOGGING FEEDBACK TO GOOGLE SHEETS ---
if st.session_state.get('feedback_given', False):
    profile_data = st.session_state.last_input[0]
    profile_data['pokemon_name'] = st.session_state.last_prediction
    
    if log_feedback_to_sheet(profile_data):
        st.toast("Thank you! The Profiler is now learning from your feedback.", icon="✨")
    
    del st.session_state.feedback_given
    del st.session_state.last_input
    del st.session_state.last_prediction
    st.rerun()
