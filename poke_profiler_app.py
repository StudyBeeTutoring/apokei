import streamlit as st
import pandas as pd
from sqlalchemy import text
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from lightgbm import LGBMClassifier
import os
import random
import joblib

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Poké-Profiler", page_icon="🔮", layout="centered")

# --- MODEL AND DATA FILENAMES ---
MODEL_PATH = "trained_pokemon_model.joblib"
DATA_PATH = "pokemon_data.csv"

# --- DATA & MODEL LOADING (Now much simpler) ---
@st.cache_resource
def load_model():
    """Loads the pre-trained model from disk."""
    if os.path.exists(MODEL_PATH):
        pipeline = joblib.load(MODEL_PATH)
        return pipeline
    else:
        st.error(f"Fatal Error: `{MODEL_PATH}` not found. Please pre-train the model and upload it.")
        st.stop()

@st.cache_data
def load_pokemon_data():
    """Loads the Pokémon data from the CSV file."""
    if os.path.exists(DATA_PATH):
        return pd.read_csv(DATA_PATH)
    else:
        st.error(f"Fatal Error: `{DATA_PATH}` not found. Please create it first.")
        st.stop()

# --- DATABASE SETUP ---
conn = st.connection("pokemon_db", type="sql", url="sqlite:///poke_profiler_v3.db", ttl=0)

def init_db(initial_df):
    """Initializes the DB and seeds it if empty."""
    with conn.session as s:
        s.execute(text("""
            CREATE TABLE IF NOT EXISTS pokemon_profiles (
                id INTEGER PRIMARY KEY, environment TEXT, battle_style TEXT,
                core_strength TEXT, personality TEXT, pokemon_name TEXT
            );
        """))
        count = s.execute(text("SELECT COUNT(*) FROM pokemon_profiles;")).scalar()
        if count == 0 and not initial_df.empty:
            seed_df = initial_df[['environment', 'battle_style', 'core_strength', 'personality', 'pokemon_name']]
            seed_df.to_sql('pokemon_profiles', conn.engine, if_exists='append', index=False)
        s.commit()

# --- INITIALIZATION ---
pipeline = load_model()
pokemon_data_df = load_pokemon_data()
init_db(pokemon_data_df)
POKEMON_INFO = pokemon_data_df.set_index('pokemon_name').to_dict('index')

# --- MAIN APP LAYOUT ---
st.title("Poké-Profiler 🔮")
st.markdown("Answer the call of the wild! Describe yourself to discover your true Pokémon partner.")
st.write("---")

# The main form and logic can now assume the pipeline is always loaded and ready.
with st.form("profiler_form"):
    st.subheader("Tell us about yourself...")
    environment = st.selectbox("Which environment do you feel most at home in?", ('Forests & Jungles', 'Oceans & Lakes', 'Mountains & Caves', 'Cities & Plains', 'Mysterious Places'))
    personality = st.radio("Which best describes your personality?", ['Bold & Competitive', 'Calm & Loyal', 'Mysterious & Cunning', 'Energetic & Free-Spirited', 'Adaptable & Friendly'])
    core_strength = st.selectbox("What do you value most in a partner?", ('Raw Power', 'Resilience', 'Speed & Evasion', 'Versatility'))
    battle_style = st.radio("How do you approach challenges?", ['Physical & Head-on', 'Strategic & Long-Range', 'Quick & Agile', 'Balanced & Versatile'])
    destiny_checked = st.checkbox("Do you feel a touch of destiny?", help="Checking this may lead to a legendary encounter...")
    submitted = st.form_submit_button("Discover My Partner!")

if submitted:
    # --- PREDICTION LOGIC ---
    input_data = pd.DataFrame([[environment, battle_style, core_strength, personality]],
                              columns=['environment', 'battle_style', 'core_strength', 'personality'])
    
    prediction = pipeline.predict(input_data)[0]
    prediction_proba = pipeline.predict_proba(input_data)
    confidence = prediction_proba.max() * 100

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
    
    # --- DISPLAY PREDICTION ---
    if is_legendary_encounter: st.success("A legendary force answers your call...", icon="🌟")
    if is_shiny: st.success("Whoa! A rare Shiny partner appeared!", icon="✨"); st.balloons()
    
    st.subheader("Your Pokémon Partner is...")
    col1, col2 = st.columns([1, 2])
    with col1:
        st.image(img_to_display, width=200)
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
    
    # --- FEEDBACK LOOP ---
    st.write("---")
    st.write("**Is this your perfect partner?** Your feedback helps the Profiler get smarter!")
    st.session_state.last_input = input_data.to_dict('records')[0]
    st.session_state.last_prediction = prediction
    feedback_cols = st.columns(3)
    if feedback_cols[0].button("✅ It's a perfect match!", use_container_width=True): st.session_state.feedback_given = True
    if feedback_cols[1].button("🤔 It's pretty close", use_container_width=True): st.session_state.feedback_given = True
    if feedback_cols[2].button("❌ Not quite right", use_container_width=True): st.session_state.feedback_given = True

# --- LOGGING FEEDBACK TO DB ---
if st.session_state.get('feedback_given', False):
    profile_data = st.session_state.last_input
    profile_data['pokemon_name'] = st.session_state.last_prediction
    with conn.session as s:
        s.execute(
            text("INSERT INTO pokemon_profiles (environment, battle_style, core_strength, personality, pokemon_name) VALUES (:environment, :battle_style, :core_strength, :personality, :pokemon_name);"),
            params=profile_data
        )
        s.commit()
    
    st.toast("Thank you! The Profiler is now learning from your feedback.", icon="✨")
    
    # For now, we will just log the data. The model can be retrained offline by the developer.
    # This prevents long retraining delays for users.
    
    del st.session_state.feedback_given
    del st.session_state.last_input
    del st.session_state.last_prediction
    st.rerun()
