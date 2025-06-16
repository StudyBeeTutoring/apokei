import streamlit as st
import pandas as pd
from sqlalchemy import text
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from lightgbm import LGBMClassifier
import os
import random
import joblib
from streamlit_gsheets.connection import GSheetsConnection

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Poké-Profiler Pro", page_icon="🌟", layout="centered")

# --- MODEL AND DATA FILENAMES (Updated) ---
MODEL_PATH = "trained_pokemon_model_v2.joblib"
DATA_PATH = "pokemon_data_v2.csv"

# --- GOOGLE SHEETS CONNECTION ---
conn_gsheets = st.connection("gsheets", type=GSheetsConnection)

# --- DATA & MODEL LOADING ---
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.error(f"Fatal Error: `{MODEL_PATH}` not found. Please pre-train the model."); st.stop()
    return joblib.load(MODEL_PATH)

@st.cache_data
def load_pokemon_data():
    if not os.path.exists(DATA_PATH):
        st.error(f"Fatal Error: `{DATA_PATH}` not found."); st.stop()
    return pd.read_csv(DATA_PATH)

# --- INITIALIZATION ---
pipeline = load_model()
pokemon_data_df = load_pokemon_data()
POKEMON_INFO = pokemon_data_df.set_index('pokemon_name').to_dict('index')

# --- MAIN APP LAYOUT ---
st.title("Poké-Profiler Pro 🌟")
st.markdown("Discover your true Pokémon partner through our enhanced personality matrix.")
st.write("---")

with st.form("profiler_form"):
    st.subheader("Tell us about your inner trainer...")
    
    # --- UPGRADED QUIZ ---
    environment = st.selectbox(
        "Which of these places calls to you?",
        ('Verdant Forests', 'Serene Waters', 'Volcanic Mountains', 'Training Dojos', 'Bustling Cities', 'Ancient Ruins', 'Mythical Skies'))
    
    personality = st.radio(
        "How do others see you?",
        ['Passionate & Fierce', 'Gentle & Protective', 'Intelligent & Mysterious', 'Playful & Energetic', 'Stoic & Loyal'])
        
    battle_style = st.selectbox(
        "What is your preferred method of overcoming obstacles?",
        ('Aggressive Brawler', 'Glass Cannon Mage', 'Immovable Wall', 'Swift Scout', 'Methodical Warrior', 'Versatile All-Rounder'))
        
    core_strength = st.radio(
        "What is your greatest strength?",
        ('Raw Power', 'Resilience', 'Speed & Evasion', 'Versatility'))

    # New Question
    social_style = st.radio(
        "Do you prefer working with a team or on your own?",
        ['Team Player', 'Lone Wolf', 'Flexible'])
            
    destiny_checked = st.checkbox("Do you feel a touch of destiny?", help="This may attract a Legendary Pokémon...")
    submitted = st.form_submit_button("Reveal My Partner!")

if submitted:
    # --- PREDICTION LOGIC ---
    input_data = pd.DataFrame([[environment, battle_style, core_strength, personality, social_style]],
                              columns=['environment', 'battle_style', 'core_strength', 'personality', 'social_style'])
    
    prediction = pipeline.predict(input_data)[0]
    is_legendary_encounter = False
    
    # Legendary logic now includes social style
    if destiny_checked and personality == 'Intelligent & Mysterious' and social_style == 'Lone Wolf':
        if random.randint(1, 10) == 1: # 10% chance
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
    st.write("**Was this your perfect partner?** Your feedback helps the Profiler get even better!")
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
    feedback_df = pd.DataFrame([profile_data])
    conn_gsheets.update(worksheet="PokeProfilerFeedback", data=feedback_df)
    st.toast("Thank you! The Profiler is now learning from your feedback.", icon="✨")
    
    del st.session_state.feedback_given
    del st.session_state.last_input
    del st.session_state.last_prediction
    st.rerun()
