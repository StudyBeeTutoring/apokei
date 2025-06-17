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
import time

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Poké-Profiler", page_icon="🔮", layout="centered")

# --- MODEL AND DATA FILENAMES ---
MODEL_PATH = "trained_pokemon_model.joblib"
DATA_PATH = "pokemon_data.csv"

# --- GOOGLE SHEETS CONNECTION ---
@st.cache_resource
def connect_to_gsheets():
    try:
        scopes = ["https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/drive"]
        creds = Credentials.from_service_account_info(st.secrets, scopes=scopes)
        client = gspread.authorize(creds)
        return client
    except Exception as e:
        st.error(f"Failed to connect to Google Sheets. Check secrets. Details: {e}")
        st.stop()

def log_feedback_to_sheet(feedback_data):
    try:
        client = connect_to_gsheets()
        # IMPORTANT: Replace "PokeProfilerFeedback" with the exact name of your Google Sheet.
        sheet = client.open("PokeProfilerFeedback").sheet1
        # Ensure the headers in your Google Sheet match the keys in feedback_data
        # For example: environment, battle_style, core_strength, personality, pokemon_name, feedback
        if sheet.row_count == 0:
            sheet.append_row(list(feedback_data.keys())) # Add header row if sheet is empty
        sheet.append_row(list(feedback_data.values()))
        return True
    except Exception as e:
        st.error(f"Failed to write to Google Sheet. Details: {e}")
        return False

# --- NEW: FEEDBACK CALLBACK FUNCTION ---
def process_feedback(feedback_text):
    """
    Callback function to log feedback and update session state.
    This function is executed when a feedback button is clicked, before the script reruns.
    """
    # 1. Prepare the data for logging by copying it from session state
    profile_data_to_log = st.session_state.last_input[0].copy()
    profile_data_to_log['pokemon_name'] = st.session_state.prediction_details['name']
    profile_data_to_log['feedback'] = feedback_text # Log which button was clicked!

    # 2. Log the data to the Google Sheet
    if log_feedback_to_sheet(profile_data_to_log):
        # 3. If logging is successful, update the session state to show the thank you message
        st.session_state.show_thank_you = True
        # Clean up the state so the prediction screen doesn't show anymore
        del st.session_state.prediction_details
    # No st.rerun() is needed. Streamlit automatically reruns after a callback.

# --- DATA & MODEL LOADING ---
@st.cache_resource
def load_model():
    if os.path.exists(MODEL_PATH): return joblib.load(MODEL_PATH)
    else: st.error(f"Fatal Error: `{MODEL_PATH}` not found."); st.stop()

@st.cache_data
def load_pokemon_data():
    if os.path.exists(DATA_PATH): return pd.read_csv(DATA_PATH)
    else: st.error(f"Fatal Error: `{DATA_PATH}` not found."); st.stop()

# --- INITIALIZATION ---
pipeline = load_model()
pokemon_data_df = load_pokemon_data()
POKEMON_INFO = pokemon_data_df.set_index('pokemon_name').to_dict('index')

# --- UI COMPONENTS ---
def display_prediction():
    """
    A dedicated function to display the prediction results and handle feedback.
    This version uses on_click callbacks for robust state management.
    """
    prediction_details = st.session_state.prediction_details
    prediction = prediction_details['name']
    is_legendary = prediction_details['is_legendary']
    is_shiny = prediction_details['is_shiny']
    
    if is_legendary: st.success("A legendary force answers your call...", icon="🌟")
    if is_shiny: st.success("Whoa! A rare Shiny partner appeared!", icon="✨"); st.balloons()
    
    st.subheader("Your Pokémon Partner is...")
    pokemon_info = POKEMON_INFO.get(prediction)
    img_to_display = pokemon_info['shiny_img_url'] if is_shiny else pokemon_info['img_url']
    
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
        
    feedback_cols = st.columns(3)
    
    # Use on_click to call our new callback function.
    # Use 'args' to pass the specific feedback text to the function.
    feedback_cols[0].button(
        "✅ It's a perfect match!",
        on_click=process_feedback,
        args=("Perfect Match",), # Note the comma to make it a tuple
        use_container_width=True
    )
    feedback_cols[1].button(
        "🤔 It's pretty close",
        on_click=process_feedback,
        args=("Pretty Close",),
        use_container_width=True
    )
    feedback_cols[2].button(
        "❌ Not quite right",
        on_click=process_feedback,
        args=("Not Right",),
        use_container_width=True
    )


def display_thank_you():
    """Displays a confirmation screen after feedback is submitted."""
    st.success("Thank you! Your feedback has been recorded and the Profiler is now learning from your insights.", icon="✨")
    st.balloons()
    if st.button("Take the Quiz Again!", use_container_width=True):
        # To restart, just remove the flag that shows the thank you message.
        # This will trigger a rerun and the router logic will show the quiz.
        del st.session_state.show_thank_you
        st.rerun()

def display_quiz():
    """Displays the main quiz form."""
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

        # Store prediction details and user input in the session state
        st.session_state.prediction_details = {
            "name": prediction,
            "is_legendary": is_legendary_encounter,
            "is_shiny": (random.randint(1, 100) == 1),
        }
        st.session_state.last_input = input_data.to_dict('records')
        st.rerun()

# --- MAIN APP ROUTER ---
st.title("Poké-Profiler 🔮")
st.markdown("Answer the call of the wild! Describe yourself to discover your true Pokémon partner.")
st.write("---")

# This logic now works perfectly because the session state is managed correctly by the callbacks.
if 'prediction_details' in st.session_state:
    display_prediction()
elif 'show_thank_you' in st.session_state:
    display_thank_you()
else:
    display_quiz()
