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
        creds = Credentials.from_service_account_info(st.secrets["gcp_service_account"], scopes=scopes)
        client = gspread.authorize(creds)
        return client
    except Exception as e:
        st.error(f"Failed to connect to Google Sheets. Check secrets. Details: {e}")
        st.stop()

def log_feedback_to_sheet(feedback_data):
    # --- IMPORTANT ---
    # PASTE THE FULL URL OF YOUR GOOGLE SHEET HERE.
    # This is the robust way to connect and prevents ambiguity.
    SHEET_URL = "YOUR_GOOGLE_SHEET_URL_HERE" 

    try:
        client = connect_to_gsheets()
        
        # Open the sheet by its specific URL
        spreadsheet = client.open_by_url(SHEET_URL)
        sheet = spreadsheet.sheet1
        
        # Convert all values to simple strings before sending to prevent format errors
        values_to_append = [str(v) for v in feedback_data.values()]
        
        # Append the new row
        sheet.append_row(values_to_append)
        
        return True
        
    except Exception as e:
        # Provide a helpful error message to the user if something goes wrong
        st.error(f"Failed to write to Google Sheet. Please try again later. Details: {e}")
        return False

def process_feedback(feedback_text):
    """
    Callback function to log feedback and update session state.
    This function is executed when a feedback button is clicked, before the script reruns.
    """
    profile_data_to_log = st.session_state.last_input[0].copy()
    profile_data_to_log['pokemon_name'] = st.session_state.prediction_details['name']
    profile_data_to_log['feedback'] = feedback_text

    if log_feedback_to_sheet(profile_data_to_log):
        st.session_state.show_thank_you = True
        del st.session_state.prediction_details

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
    Displays the prediction results and handles feedback using on_click callbacks.
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
    
    feedback_cols[0].button(
        "✅ It's a perfect match!",
        on_click=process_feedback,
        args=("Perfect Match",),
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

if 'prediction_details' in st.session_state:
    display_prediction()
elif 'show_thank_you' in st.session_state:
    display_thank_you()
else:
    display_quiz()
