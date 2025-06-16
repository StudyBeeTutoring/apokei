import streamlit as st
import pandas as pd
from sqlalchemy import text
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
import os

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Poké-Profiler",
    page_icon="🔮",
    layout="centered"
)

# --- DATA LOADING & CACHING ---
@st.cache_data
def load_pokemon_data():
    """Loads the Pokémon data from the pre-processed CSV file."""
    if os.path.exists("pokemon_data.csv"):
        return pd.read_csv("pokemon_data.csv")
    else:
        st.error("Fatal Error: `pokemon_data.csv` not found. Please run the `fetch_poke_data.py` script first.")
        return pd.DataFrame()

# --- DATABASE SETUP ---
conn = st.connection("pokemon_db", type="sql", url="sqlite:///poke_profiler.db", ttl=0)

def init_db(initial_df):
    """Initializes the Pokémon profiles table and seeds it from the CSV if empty."""
    with conn.session as s:
        s.execute(text("""
            CREATE TABLE IF NOT EXISTS pokemon_profiles (
                id INTEGER PRIMARY KEY,
                environment TEXT,
                battle_style TEXT,
                core_strength TEXT,
                personality TEXT,
                pokemon_name TEXT
            );
        """))
        count = s.execute(text("SELECT COUNT(*) FROM pokemon_profiles;")).scalar()
        if count == 0 and not initial_df.empty:
            seed_df = initial_df[['environment', 'battle_style', 'core_strength', 'personality', 'pokemon_name']]
            seed_df.to_sql('pokemon_profiles', conn.engine, if_exists='append', index=False)
        s.commit()

# --- MODEL TRAINING ---
@st.cache_data(ttl=3600)
def get_and_train_model():
    """Loads data from the DB, trains a model, and returns it."""
    df = conn.query("SELECT * FROM pokemon_profiles;", ttl=0)
    if len(df) < 20: # Ensure we have a decent amount of data
        return None, None, None

    features = ['environment', 'battle_style', 'core_strength', 'personality']
    target = 'pokemon_name'
    X = df[features]
    y = df[target]

    encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    X_encoded = pd.DataFrame(encoder.fit_transform(X))
    X_encoded.columns = encoder.get_feature_names_out(features)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_encoded, y)

    return model, encoder, list(X_encoded.columns)

# --- UI & APP LOGIC ---
st.title("Poké-Profiler 🔮")
st.markdown("Answer the call of the wild! Describe yourself to discover your true Pokémon partner.")

# --- INITIALIZATION ---
pokemon_data_df = load_pokemon_data()
init_db(pokemon_data_df)
model, encoder, trained_columns = get_and_train_model()

# Create a dictionary for easy info lookup from the CSV data
POKEMON_INFO = pokemon_data_df.set_index('pokemon_name').to_dict('index')

if model is None:
    st.warning("The Profiler is still gathering data from trainers around the world... Predictions will be available soon!")
else:
    # --- PROFILER QUIZ FORM ---
    with st.form("profiler_form"):
        st.subheader("Tell us about yourself...")
        
        environment = st.selectbox(
            "Which environment do you feel most at home in?",
            ('Forests & Jungles', 'Oceans & Lakes', 'Mountains & Caves', 'Cities & Plains', 'Mysterious Places'))
        
        personality = st.radio(
            "Which best describes your personality?",
            ['Bold & Competitive', 'Calm & Loyal', 'Mysterious & Cunning', 'Energetic & Free-Spirited', 'Adaptable & Friendly'], horizontal=True)
            
        core_strength = st.selectbox(
            "What do you value most in a partner?",
            ('Raw Power', 'Resilience', 'Speed & Evasion', 'Versatility'))
            
        battle_style = st.radio(
            "How do you approach challenges?",
            ['Physical & Head-on', 'Strategic & Long-Range', 'Quick & Agile', 'Balanced & Versatile'], horizontal=True)

        submitted = st.form_submit_button("Discover My Partner!")

    if submitted:
        # --- PREDICTION LOGIC ---
        input_data = pd.DataFrame([[environment, battle_style, core_strength, personality]],
                                  columns=['environment', 'battle_style', 'core_strength', 'personality'])
        
        input_encoded = pd.DataFrame(encoder.transform(input_data))
        input_encoded.columns = encoder.get_feature_names_out(['environment', 'battle_style', 'core_strength', 'personality'])
        
        input_processed = input_encoded.reindex(columns=trained_columns, fill_value=0)

        prediction = model.predict(input_processed)[0]
        prediction_proba = model.predict_proba(input_processed)
        confidence = prediction_proba.max() * 100

        # --- DISPLAY PREDICTION ---
        pokemon_info = POKEMON_INFO.get(prediction, {"img_url": "", "type1": "normal"})
        
        st.subheader("Your Pokémon Partner is...")
        col1, col2 = st.columns([1, 2])
        with col1:
            st.image(pokemon_info["img_url"], width=200)
        with col2:
            st.markdown(f"## {prediction}")
            st.markdown(f"**Type:** `{pokemon_info['type1'].capitalize()}`")
            st.progress(confidence / 100)
            st.caption(f"Match Confidence: {confidence:.1f}%")
        
        # --- FEEDBACK LOOP ---
        st.write("---")
        st.write("**Is this your perfect partner?** Your feedback helps the Profiler learn and grow!")
        
        st.session_state.last_input = input_data.to_dict('records')[0]
        st.session_state.last_prediction = prediction

        feedback_cols = st.columns(3)
        if feedback_cols[0].button("✅ It's a perfect match!", use_container_width=True):
            st.session_state.feedback_given = True
        if feedback_cols[1].button("🤔 It's pretty close", use_container_width=True):
            st.session_state.feedback_given = True
        if feedback_cols[2].button("❌ Not quite right", use_container_width=True):
            st.session_state.feedback_given = True

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
    
    st.toast("Thank you! Your bond has strengthened the Profiler's knowledge!", icon="✨")
    
    st.cache_data.clear() # Clear cache to force model retraining
    
    del st.session_state.feedback_given
    del st.session_state.last_input
    del st.session_state.last_prediction
