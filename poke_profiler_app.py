# poke_profiler_app.py (No changes needed in this file)

import streamlit as st
import pandas as pd
from sqlalchemy import text
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
import os
import random
from streamlit_extras.keyboard_text import key_to_text # This line will now work

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Poké-Profiler 2.0", page_icon="🔮", layout="wide")

# --- DATA LOADING & CACHING ---
@st.cache_data
def load_pokemon_data():
    csv_path = "pokemon_data.csv"
    if os.path.exists(csv_path):
        return pd.read_csv(csv_path)
    else:
        st.error(f"Fatal Error: `{csv_path}` not found. Please run the data fetching script first.")
        st.stop()

# --- DATABASE SETUP ---
conn = st.connection("pokemon_db", type="sql", url="sqlite:///poke_profiler_v2.db", ttl=0)

def init_db(initial_df):
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

# --- MODEL TRAINING ---
@st.cache_data(ttl=3600)
def get_and_train_model():
    df = conn.query("SELECT * FROM pokemon_profiles;", ttl=0)
    if len(df) < 20: return None, None, None
    features = ['environment', 'battle_style', 'core_strength', 'personality']
    target = 'pokemon_name'
    X = df[features]; y = df[target]
    encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    X_encoded = pd.DataFrame(encoder.fit_transform(X))
    X_encoded.columns = encoder.get_feature_names_out(features)
    model = RandomForestClassifier(n_estimators=100, random_state=42); model.fit(X_encoded, y)
    return model, encoder, list(X_encoded.columns)

# --- UI COMPONENTS ---
def show_hall_of_fame():
    st.subheader("🏆 Hall of Fame")
    st.caption("The most frequent partners chosen by trainers worldwide.")
    profile_df = conn.query("SELECT pokemon_name FROM pokemon_profiles;", ttl=60)
    if profile_df.empty:
        st.info("The Hall of Fame is empty. Be the first to contribute!")
        return
    
    top_pokemon = profile_df['pokemon_name'].value_counts().nlargest(5)
    cols = st.columns(len(top_pokemon))
    for i, (name, count) in enumerate(top_pokemon.items()):
        with cols[i]:
            pokemon_info = POKEMON_INFO.get(name)
            if pokemon_info:
                st.image(pokemon_info["img_url"])
                st.metric(label=name, value=f"{count} Trainers")

def show_training_visualization():
    st.subheader("🧠 How The Profiler Learns")
    st.caption("This shows how many times each Pokémon has been confirmed as a partner by users.")
    profile_df = conn.query("SELECT pokemon_name FROM pokemon_profiles;", ttl=60)
    if not profile_df.empty:
        counts = profile_df['pokemon_name'].value_counts()
        st.bar_chart(counts)
    else:
        st.info("No user feedback has been submitted yet.")

# --- INITIALIZATION ---
pokemon_data_df = load_pokemon_data()
init_db(pokemon_data_df)
model, encoder, trained_columns = get_and_train_model()
POKEMON_INFO = pokemon_data_df.set_index('pokemon_name').to_dict('index')

# --- MAIN APP LAYOUT ---
st.title("Poké-Profiler 2.0 🔮")
st.markdown("Answer the call of the wild! Describe yourself to discover your true Pokémon partner.")

tab1, tab2 = st.tabs(["Profiler Quiz", "Hall of Fame"])

with tab1:
    if model is None:
        st.warning("The Profiler is still gathering data... Predictions will be available soon!")
    else:
        with st.form("profiler_form"):
            st.subheader("Tell us about yourself...")
            environment = st.selectbox("Which environment do you feel most at home in?", ('Forests & Jungles', 'Oceans & Lakes', 'Mountains & Caves', 'Cities & Plains', 'Mysterious Places'))
            personality = st.radio("Which best describes your personality?", ['Bold & Competitive', 'Calm & Loyal', 'Mysterious & Cunning', 'Energetic & Free-Spirited', 'Adaptable & Friendly'])
            core_strength = st.selectbox("What do you value most in a partner?", ('Raw Power', 'Resilience', 'Speed & Evasion', 'Versatility'))
            battle_style = st.radio("How do you approach challenges?", ['Physical & Head-on', 'Strategic & Long-Range', 'Quick & Agile', 'Balanced & Versatile'])
            destiny_checked = st.checkbox("Do you feel a touch of destiny?", help="Checking this may lead to a legendary encounter...")
            submitted = st.form_submit_button("Discover My Partner!")

        if submitted:
            input_data = pd.DataFrame([[environment, battle_style, core_strength, personality]], columns=['environment', 'battle_style', 'core_strength', 'personality'])
            input_encoded = pd.DataFrame(encoder.transform(input_data), columns=encoder.get_feature_names_out(['environment', 'battle_style', 'core_strength', 'personality']))
            input_processed = input_encoded.reindex(columns=trained_columns, fill_value=0)
            
            prediction = model.predict(input_processed)[0]
            is_legendary_encounter = False
            
            if destiny_checked and personality == 'Mysterious & Cunning' and core_strength == 'Raw Power':
                if random.randint(1, 20) == 1: # 5% chance
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
            
            st.write("---")
            app_url = "https://your-app-url.streamlit.app" # Replace with your actual app URL
            share_text = f"My perfect Pokémon partner is {prediction}! 🔮 Find yours at the Poké-Profiler: {app_url}"
            if st.button("💌 Share Your Result!"):
                key_to_text(share_text)
                st.toast("Copied to clipboard!")

            st.write("**Is this your perfect partner?** Your feedback helps the Profiler learn!")
            st.session_state.last_input = input_data.to_dict('records')[0]
            st.session_state.last_prediction = prediction
            feedback_cols = st.columns(3)
            if feedback_cols[0].button("✅ It's a perfect match!", use_container_width=True): st.session_state.feedback_given = True
            if feedback_cols[1].button("🤔 It's pretty close", use_container_width=True): st.session_state.feedback_given = True
            if feedback_cols[2].button("❌ Not quite right", use_container_width=True): st.session_state.feedback_given = True

with tab2:
    show_hall_of_fame()

with st.sidebar:
    st.header("Behind The Scenes")
    with st.expander("How The Profiler Learns", expanded=False):
        show_training_visualization()

if st.session_state.get('feedback_given', False):
    profile_data = st.session_state.last_input
    profile_data['pokemon_name'] = st.session_state.last_prediction
    with conn.session as s:
        s.execute(text("INSERT INTO pokemon_profiles (environment, battle_style, core_strength, personality, pokemon_name) VALUES (:environment, :battle_style, :core_strength, :personality, :pokemon_name);"), params=profile_data)
        s.commit()
    st.toast("Thank you! Your bond has strengthened the Profiler's knowledge!", icon="✨")
    st.cache_data.clear()
    del st.session_state.feedback_given
    del st.session_state.last_input
    del st.session_state.last_prediction
