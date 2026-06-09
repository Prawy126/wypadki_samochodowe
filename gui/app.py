import sys
from pathlib import Path

# Add project root to sys.path to allow relative imports under streamlit execution
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

import streamlit as st
import pandas as pd
import numpy as np
import json
import joblib
import torch
import os
from datetime import datetime, time

# Set page layout to wide
st.set_page_config(
    page_title="Kalkulator Ciężkości Wypadków Drogowych",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Import predictor & config from training module
from uczenie.predictor import AccidentPredictor, LGBMPredictor
from uczenie import config
from preprocessing.preprocessing import (
    WIND_DIR_MAP,
    WEATHER_MAP,
    haversine_np,
    US_CENTER_LAT,
    US_CENTER_LNG,
    BOOL_COLS,
    TWILIGHT_COLS
)

# Polish translations for severity levels
POLISH_LABELS = {
    0: "Stopień I (Łagodny / Severity 1)",
    1: "Stopień II (Umiarkowany / Severity 2)",
    2: "Stopień III (Poważny / Severity 3)",
    3: "Stopień IV (Krytyczny / Severity 4)"
}

# Polish description of severity levels
POLISH_DESCS = {
    0: "Wypadek o minimalnym wpływie na ruch drogowy. Krótkie opóźnienia, brak znaczących zniszczeń infrastruktury lub ciężkich obrażeń.",
    1: "Wypadek o umiarkowanym wpływie. Częściowe zablokowanie pasów ruchu, możliwe kolizje i lekkie obrażenia uczestników.",
    2: "Poważne zdarzenie drogowe. Zablokowanie głównych pasów, interwencja służb ratunkowych, znaczne opóźnienia, możliwe cięższe obrażenia.",
    3: "Krytyczny wypadek drogowy. Całkowite zablokowanie drogi/autostrady, wielogodzinny paraliż, poważne ofiary w ludziach."
}

# Pre-defined scenarios to make testing quick and fun
PRESETS = {
    "Wybierz scenariusz...": None,
    "☀️ Standardowa podróż autostradą (Niskie ryzyko)": {
        "Start_Time": datetime(2023, 6, 15, 12, 0),
        "Duration_min": 30.0,
        "Distance(mi)": 0.5,
        "Start_Lat": 37.7749,
        "Start_Lng": -122.4194,
        "State": "CA",
        "City": "San Francisco",
        "County": "San Francisco",
        "Timezone": "US/Pacific",
        "Source": "Source2",
        "Temperature(F)": 72.0,
        "Wind_Chill(F)": 72.0,
        "Humidity(%)": 50.0,
        "Pressure(in)": 29.92,
        "Visibility(mi)": 10.0,
        "Wind_Speed(mph)": 5.0,
        "Precipitation(in)": 0.0,
        "Wind_Direction": "CALM",
        "Weather_Condition": "Clear",
        "Sunrise_Sunset": "Day",
        "Civil_Twilight": "Day",
        "Nautical_Twilight": "Day",
        "Astronomical_Twilight": "Day",
        "Amenity": False, "Bump": False, "Crossing": False, "Give_Way": False,
        "Junction": False, "No_Exit": False, "Railway": False, "Station": False,
        "Stop": False, "Traffic_Calming": False, "Traffic_Signal": False
    },
    "🌧️ Nocny wypadek na skrzyżowaniu podczas ulewy (Wysokie ryzyko)": {
        "Start_Time": datetime(2023, 10, 22, 23, 45),
        "Duration_min": 120.0,
        "Distance(mi)": 0.1,
        "Start_Lat": 42.3314,
        "Start_Lng": -83.0458,
        "State": "MI",
        "City": "Detroit",
        "County": "Wayne",
        "Timezone": "US/Eastern",
        "Source": "Source1",
        "Temperature(F)": 48.0,
        "Wind_Chill(F)": 44.0,
        "Humidity(%)": 92.0,
        "Pressure(in)": 29.50,
        "Visibility(mi)": 4.0,
        "Wind_Speed(mph)": 15.0,
        "Precipitation(in)": 0.25,
        "Wind_Direction": "SSW",
        "Weather_Condition": "Heavy Rain",
        "Sunrise_Sunset": "Night",
        "Civil_Twilight": "Night",
        "Nautical_Twilight": "Night",
        "Astronomical_Twilight": "Night",
        "Amenity": False, "Bump": False, "Crossing": True, "Give_Way": False,
        "Junction": False, "No_Exit": False, "Railway": False, "Station": False,
        "Stop": False, "Traffic_Calming": False, "Traffic_Signal": True
    },
    "❄️ Karambol podczas zamieci śnieżnej na węźle (Krytyczne ryzyko)": {
        "Start_Time": datetime(2023, 1, 18, 6, 15),
        "Duration_min": 240.0,
        "Distance(mi)": 2.5,
        "Start_Lat": 40.4406,
        "Start_Lng": -79.9959,
        "State": "PA",
        "City": "Pittsburgh",
        "County": "Allegheny",
        "Timezone": "US/Eastern",
        "Source": "Source1",
        "Temperature(F)": 18.0,
        "Wind_Chill(F)": 5.0,
        "Humidity(%)": 85.0,
        "Pressure(in)": 29.10,
        "Visibility(mi)": 0.5,
        "Wind_Speed(mph)": 25.0,
        "Precipitation(in)": 0.10,
        "Wind_Direction": "WNW",
        "Weather_Condition": "Heavy Snow",
        "Sunrise_Sunset": "Night",
        "Civil_Twilight": "Night",
        "Nautical_Twilight": "Night",
        "Astronomical_Twilight": "Night",
        "Amenity": False, "Bump": False, "Crossing": False, "Give_Way": False,
        "Junction": True, "No_Exit": False, "Railway": False, "Station": False,
        "Stop": False, "Traffic_Calming": False, "Traffic_Signal": False
    }
}

# ---------------------------------------------------------------------------
# Load models & artifacts (cached to prevent reload latency)
# ---------------------------------------------------------------------------
@st.cache_resource
def load_assets():
    try:
        # Paths
        meta_path = config.DATA_DIR / "metadata.json"
        scaler_path = config.DATA_DIR / "scaler.joblib"
        kmeans_path = config.DATA_DIR / "kmeans.joblib"
        mappings_path = config.DATA_DIR / "categorical_mappings.joblib"
        mlp_path = config.OUTPUT_DIR / "best_mlp.pt"
        lgbm_path = config.OUTPUT_DIR / "lgbm_model.joblib"
        summary_path = config.OUTPUT_DIR / "results_summary.json"

        # Check if assets exist
        for p in [meta_path, scaler_path, kmeans_path, mappings_path, mlp_path, lgbm_path]:
            if not p.exists():
                raise FileNotFoundError(f"Nie znaleziono wymaganego pliku: {p.name}. Upewnij się, że preprocessing oraz trening zostały ukończone.")

        # Load
        with open(meta_path, encoding="utf-8") as f:
            metadata = json.load(f)

        scaler = joblib.load(scaler_path)
        kmeans = joblib.load(kmeans_path)
        cat_mappings = joblib.load(mappings_path)
        
        # Predictors
        mlp_predictor = AccidentPredictor.from_checkpoint(mlp_path)
        lgbm_predictor = LGBMPredictor.from_joblib(lgbm_path, metadata)
        
        # Load training metrics
        metrics = []
        if summary_path.exists():
            with open(summary_path, encoding="utf-8") as f:
                metrics = json.load(f)
                
        return metadata, scaler, kmeans, cat_mappings, mlp_predictor, lgbm_predictor, metrics, None
    except Exception as e:
        return None, None, None, None, None, None, [], str(e)

metadata, scaler, kmeans, cat_mappings, mlp_predictor, lgbm_predictor, metrics_summary, load_error = load_assets()

# ---------------------------------------------------------------------------
# Custom CSS for styling
# ---------------------------------------------------------------------------
st.markdown("""
    <style>
    .main-header {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        padding: 2.5rem;
        border-radius: 12px;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px -1px rgb(0 0 0 / 0.1), 0 2px 4px -2px rgb(0 0 0 / 0.1);
        text-align: center;
    }
    .main-title {
        font-size: 2.5rem;
        font-weight: 800;
        margin-bottom: 0.5rem;
        color: #f8fafc;
    }
    .main-subtitle {
        font-size: 1.1rem;
        color: #cbd5e1;
        font-weight: 400;
    }
    .metric-card {
        background-color: #1e293b;
        border-left: 5px solid #3b82f6;
        padding: 1rem;
        border-radius: 6px;
        margin-bottom: 1rem;
        color: #f1f5f9;
    }
    .metric-title {
        font-size: 0.85rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .metric-value {
        font-size: 1.4rem;
        font-weight: 700;
        color: #f8fafc;
    }
    .severity-card {
        padding: 1.5rem;
        border-radius: 12px;
        margin-top: 1rem;
        margin-bottom: 1.5rem;
        color: white;
        box-shadow: 0 10px 15px -3px rgb(0 0 0 / 0.1);
    }
    .severity-title {
        font-size: 1.8rem;
        font-weight: 800;
        margin-bottom: 0.5rem;
    }
    .severity-desc {
        font-size: 1.05rem;
        opacity: 0.95;
        line-height: 1.5;
    }
    .risk-factor-tag {
        display: inline-block;
        background-color: rgba(244, 67, 54, 0.15);
        color: #ef4444;
        border: 1px solid rgba(244, 67, 54, 0.3);
        padding: 0.25rem 0.75rem;
        border-radius: 9999px;
        font-size: 0.85rem;
        margin-right: 0.5rem;
        margin-bottom: 0.5rem;
        font-weight: 500;
    }
    .risk-factor-container {
        margin-top: 1rem;
        padding: 1rem;
        background-color: #0f172a;
        border-radius: 8px;
        border: 1px solid #334155;
    }
    </style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Sidebar UI
# ---------------------------------------------------------------------------
st.sidebar.image("https://img.icons8.com/color/96/car-crash.png", width=80)
st.sidebar.markdown("# Panel Sterowania")

if load_error:
    st.error(f"⚠️ Błąd inicjalizacji: {load_error}")
    st.sidebar.error("Sprawdź konsolę i upewnij się, że modele są wytrenowane.")
    st.stop()

# Model Selection
model_choice = st.sidebar.selectbox(
    "Wybierz model predykcyjny",
    ["Oba modele (Porównanie)", "MLP (Entity Embeddings)", "LightGBM"]
)

st.sidebar.divider()

# Preset scenarios loader
st.sidebar.subheader("🚀 Szybkie Scenariusze (Presety)")
selected_preset_name = st.sidebar.selectbox(
    "Wczytaj gotowy zestaw parametrów:",
    list(PRESETS.keys()),
    key="selected_preset"
)

# Apply preset state update callback
if selected_preset_name != "Wybierz scenariusz...":
    preset_data = PRESETS[selected_preset_name]
    if preset_data:
        # Initialize or update state variables
        st.session_state['start_date'] = preset_data['Start_Time'].date()
        st.session_state['start_time'] = preset_data['Start_Time'].time()
        st.session_state['duration'] = preset_data['Duration_min']
        st.session_state['distance'] = preset_data['Distance(mi)']
        st.session_state['lat'] = preset_data['Start_Lat']
        st.session_state['lng'] = preset_data['Start_Lng']
        st.session_state['state'] = preset_data['State']
        st.session_state['city'] = preset_data['City']
        st.session_state['county'] = preset_data['County']
        st.session_state['timezone'] = preset_data['Timezone']
        st.session_state['source'] = preset_data['Source']
        st.session_state['temp'] = preset_data['Temperature(F)']
        st.session_state['wind_chill_val'] = preset_data['Wind_Chill(F)']
        st.session_state['wind_chill_missing'] = False
        st.session_state['humidity'] = preset_data['Humidity(%)']
        st.session_state['pressure'] = preset_data['Pressure(in)']
        st.session_state['visibility'] = preset_data['Visibility(mi)']
        st.session_state['wind_speed_val'] = preset_data['Wind_Speed(mph)']
        st.session_state['wind_speed_missing'] = False
        st.session_state['precip_val'] = preset_data['Precipitation(in)']
        st.session_state['precip_missing'] = False
        st.session_state['wind_dir'] = preset_data['Wind_Direction']
        st.session_state['weather_cond'] = preset_data['Weather_Condition']
        st.session_state['sunrise_sunset'] = preset_data['Sunrise_Sunset']
        st.session_state['civil_twilight'] = preset_data['Civil_Twilight']
        st.session_state['nautical_twilight'] = preset_data['Nautical_Twilight']
        st.session_state['astronomical_twilight'] = preset_data['Astronomical_Twilight']
        for col in BOOL_COLS:
            st.session_state[col.lower()] = preset_data[col]

st.sidebar.divider()

# Display Model metrics in sidebar
st.sidebar.subheader("📈 Metryki Jakości Modeli")
for model_m in metrics_summary:
    st.sidebar.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">{model_m['model']}</div>
        <div class="metric-value">Accuracy: {model_m['accuracy']:.2%}</div>
        <div style="font-size:0.85rem; color:#cbd5e1; margin-top:0.25rem;">
            Macro F1: <b>{model_m['macro_f1']:.3f}</b> | ROC AUC: <b>{model_m['roc_auc_macro']:.3f}</b>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Header Section
# ---------------------------------------------------------------------------
st.markdown("""
    <div class="main-header">
        <div class="main-title">Kalkulator Stopnia Ciężkości Wypadków Drogowych</div>
        <div class="main-subtitle">Interaktywny system analizy ryzyka oparty o uczenie maszynowe (MLP & LightGBM) na bazie 1 miliona wypadków w USA</div>
    </div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Session State Initialization
# ---------------------------------------------------------------------------
def init_state():
    if 'start_date' not in st.session_state:
        st.session_state['start_date'] = datetime(2022, 6, 9).date()
    if 'start_time' not in st.session_state:
        st.session_state['start_time'] = time(14, 30)
    if 'duration' not in st.session_state:
        st.session_state['duration'] = 60.0
    if 'distance' not in st.session_state:
        st.session_state['distance'] = 0.1
    if 'lat' not in st.session_state:
        st.session_state['lat'] = 37.7749
    if 'lng' not in st.session_state:
        st.session_state['lng'] = -122.4194
    if 'state' not in st.session_state:
        st.session_state['state'] = 'CA'
    if 'city' not in st.session_state:
        st.session_state['city'] = 'San Francisco'
    if 'county' not in st.session_state:
        st.session_state['county'] = 'San Francisco'
    if 'timezone' not in st.session_state:
        st.session_state['timezone'] = 'US/Pacific'
    if 'source' not in st.session_state:
        st.session_state['source'] = 'Source2'
    if 'temp' not in st.session_state:
        st.session_state['temp'] = 64.0
    if 'wind_chill_val' not in st.session_state:
        st.session_state['wind_chill_val'] = 64.0
    if 'wind_chill_missing' not in st.session_state:
        st.session_state['wind_chill_missing'] = True
    if 'humidity' not in st.session_state:
        st.session_state['humidity'] = 67.0
    if 'pressure' not in st.session_state:
        st.session_state['pressure'] = 29.87
    if 'visibility' not in st.session_state:
        st.session_state['visibility'] = 10.0
    if 'wind_speed_val' not in st.session_state:
        st.session_state['wind_speed_val'] = 7.0
    if 'wind_speed_missing' not in st.session_state:
        st.session_state['wind_speed_missing'] = True
    if 'precip_val' not in st.session_state:
        st.session_state['precip_val'] = 0.0
    if 'precip_missing' not in st.session_state:
        st.session_state['precip_missing'] = True
    if 'wind_dir' not in st.session_state:
        st.session_state['wind_dir'] = 'CALM'
    if 'weather_cond' not in st.session_state:
        st.session_state['weather_cond'] = 'Clear'
    if 'sunrise_sunset' not in st.session_state:
        st.session_state['sunrise_sunset'] = 'Day'
    if 'civil_twilight' not in st.session_state:
        st.session_state['civil_twilight'] = 'Day'
    if 'nautical_twilight' not in st.session_state:
        st.session_state['nautical_twilight'] = 'Day'
    if 'astronomical_twilight' not in st.session_state:
        st.session_state['astronomical_twilight'] = 'Day'
    for col in BOOL_COLS:
        col_key = col.lower()
        if col_key not in st.session_state:
            st.session_state[col_key] = False

init_state()

# Extract list options from loaded mappings
states_list = sorted([s for s in cat_mappings['State'].keys() if s != 'Unknown'])
cities_list = sorted([c for c in cat_mappings['City'].keys() if c != 'Unknown'])
counties_list = sorted([c for c in cat_mappings['County'].keys() if c != 'Unknown'])
timezones_list = sorted([t for t in cat_mappings['Timezone'].keys() if t != 'Unknown'])
sources_list = sorted([s for s in cat_mappings['Source'].keys() if s != 'Unknown'])
wind_dirs_list = sorted([w for w in cat_mappings['Wind_Direction'].keys() if w != 'Unknown'])

# Weather Condition list
weather_conditions = sorted(list(set(WEATHER_MAP.keys())))

# ---------------------------------------------------------------------------
# Tabs for parameters
# ---------------------------------------------------------------------------
tab_loc_time, tab_weather, tab_infra = st.tabs([
    "📅 Czas i Położenie", 
    "⛈️ Warunki Atmosferyczne", 
    "🚧 Infrastruktura i Otoczenie"
])

with tab_loc_time:
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### 🕒 Czas Zdarzenia")
        start_date = st.date_input("Data zdarzenia", key="start_date")
        start_time = st.time_input("Godzina zdarzenia", key="start_time")
        duration = st.number_input(
            "Czas trwania utrudnień (min)", 
            min_value=0.0, 
            max_value=1440.0, 
            key="duration",
            help="Czas, przez jaki wypadek blokował przejazd (w minutach)."
        )
        distance = st.number_input(
            "Długość korka / zatoru (mi)", 
            min_value=0.0, 
            max_value=100.0, 
            key="distance",
            help="Dystans wpływu wypadku na drogę w milach."
        )

    with col2:
        st.markdown("### 📍 Położenie geograficzne")
        lat = st.number_input(
            "Szerokość geograficzna (Latitude)", 
            min_value=24.0, 
            max_value=50.0, 
            format="%.6f", 
            key="lat",
            help="Szerokość geograficzna w USA (np. 30..49)"
        )
        lng = st.number_input(
            "Długość geograficzna (Longitude)", 
            min_value=-125.0, 
            max_value=-66.0, 
            format="%.6f", 
            key="lng",
            help="Długość geograficzna w USA (np. -124..-67)"
        )
        source = st.selectbox(
            "Źródło danych (Source)", 
            options=sources_list, 
            key="source"
        )
        timezone = st.selectbox(
            "Strefa czasowa (Timezone)", 
            options=timezones_list, 
            key="timezone"
        )

    with col3:
        st.markdown("### 🏛️ Podział Administracyjny")
        # Find default index of selected state
        state_idx = states_list.index(st.session_state.state) if st.session_state.state in states_list else 0
        state = st.selectbox("Stan (State)", options=states_list, index=state_idx, key="state")
        
        # City selectbox (with search)
        city_idx = cities_list.index(st.session_state.city) if st.session_state.city in cities_list else 0
        city = st.selectbox("Miasto (City)", options=cities_list, index=city_idx, key="city")
        
        # County selectbox
        county_idx = counties_list.index(st.session_state.county) if st.session_state.county in counties_list else 0
        county = st.selectbox("Hrabstwo (County)", options=counties_list, index=county_idx, key="county")

with tab_weather:
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### 🌡️ Temperatura i Ciśnienie")
        temp = st.slider("Temperatura (°F)", min_value=-40.0, max_value=120.0, key="temp")
        st.caption(f"Odpowiednik: {round((temp-32)*5/9, 1)} °C")
        
        # Wind chill input with missing support
        wind_chill_missing = st.checkbox("Brak danych o temperaturze odczuwalnej (Wind Chill)", key="wind_chill_missing")
        wind_chill = None
        if not wind_chill_missing:
            wind_chill = st.number_input("Temperatura odczuwalna (Wind Chill °F)", min_value=-60.0, max_value=120.0, key="wind_chill_val")
            st.caption(f"Odpowiednik: {round((wind_chill-32)*5/9, 1)} °C")
            
        pressure = st.number_input("Ciśnienie (in)", min_value=20.0, max_value=35.0, key="pressure")

    with col2:
        st.markdown("### 💨 Wiatr i Widoczność")
        humidity = st.slider("Wilgotność powietrza (%)", min_value=0.0, max_value=100.0, key="humidity")
        visibility = st.slider("Widoczność (mile)", min_value=0.0, max_value=10.0, key="visibility")
        
        # Wind speed input with missing support
        wind_speed_missing = st.checkbox("Brak danych o prędkości wiatru", key="wind_speed_missing")
        wind_speed = None
        if not wind_speed_missing:
            wind_speed = st.number_input("Prędkość wiatru (mph)", min_value=0.0, max_value=100.0, key="wind_speed_val")
            st.caption(f"Odpowiednik: {round(wind_speed * 1.609, 1)} km/h")

    with col3:
        st.markdown("### 🌧️ Opady i Warunki Ogólne")
        # Precipitation with missing support
        precip_missing = st.checkbox("Brak danych o opadach", key="precip_missing")
        precip = None
        if not precip_missing:
            precip = st.number_input("Opady (cale / Precipitation in)", min_value=0.0, max_value=5.0, format="%.2f", key="precip_val")
            st.caption(f"Odpowiednik: {round(precip * 25.4, 1)} mm")
            
        wind_dir = st.selectbox("Kierunek wiatru (Wind Direction)", options=wind_dirs_list, key="wind_dir")
        
        weather_cond_idx = weather_conditions.index(st.session_state.weather_cond) if st.session_state.weather_cond in weather_conditions else 0
        weather_cond = st.selectbox("Stan pogody (Weather Condition)", options=weather_conditions, index=weather_cond_idx, key="weather_cond")

with tab_infra:
    st.markdown("### 🚧 Otoczenie drogi i oświetlenie (Zaznacz jeśli występuje blisko miejsca wypadku)")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**Elementy Bezpieczeństwa / Spowalniacze**")
        bump = st.checkbox("Spowalniacz (Bump)", key="bump")
        crossing = st.checkbox("Przejście dla pieszych (Crossing)", key="crossing")
        traffic_calming = st.checkbox("Uspokojenie ruchu (Traffic Calming)", key="traffic_calming")
        traffic_signal = st.checkbox("Sygnalizacja świetlna (Traffic Signal)", key="traffic_signal")
        stop_sign = st.checkbox("Znak STOP (Stop)", key="stop")

    with col2:
        st.markdown("**Punkty Nawigacyjne / Skrzyżowania**")
        amenity = st.checkbox("Udogodnienia / Obiekt użyteczności (Amenity)", key="amenity")
        give_way = st.checkbox("Ustąp pierwszeństwa (Give Way)", key="give_way")
        junction = st.checkbox("Węzeł drogowy / Skrzyżowanie bezkolizyjne (Junction)", key="junction")
        no_exit = st.checkbox("Ślepa ulica / Brak zjazdu (No Exit)", key="no_exit")
        railway = st.checkbox("Przejazd kolejowy (Railway)", key="railway")
        station = st.checkbox("Stacja / Przystanek (Station)", key="station")

    with col3:
        st.markdown("**Jasność i Pory Dnia**")
        sunrise_sunset = st.selectbox("Pora dnia (Sunrise/Sunset)", options=["Day", "Night"], key="sunrise_sunset")
        civil_twilight = st.selectbox("Zmierzch cywilny", options=["Day", "Night"], key="civil_twilight")
        nautical_twilight = st.selectbox("Zmierzch nawigacyjny", options=["Day", "Night"], key="nautical_twilight")
        astronomical_twilight = st.selectbox("Zmierzch astronomiczny", options=["Day", "Night"], key="astronomical_twilight")

# ---------------------------------------------------------------------------
# Single Input Preprocessing Helper
# ---------------------------------------------------------------------------
def process_form_data():
    # Merge Date & Time
    start_dt = datetime.combine(start_date, start_time)
    
    # Pack raw input data for preprocessing
    raw_input = {
        'Start_Time': start_dt,
        'Duration_min': duration,
        'Distance(mi)': distance,
        'Start_Lat': lat,
        'Start_Lng': lng,
        'State': state,
        'City': city,
        'County': county,
        'Timezone': timezone,
        'Source': source,
        'Temperature(F)': temp,
        'Wind_Chill(F)': wind_chill,
        'Humidity(%)': humidity,
        'Pressure(in)': pressure,
        'Visibility(mi)': visibility,
        'Wind_Speed(mph)': wind_speed,
        'Precipitation(in)': precip,
        'Wind_Direction': wind_dir,
        'Weather_Condition': weather_cond,
        'Sunrise_Sunset': sunrise_sunset,
        'Civil_Twilight': civil_twilight,
        'Nautical_Twilight': nautical_twilight,
        'Astronomical_Twilight': astronomical_twilight,
    }
    for col in BOOL_COLS:
        raw_input[col] = st.session_state[col.lower()]
        
    # Derive features
    hour = start_dt.hour
    dayofweek = start_dt.weekday()
    month = start_dt.month
    year = start_dt.year
    
    is_weekend = 1 if dayofweek >= 5 else 0
    is_rush_hour = 1 if hour in [7, 8, 9, 16, 17, 18, 19] else 0
    
    hour_sin = np.sin(2 * np.pi * hour / 24.0)
    hour_cos = np.cos(2 * np.pi * hour / 24.0)
    dayofweek_sin = np.sin(2 * np.pi * dayofweek / 7.0)
    dayofweek_cos = np.cos(2 * np.pi * dayofweek / 7.0)
    month_sin = np.sin(2 * np.pi * month / 12.0)
    month_cos = np.cos(2 * np.pi * month / 12.0)
    
    # Geo distance from center
    dist_from_center = haversine_np(lng, lat, US_CENTER_LNG, US_CENTER_LAT)
    
    # K-means cluster
    cluster_id = int(kmeans.predict([[lat, lng]])[0])
    
    # Missingness Indicators
    wind_chill_was_missing = 1 if wind_chill is None else 0
    precip_was_missing = 1 if precip is None else 0
    wind_speed_was_missing = 1 if wind_speed is None else 0
    
    # Resolving imputation and capping
    imp = metadata['imputation_values']
    caps = metadata['capping_rules']
    
    temp_val = temp
    temp_val = np.clip(temp_val, caps['Temperature(F)'][0], caps['Temperature(F)'][1])
    
    wc_val = wind_chill
    if wc_val is None:
        wc_val = temp_val # Wind chill default
        
    hum_val = humidity
    hum_val = np.clip(hum_val, caps['Humidity(%)'][0], caps['Humidity(%)'][1])
    
    pres_val = pressure
    pres_val = np.clip(pres_val, caps['Pressure(in)'][0], caps['Pressure(in)'][1])
    
    vis_val = visibility
    vis_val = np.clip(vis_val, caps['Visibility(mi)'][0], caps['Visibility(mi)'][1])
    
    ws_val = wind_speed
    if ws_val is None:
        ws_val = imp['Wind_Speed(mph)']
    ws_val = np.clip(ws_val, caps['Wind_Speed(mph)'][0], caps['Wind_Speed(mph)'][1])
    
    pr_val = precip
    if pr_val is None:
        pr_val = imp['Precipitation(in)']
    pr_val = np.clip(pr_val, caps['Precipitation(in)'][0], caps['Precipitation(in)'][1])
    
    dist_val = distance
    dist_val = np.clip(dist_val, caps['Distance(mi)'][0], caps['Distance(mi)'][1])
    
    dur_val = duration
    dur_val = np.clip(dur_val, caps['Duration_min'][0], caps['Duration_min'][1])
    
    # Log transforms
    ws_log = np.log1p(ws_val)
    pr_log = np.log1p(pr_val)
    dist_log = np.log1p(dist_val)
    dur_log = np.log1p(dur_val)
    
    # Order numerical features for scaler
    num_features_ordered = {
        "Temperature(F)": temp_val,
        "Wind_Chill(F)": wc_val,
        "Humidity(%)": hum_val,
        "Pressure(in)": pres_val,
        "Visibility(mi)": vis_val,
        "Wind_Speed(mph)_log": ws_log,
        "Precipitation(in)_log": pr_log,
        "Distance(mi)_log": dist_log,
        "Duration_min_log": dur_log,
        "Start_Lat": lat,
        "Start_Lng": lng,
        "Dist_from_center": dist_from_center,
        "Hour_sin": hour_sin,
        "Hour_cos": hour_cos,
        "DayOfWeek_sin": dayofweek_sin,
        "DayOfWeek_cos": dayofweek_cos,
        "Month_sin": month_sin,
        "Month_cos": month_cos,
        "Year": float(year)
    }
    
    num_df = pd.DataFrame([num_features_ordered], columns=metadata['num_cols'])
    num_scaled = scaler.transform(num_df)[0]
    scaled_num_dict = {col: num_scaled[i] for i, col in enumerate(metadata['num_cols'])}
    
    # Categoricals mapping
    mapped_wind_dir = WIND_DIR_MAP.get(wind_dir, 'VAR')
    mapped_weather_cat = WEATHER_MAP.get(weather_cond, 'Other')
    
    raw_cats = {
        'Source': source,
        'State': state,
        'City': city,
        'County': county,
        'Timezone': timezone,
        'Wind_Direction': mapped_wind_dir,
        'Weather_Category': mapped_weather_cat,
        'Geo_Cluster': cluster_id
    }
    
    mapped_cat_dict = {}
    for col in metadata['cat_cols']:
        val = raw_cats[col]
        mapping = cat_mappings[col]
        mapped_cat_dict[col] = int(mapping.get(val, 0))
        
    # Binary setup
    twilight_dict = {}
    for col in TWILIGHT_COLS:
        twilight_dict[col] = 1 if raw_input[col] == 'Night' else 0
        
    bool_dict = {}
    for col in BOOL_COLS:
        bool_dict[col] = 1 if raw_input[col] else 0
        
    was_missing_dict = {
        'Wind_Chill(F)_was_missing': wind_chill_was_missing,
        'Precipitation(in)_was_missing': precip_was_missing,
        'Wind_Speed(mph)_was_missing': wind_speed_was_missing,
    }
    
    extra_bin_dict = {
        'IsWeekend': is_weekend,
        'IsRushHour': is_rush_hour
    }
    
    bin_dict = {}
    bin_dict.update(bool_dict)
    bin_dict.update(twilight_dict)
    bin_dict.update(was_missing_dict)
    bin_dict.update(extra_bin_dict)
    
    # Combine all
    row_dict = {}
    row_dict.update(scaled_num_dict)
    row_dict.update(mapped_cat_dict)
    row_dict.update(bin_dict)
    
    return row_dict, raw_input

# ---------------------------------------------------------------------------
# Trigger Predyckji
# ---------------------------------------------------------------------------
st.markdown("<br>", unsafe_allow_html=True)
col_btn, _ = st.columns([1, 3])
with col_btn:
    predict_clicked = st.button("🔮 Oblicz stopień ciężkości wypadku", type="primary", use_container_width=True)

if predict_clicked:
    # Process
    processed_row, raw_input = process_form_data()
    
    # Predict based on selected model
    results = {}
    
    with st.spinner("Trwa obliczanie prawdopodobieństwa..."):
        if model_choice in ["MLP (Entity Embeddings)", "Oba modele (Porównanie)"]:
            try:
                results["MLP (Entity Embeddings)"] = mlp_predictor.predict(processed_row)
            except Exception as e:
                st.error(f"Błąd MLP: {str(e)}")
                
        if model_choice in ["LightGBM", "Oba modele (Porównanie)"]:
            try:
                results["LightGBM"] = lgbm_predictor.predict(processed_row)
            except Exception as e:
                st.error(f"Błąd LightGBM: {str(e)}")
                
    st.markdown("## 📊 Wyniki Predykcji")
    
    # Layout based on results
    num_models = len(results)
    cols = st.columns(num_models)
    
    for idx, (m_name, res) in enumerate(results.items()):
        with cols[idx]:
            pred_class = res['class']
            confidence = res['confidence']
            probs = res['probabilities']
            color = config.SEVERITY_COLORS.get(pred_class, '#3b82f6')
            pl_label = POLISH_LABELS.get(pred_class, f"Severity {pred_class + 1}")
            pl_desc = POLISH_DESCS.get(pred_class, "")
            
            # Predict Card
            st.markdown(f"""
                <div class="severity-card" style="background-color: {color};">
                    <div style="font-size:0.9rem; opacity:0.8; text-transform:uppercase; font-weight:700;">Model: {m_name}</div>
                    <div class="severity-title">{pl_label}</div>
                    <div style="font-size:1.1rem; font-weight:600; margin-bottom:1rem;">Pewność modelu: {confidence:.2%}</div>
                    <div class="severity-desc">{pl_desc}</div>
                </div>
            """, unsafe_allow_html=True)
            
            # Probabilities chart
            st.markdown("### Rozkład prawdopodobieństwa:")
            prob_df = pd.DataFrame({
                "Stopień": [POLISH_LABELS[i] for i in range(4)],
                "Prawdopodobieństwo": probs
            })
            
            # Bar chart
            st.bar_chart(prob_df.set_index("Stopień"), y="Prawdopodobieństwo", color=color, horizontal=True)

    # ---------------------------------------------------------------------------
    # Explainer & Map
    # ---------------------------------------------------------------------------
    st.divider()
    col_map, col_factors = st.columns([1, 1])
    
    with col_map:
        st.markdown("### 🗺️ Lokalizacja zdarzenia na mapie")
        map_df = pd.DataFrame({'lat': [lat], 'lon': [lng]})
        st.map(map_df, zoom=10)
        st.caption(f"Współrzędne: {lat:.5f}, {lng:.5f}. Geoklaster K-Means: {processed_row.get('Geo_Cluster', 'Nieznany')}")
        
    with col_factors:
        st.markdown("### 🔍 Analiza Czynników Ryzyka")
        
        # Risk factor analyzer
        def analyze_risk_factors(raw_in):
            factors = []
            if raw_in.get('Junction'):
                factors.append("Skrzyżowanie bezkolizyjne / węzeł autostradowy (częsty punkt kolizji o dużej prędkości)")
            if raw_in.get('Crossing'):
                factors.append("Przejście dla pieszych (zwiększa ryzyko potrąceń i gwałtownego hamowania)")
            if raw_in.get('Traffic_Signal'):
                factors.append("Sygnalizacja świetlna (związana z kolizjami bocznymi lub najechaniem na tył)")
            if raw_in.get('Stop'):
                factors.append("Znak STOP (częste wymuszenia pierwszeństwa)")
                
            w_cond = raw_in.get('Weather_Condition', 'Clear')
            w_cat = WEATHER_MAP.get(w_cond, 'Other')
            if w_cat in ['Storm', 'Snow', 'Rain', 'Fog']:
                factors.append(f"Złe warunki atmosferyczne: {w_cond} (zmniejszona przyczepność i widoczność)")
                
            t = raw_in.get('Temperature(F)', 64.0)
            if t is not None and t < 32.0:
                factors.append(f"Mroz ({t}°F / {round((t-32)*5/9, 1)}°C) — ryzyko gołoledzi i oblodzenia jezdni")
                
            v = raw_in.get('Visibility(mi)', 10.0)
            if v is not None and v < 3.0:
                factors.append(f"Niska widoczność ({v} mil) — skrócony czas na reakcję")
                
            p = raw_in.get('Precipitation(in)', 0.0)
            if p is not None and p > 0.05:
                factors.append(f"Ślad opadowy ({p} cala) — aquaplaning lub błoto pośniegowe")
                
            dt = raw_in['Start_Time']
            h = dt.hour
            if h in [7, 8, 9, 16, 17, 18, 19]:
                factors.append("Godziny szczytu drogowego (wzmożone natężenie ruchu zwiększa efekt zatoru)")
                
            twilight = raw_in.get('Sunrise_Sunset', 'Day')
            if twilight == 'Night':
                factors.append("Pora nocna — ograniczona widoczność, większe zmęczenie kierowców")
                
            if raw_in.get('Duration_min', 60.0) > 120.0:
                factors.append("Przewidywany czas utrudnień przekracza 2 godziny (paraliż komunikacyjny)")
                
            if raw_in.get('Distance(mi)', 0.1) > 1.0:
                factors.append("Zasięg zatoru przekracza 1 milę (poważne zakłócenia ruchu drogowego)")
                
            return factors
            
        factors_list = analyze_risk_factors(raw_input)
        
        if factors_list:
            st.markdown("Wykryte czynniki mogące podnieść stopień ciężkości zdarzenia:")
            st.markdown('<div class="risk-factor-container">', unsafe_allow_html=True)
            for f in factors_list:
                st.markdown(f'<span class="risk-factor-tag">⚠️ {f}</span>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.success("Brak wyraźnych, negatywnych czynników infrastruktury i pogody. Warunki sprzyjające bezpiecznej jeździe.")
            
        st.markdown("""
        > **Uwaga metodologiczna:** Klasyfikacja ciężkości wypadków opiera się na skali **US Severity (1-4)**, 
        > gdzie stopień 1 to zdarzenia o minimalnym wpływie na ruch, a stopień 4 oznacza zdarzenia krytyczne 
        > wywołujące całkowitą blokadę ruchu na długi czas. Modele zostały wytrenowane na zbalansowanej próbie 
        > 1 000 000 rekordów.
        """)
