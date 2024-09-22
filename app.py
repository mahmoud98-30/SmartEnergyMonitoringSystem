from datetime import datetime

import streamlit as st
import pandas as pd
from pycaret.regression import load_model, predict_model


model = load_model('renewable_energy_pipeline')

st.title('Smart Energy Monitoring System')

st.markdown('''
This app predicts the **Solar Power Production (kw)** based on various environmental and system parameters.
''')

# Sidebar inputs
st.sidebar.header('Input Parameters')


def user_input_features():
    start_date = st.sidebar.date_input('Date of Measurement', datetime.now())
    capacity_clipped = st.sidebar.number_input('Maximum Capacity', min_value=0.0, value=1000.0)
    time_hourly = st.sidebar.date_input('time hourly', datetime.now())
    S_d = st.sidebar.number_input('Direct Beam Solar Radiation Intensity', min_value=0.0, value=500.0)
    airmass = st.sidebar.number_input('Airmass', min_value=0.0, value=1.5)
    altitude = st.sidebar.number_input('Altitude (degrees)', min_value=0.0, max_value=90.0, value=45.0)
    azimuth = st.sidebar.number_input('Azimuth (degrees)', min_value=0.0, max_value=360.0, value=180.0)
    irradiation = st.sidebar.number_input('Irradiation', min_value=0.0, value=600.0)
    fold_cos = st.sidebar.number_input('Fold Cosine', min_value=-1.0, max_value=1.0, value=0.5)
    panel_cos = st.sidebar.number_input('Panel Cosine', min_value=-1.0, max_value=1.0, value=0.5)
    day = st.sidebar.date_input('Day of Year', datetime.now())
    sunrise = st.sidebar.date_input('Sunrise Time (hour)', datetime.now())
    sunset = st.sidebar.date_input('Sunset Time (hour)', datetime.now())
    rad_lw_mean = st.sidebar.number_input('Long Wave Radiation', min_value=0.0, value=400.0)
    cloud_total_mean = st.sidebar.slider('Total Cloud Cover (%)', 0, 100, 50)
    temp_total_mean = st.sidebar.number_input('Temperature (°C)', min_value=-50.0, max_value=60.0, value=25.0)
    cloud_high_mean = st.sidebar.slider('High Cloud Cover (%)', 0, 100, 20)
    rad_global_mean = st.sidebar.number_input('Global Radiation', min_value=0.0, value=700.0)
    cloud_low_mean = st.sidebar.slider('Low Cloudiness (%)', 0, 100, 30)
    radNetS_lw_mean = st.sidebar.number_input('Net Long-Wave Radiation', min_value=0.0, value=350.0)
    precip_total_mean = st.sidebar.number_input('precip total mean', min_value=0.0, value=350.0)
    cloud_mid_mean = st.sidebar.slider('Medium Cloud Cover (%)', 0, 100, 25)
    radNetS_sw_mean = st.sidebar.number_input('Net Short-Wave Radiation', min_value=0.0, value=300.0)
    data = {
        'start': pd.to_datetime(start_date, errors='coerce'),
        'capacity_clipped': capacity_clipped,
        'time_hourly': pd.to_datetime(time_hourly, errors='coerce'),
        'S_d': S_d,
        'airmass': airmass,
        'altitude': altitude,
        'azimuth': azimuth,
        'irradiation': irradiation,
        'fold_cos': fold_cos,
        'panel_cos': panel_cos,
        'day': pd.to_datetime(day, errors='coerce'),
        'sunrise': pd.to_datetime(sunrise, errors='coerce'),
        'sunset': pd.to_datetime(sunset, errors='coerce'),
        'rad_lw_mean': rad_lw_mean,
        'cloud_total_mean': cloud_total_mean,
        'temp_total_mean': temp_total_mean,
        'cloud_high_mean': cloud_high_mean,
        'rad_global_mean': rad_global_mean,
        'cloud_low_mean': cloud_low_mean,
        'precip_total_mean': precip_total_mean,
        'radNetS_lw_mean': radNetS_lw_mean,
        'cloud_mid_mean': cloud_mid_mean,
        'radNetS_sw_mean': radNetS_sw_mean
    }
    features = pd.DataFrame(data, index=[0])
    return features


input_df = user_input_features()

# Display user input
st.subheader('User Input Parameters')
st.write(input_df)

# Make prediction
prediction = predict_model(model, data=input_df)

st.subheader('Predicted Solar Power Production (kw)')
Label = str(prediction['prediction_label'].iloc[0])
st.write(f'{Label} kw')
