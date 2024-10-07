import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import KNNImputer

# Load pre-saved models
imputer = pickle.load(open('/home/smayan/Desktop/DJS-compute-tasks/DJS-compute-tasks/Task 3/Data/knnimputer.sav', 'rb'))
lr = pickle.load(open('/home/smayan/Desktop/DJS-compute-tasks/DJS-compute-tasks/Task 3/Data/lr_clf.sav', 'rb'))
scaler = pickle.load(open('/home/smayan/Desktop/DJS-compute-tasks/DJS-compute-tasks/Task 3/Data/lr_scaler.sav', 'rb'))
label_encoder = pickle.load(open('/home/smayan/Desktop/DJS-compute-tasks/DJS-compute-tasks/Task 3/Data/lr_encoder.sav', 'rb'))
columns_to_encode = ['Location', 'WindGustDir', 'WindDir9am', 'WindDir3pm']
# Streamlit app title
st.title('Rain Prediction in Australia')

# Load dataset for reference
df = pd.read_csv('/home/smayan/Desktop/DJS-compute-tasks/DJS-compute-tasks/Task 3/Data/weather_australia.csv')

# Define features and inputs for user
st.header("Input Weather Features for Prediction")
location = st.selectbox('Location', df['Location'].unique())
min_temp = st.number_input('Min Temp', min_value=-10.0, max_value=50.0, value=20.0)
max_temp = st.number_input('Max Temp', min_value=-10.0, max_value=50.0, value=25.0)
rainfall = st.number_input('Rainfall (mm)', min_value=0.0, max_value=500.0, value=5.0)
wind_gust_dir = st.selectbox('Wind Gust Direction', df['WindGustDir'].unique())
wind_gust_speed = st.number_input('Wind Gust Speed (km/h)', min_value=0.0, max_value=200.0, value=30.0)
humidity_9am = st.number_input('Humidity at 9am (%)', min_value=0.0, max_value=100.0, value=75.0)
humidity_3pm = st.number_input('Humidity at 3pm (%)', min_value=0.0, max_value=100.0, value=60.0)
pressure_9am = st.number_input('Pressure at 9am (hPa)', min_value=900.0, max_value=1100.0, value=1010.0)
pressure_3pm = st.number_input('Pressure at 3pm (hPa)', min_value=900.0, max_value=1100.0, value=1008.0)
temp_9am = st.number_input('Temp at 9am (°C)', min_value=-10.0, max_value=50.0, value=22.0)
temp_3pm = st.number_input('Temp at 3pm (°C)', min_value=-10.0, max_value=50.0, value=24.0)
wind_dir_9am = st.selectbox('Wind Direction at 9am', df['WindDir9am'].unique())
wind_dir_3pm = st.selectbox('Wind Direction at 3pm', df['WindDir3pm'].unique())
wind_speed_3pm = st.number_input('Wind Speed at 3pm (km/h)', min_value=0.0, max_value=150.0, value=20.0)
rain_today = st.selectbox('Rain Today?', ['Yes', 'No'])

# Preprocess inputs
input_data = pd.DataFrame({
    'Location': [location],
    'MinTemp': [min_temp],
    'MaxTemp': [max_temp],
    'Rainfall': [rainfall],
    'WindGustDir': [wind_gust_dir],
    'WindGustSpeed': [wind_gust_speed],
    'Humidity9am': [humidity_9am],
    'Humidity3pm': [humidity_3pm],
    'Pressure9am': [pressure_9am],
    'Pressure3pm': [pressure_3pm],
    'Temp9am': [temp_9am],
    'Temp3pm': [temp_3pm],
    'WindDir9am': [wind_dir_9am],
    'WindDir3pm': [wind_dir_3pm],
    'WindSpeed3pm': [wind_speed_3pm],
    'RainToday': [1 if rain_today == 'Yes' else 0]
})

# Apply label encoding on categorical features
input_data[columns_to_encode] = input_data[columns_to_encode].apply(lambda col: label_encoder.transform(col))

# Impute missing values and scale data
input_data_imputed = imputer.transform(input_data)
input_data_scaled = scaler.transform(input_data_imputed)

# Prediction
if st.button('Predict Rain Tomorrow'):
    prediction = lr.predict(input_data_scaled)
    if prediction[0] == 1:
        st.success("It will rain tomorrow!")
    else:
        st.success("It won't rain tomorrow!")

# Conclusion
st.write("This app predicts whether it will rain tomorrow using your input for weather features.")
