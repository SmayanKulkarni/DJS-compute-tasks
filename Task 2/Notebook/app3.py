import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pickle

# Load the saved model
model = pickle.load(open('/home/smayan/Desktop/DJS-compute-tasks/DJS-compute-tasks/Task 2/Data/model.sav', 'rb'))

# Load the final data
df_combined = pd.read_csv('final_data.csv')

# Create a Streamlit app
st.title("Automobile Price Prediction App")

# Add a sidebar with options
st.sidebar.header("Options")
option = st.sidebar.selectbox("Select an option", ["View Data", "Make a Prediction"])

# View Data option
if option == "View Data":
    st.header("Data")
    st.write(df_combined.head(10))

# Make a Prediction option
elif option == "Make a Prediction":
    st.header("Make a Prediction")
    # Create input fields for the user to enter values
    num_doors = st.number_input("Number of Doors")
    bore = st.number_input("Bore")
    stroke = st.number_input("Stroke")
    horsepower = st.number_input("Horsepower")
    peak_rpm = st.number_input("Peak RPM")
    price = st.number_input("Price")
    city_mpg = st.number_input("City MPG")
    highway_mpg = st.number_input("Highway MPG")
    engine_size = st.number_input("Engine Size")
    curb_weight = st.number_input("Curb Weight")
    make = st.selectbox("Make", df_combined['make'].unique())
    fuel_type = st.selectbox("Fuel Type", df_combined['fuel-type'].unique())
    aspiration = st.selectbox("Aspiration", df_combined['aspiration'].unique())
    body_style = st.selectbox("Body Style", df_combined['body-style'].unique())
    drive_wheels = st.selectbox("Drive Wheels", df_combined['drive-wheels'].unique())
    engine_location = st.selectbox("Engine Location", df_combined['engine-location'].unique())
    engine_type = st.selectbox("Engine Type", df_combined['engine-type'].unique())
    fuel_system = st.selectbox("Fuel System", df_combined['fuel-system'].unique())

    # Create a dictionary to store the input values
    input_values = {
        'num-of-doors': num_doors,
        'bore': bore,
        'stroke': stroke,
        'horsepower': horsepower,
        'peak-rpm': peak_rpm,
        'price': price,
        'city-mpg': city_mpg,
        'highway-mpg': highway_mpg,
        'engine-size': engine_size,
        'curb-weight': curb_weight,
        'make': make,
        'fuel-type': fuel_type,
        'aspiration': aspiration,
        'body-style': body_style,
        'drive-wheels': drive_wheels,
        'engine-location': engine_location,
        'engine-type': engine_type,
        'fuel-system': fuel_system
    }

    # Create a DataFrame from the input values
    input_df = pd.DataFrame([input_values])

    # Encode the categorical values
    nonnumeric = ['make','fuel-type','aspiration','body-style','drive-wheels','engine-location','engine-type','fuel-system']
    input_df_encoded = pd.get_dummies(input_df, columns=nonnumeric)

    # Scale the input values
    input_scaled = scale(input_df_encoded)

    # Apply PCA to the scaled input values
    pca = PCA(n_components=0.96)
    input_pca = pca.fit_transform(input_scaled)

    # Make a prediction using the model
    prediction = model.predict(input_pca)

    # Display the prediction
    st.write("Predicted Price: ${:.2f}".format(prediction[0]))