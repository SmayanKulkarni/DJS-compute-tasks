import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import scale
from sklearn.decomposition import PCA

# Load the model
model_path = '/home/smayan/Desktop/DJS-compute-tasks/DJS-compute-tasks/Task 2/Data/model.sav'
model = pickle.load(open(model_path, 'rb'))

# Load training data to fit PCA
data_path = '/home/smayan/Desktop/DJS-compute-tasks/DJS-compute-tasks/Task 2/Notebook/final_data.csv'
df_combined = pd.read_csv(data_path)

# Verify columns
print("Columns in the dataset:", df_combined.columns.tolist())

# Preprocessing steps
nonnumeric = ['make', 'fuel-type', 'aspiration', 'body-style', 'drive-wheels', 'engine-location', 'engine-type', 'fuel-system']

# Check if nonnumeric columns are in the DataFrame
missing_cols = [col for col in nonnumeric if col not in df_combined.columns]
if missing_cols:
    st.error(f"Missing columns in the dataset: {', '.join(missing_cols)}")
else:
    df_encoded = pd.get_dummies(df_combined, columns=nonnumeric)
    df_combined = df_combined.drop(columns=nonnumeric)
    df_combined = pd.concat([df_combined, df_encoded], axis=1)
    df_combined = df_combined.fillna(0)  # Handle missing values

    # Split data for PCA
    X = df_combined.drop(['price'], axis=1)
    y = df_combined['price']

    # Fit PCA with the same parameters as used in training
    pca = PCA(n_components=0.96)
    X_reduced = pca.fit_transform(scale(X))

    # Function to preprocess the input data
    def preprocess_input(data):
        df = pd.DataFrame([data])
        df['num-of-doors'] = df['num-of-doors'].map({'two': 2, 'four': 4})
        df['num-of-cylinders'] = df['num-of-cylinders'].map({'six': 6, 'four': 4, 'five': 5, 'three': 3, 'twelve': 12, 'eight': 8})
        df_encoded = pd.get_dummies(df, columns=nonnumeric)
        for col in X.columns:
            if col not in df_encoded.columns:
                df_encoded[col] = 0
        df_encoded = df_encoded.fillna(0)
        return df_encoded

    # Streamlit app
    st.title("Car Price Prediction")

    # Input fields for user to enter car details
    make = st.selectbox('Make', ['audi', 'bmw', 'chevrolet', 'dodge', 'honda', 'jaguar', 'mazda', 'mitsubishi', 'nissan', 'peugeot', 'plymouth', 'saab', 'subaru', 'toyota', 'volkswagen'])
    fuel_type = st.selectbox('Fuel Type', ['gas', 'diesel'])
    aspiration = st.selectbox('Aspiration', ['std', 'turbo'])
    body_style = st.selectbox('Body Style', ['hardtop', 'wagon', 'sedan', 'convertible', 'hatchback'])
    drive_wheels = st.selectbox('Drive Wheels', ['fwd', 'rwd', '4wd'])
    engine_location = st.selectbox('Engine Location', ['front', 'rear'])
    engine_type = st.selectbox('Engine Type', ['dohc', 'ohc', 'l', 'ohcf', 'rotor'])
    fuel_system = st.selectbox('Fuel System', ['mpfi', '2bbl', '4bbl', 'spdi', '1bbl', 'spfi'])

    num_of_doors = st.selectbox('Number of Doors', ['two', 'four'])
    num_of_cylinders = st.selectbox('Number of Cylinders', ['four', 'six', 'five', 'three', 'twelve', 'eight'])
    bore = st.number_input('Bore', min_value=0.0, step=0.1)
    stroke = st.number_input('Stroke', min_value=0.0, step=0.1)
    horsepower = st.number_input('Horsepower', min_value=0, step=1)
    peak_rpm = st.number_input('Peak RPM', min_value=0, step=1)

    # Predict button
    if st.button('Predict'):
        input_data = {
            'make': make,
            'fuel-type': fuel_type,
            'aspiration': aspiration,
            'body-style': body_style,
            'drive-wheels': drive_wheels,
            'engine-location': engine_location,
            'engine-type': engine_type,
            'fuel-system': fuel_system,
            'num-of-doors': num_of_doors,
            'num-of-cylinders': num_of_cylinders,
            'bore': bore,
            'stroke': stroke,
            'horsepower': horsepower,
            'peak-rpm': peak_rpm
        }
        
        preprocessed_data = preprocess_input(input_data)
        X_reduced = pca.transform(scale(preprocessed_data))
        prediction = model.predict(X_reduced)
        
        st.write(f"Predicted Price: ${prediction[0]:,.2f}")

