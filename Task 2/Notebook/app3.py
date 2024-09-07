import streamlit as st
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import scale
from sklearn.decomposition import PCA

# Load the trained model
model = pickle.load(open('/home/smayan/Desktop/DJS-compute-tasks/DJS-compute-tasks/Task 2/Data/model.sav', 'rb'))

# Load the dataset with accurate categories and columns (forstreamlit.csv)
df_forstreamlit = pd.read_csv('/home/smayan/Desktop/DJS-compute-tasks/DJS-compute-tasks/Task 2/Notebook/forstreamlit.csv')
df_forstreamlit.drop(columns=['Unnamed: 0'], axis=1, inplace=True)

# Define categorical and numerical columns based on the dataset
nonnumeric = ['make', 'fuel-type', 'aspiration', 'body-style', 'drive-wheels',
              'engine-location', 'engine-type', 'fuel-system']
numerical_columns = ['wheel-base', 'length', 'width', 'curb-weight', 
                     'engine-size', 'bore', 'horsepower', 'city-mpg', 'highway-mpg']
features = df_forstreamlit.drop(columns=['price'], axis=1).columns.tolist()

# Fit PCA using the scaled version of the dataset (excluding 'price')
X_for_pca = df_forstreamlit.drop(columns=['price'])
X_for_pca_scaled = scale(pd.get_dummies(X_for_pca, columns=nonnumeric))
pca = PCA(n_components=0.96)
pca.fit(X_for_pca_scaled)

# Streamlit app interface
st.title("Car Price Prediction App")

# Categorical inputs (dropdowns) based on actual data from the DataFrame
input_data = {}
for col in nonnumeric:
    unique_values = df_forstreamlit[col].unique().tolist()
    input_data[col] = st.selectbox(f"Select {col}", unique_values)

# Numerical inputs (sliders) based on actual min/max values from the DataFrame
for col in numerical_columns:
    min_val = float(df_forstreamlit[col].min())
    max_val = float(df_forstreamlit[col].max())
    input_data[col] = st.slider(f"Select {col}", min_val, max_val, (min_val + max_val) / 2)

# When the user clicks "Predict"
if st.button("Predict"):
    # Combine user input categorical and numerical data into a single DataFrame
    input_df = pd.DataFrame([input_data], columns=df_forstreamlit.columns)

    # Encode categorical data
    X_encoded = pd.get_dummies(input_df, columns=nonnumeric)

    # Ensure that the input has all required columns (align with training data)
    X_encoded = X_encoded.reindex(columns=features, fill_value=0).dropna()

    # Scale the input data and apply PCA transformation
    X_input_scaled = scale(X_encoded)
    X_input_pca = pca.transform(X_input_scaled)[:,:29]

    # Predict price based on user input
    user_prediction = model.predict(X_input_pca)

    # Display the predicted price for user input
    st.write(f"Predicted Price (User Input): ${user_prediction[0]:,.2f}")

    # ---- Dummy Input Generation and Prediction ----

    # Create random dummy values for all columns (either 0 or 1 for categorical)
    dummy_values = np.random.randint(0, 2, size=(1, len(features)))
    dummy_df = pd.DataFrame(dummy_values, columns=features)

    # Scale and apply PCA transformation to dummy input
    dummy_scaled = scale(dummy_df)
    dummy_pca = pca.transform(dummy_scaled)[:,:29]

    # Predict price based on the dummy input
    dummy_prediction = model.predict(dummy_pca)

    # Display the predicted price for dummy input
    st.write(f"Predicted Price (Dummy Input): ${dummy_prediction[0]:,.2f}")
