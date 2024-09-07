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

# Define categorical and numerical columns based on the dataset
nonnumeric = ['make', 'fuel-type', 'aspiration', 'body-style', 'drive-wheels',
              'engine-location', 'engine-type', 'fuel-system']
numerical_columns = ['wheel-base', 'length', 'width', 'curb-weight', 
                     'engine-size', 'bore', 'horsepower', 'city-mpg', 'highway-mpg']

# Load the final training dataset for PCA transformation (final_data.csv)
df_final = pd.read_csv('/home/smayan/Desktop/DJS-compute-tasks/DJS-compute-tasks/Task 2/Notebook/final_data.csv')

# Extract feature names from the final dataset used for PCA transformation
features = df_final.drop(['price'], axis=1).columns.tolist()

# Fit the PCA on the original training data (use final_data.csv)
pca = PCA(n_components=0.96)
X_train = df_final.drop(['price'], axis=1)
X_train_scaled = scale(X_train)
X_train_pca = pca.fit_transform(X_train_scaled)

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
    # Combine categorical and numerical data into a single DataFrame
    input_df = pd.DataFrame([input_data], columns=df_forstreamlit.columns)

    # One-hot encode the categorical data
    X_encoded = pd.get_dummies(input_df, columns=nonnumeric)

    # Ensure that the input has all the required columns (fill missing columns with 0s)
    X_encoded = X_encoded.reindex(columns=features, fill_value=0)

    # Fill any remaining NaN values in numerical columns (e.g., mean for numerical columns)
    X_encoded[numerical_columns] = X_encoded[numerical_columns].fillna(df_forstreamlit[numerical_columns].mean())

    # Scale the input data and apply the trained PCA transformation
    X_input_scaled = scale(X_encoded)
    X_input_pca = pca.transform(X_input_scaled)

    # Predict the price using the trained model
    prediction = model.predict(X_input_pca)

    # Display the predicted price
    st.write(f"Predicted Price: ${prediction[0]:,.2f}")
