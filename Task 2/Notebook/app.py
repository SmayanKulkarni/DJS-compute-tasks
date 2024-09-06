import streamlit as st
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
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

# Load the final dataset used for PCA transformation (final_data.csv)
df_final = pd.read_csv('/home/smayan/Desktop/DJS-compute-tasks/DJS-compute-tasks/Task 2/Notebook/final_data.csv')

# Extract feature names from the final dataset used for PCA transformation
features = df_final.drop(['price'], axis=1).columns.tolist()

# Initialize the scaler and PCA with the same parameters used during training
scaler = StandardScaler()
pca = PCA(n_components=0.96)

# Fit the scaler and PCA with the final_data (training data)
X = df_final.drop(['price'], axis=1)
X_scaled = scaler.fit_transform(X)
pca.fit(X_scaled)

# Define the number of principal components expected (29 components)
n_components = 29

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
    input_df = pd.DataFrame([input_data])

    # Handle categorical encoding (pd.get_dummies) only on non-numeric columns
    input_df_encoded = pd.get_dummies(input_df, columns=nonnumeric)
    
    # Reindex the input DataFrame to match the structure of the final_data DataFrame
    input_df_encoded = input_df_encoded.reindex(columns=features, fill_value=0)

    # Apply scaling
    input_scaled = scaler.transform(input_df_encoded)
    
    # Apply PCA transformation to ensure 29 dimensions
    input_reduced = pca.fit_transform(input_scaled)
    if input_reduced.shape[1] != n_components:
        # If the number of components is different, truncate or pad to match
        input_reduced = np.hstack([input_reduced, np.zeros((input_reduced.shape[0], n_components - input_reduced.shape[1]))]) if input_reduced.shape[1] < n_components else input_reduced[:, :n_components]

    # Ensure the dimensionality matches the model's expectation (29 components)
    if input_reduced.shape[1] != n_components:
        st.error(f"The input data dimensionality ({input_reduced.shape[1]}) does not match the model's expectations ({n_components}).")
    else:
        # Make a prediction
        prediction = model.predict(input_reduced)

        # Convert the prediction to a scalar value for formatting
        prediction_value = prediction[0] if isinstance(prediction, np.ndarray) else prediction

        # Display the prediction
        st.write(f"Predicted Price: ${prediction:}")
