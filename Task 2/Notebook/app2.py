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
df_forstreamlit.drop(columns=['Unnamed: 0'], axis=1,inplace=True)
# Define categorical and numerical columns based on the dataset
nonnumeric = ['make', 'fuel-type', 'aspiration', 'body-style', 'drive-wheels',
              'engine-location', 'engine-type', 'fuel-system']
numerical_columns = ['wheel-base', 'length', 'width', 'curb-weight', 
                     'engine-size', 'bore', 'horsepower', 'city-mpg', 'highway-mpg']
features = df_forstreamlit.drop(columns=['price'], axis=1).columns.tolist()
pca = PCA(n_components=0.96)
# Streamlit app interface
st.title("Car Price Prediction App")

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

    # Trial code
    columns = input_df.columns

    # Generate random values (0 or 1) for all columns in input_df
    random_values = np.random.randint(0, 2, size=(1, len(columns)))

    # Create a DataFrame with the random values
    input_data = pd.DataFrame(random_values, columns=columns)


    X_encoded = pd.get_dummies(input_data, columns=nonnumeric)
    df_combined = pd.concat([input_data.drop(columns=nonnumeric).reset_index(allow_duplicates=False, drop=True), X_encoded.reset_index(allow_duplicates=False,drop=True)], axis=1)

    X = df_combined.drop(['price'], axis = 1)
    X_reduced = pca.fit_transform(scale(X_encoded))
    X_reduced_test= pca.fit_transform(scale(X))[:,:24]
    prediction = model.predict(X_reduced_test)
    # Display the predicted price
    st.write(f"Predicted Price: ${prediction[0]:,.2f}")
