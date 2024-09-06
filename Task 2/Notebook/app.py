import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression

# Load the model (replace with your actual path)
model = pickle.load(open('/home/smayan/Desktop/DJS-compute-tasks/DJS-compute-tasks/Task 2/Data/model.sav', 'rb'))

# Load the DataFrame from the CSV file
df = pd.read_csv('/home/smayan/Desktop/DJS-compute-tasks/DJS-compute-tasks/Task 2/Notebook/forstreamlit.csv')

# Encode categorical variables
le = LabelEncoder()
df['make'] = le.fit_transform(df['make'])
# Encode other categorical variables as needed

# Streamlit app
st.title("Linear Regression Prediction")

# Categorical features
make = st.selectbox("Make", df['make'].unique())
fuel_type = st.selectbox("Fuel Type", df['fuel-type'].unique())
aspiration = st.selectbox("Aspiration", df['aspiration'].unique())
num_of_doors = st.selectbox("Number of Doors", df['num-of-doors'].unique())
body_style = st.selectbox("Body Style", df['body-style'].unique())
drive_wheels = st.selectbox("Drive Wheels", df['drive-wheels'].unique())
engine_location = st.selectbox("Engine Location", df['engine-location'].unique())
engine_type = st.selectbox("Engine Type", df['engine-type'].unique())
num_of_cylinders = st.selectbox("Number of Cylinders", df['num-of-cylinders'].unique())
fuel_system = st.selectbox("Fuel System", df['fuel-system'].unique())

# Numerical features
symboling = st.slider("Symboling", min_value=df['symboling'].min(), max_value=df['symboling'].max())
normalized_losses = st.slider("Normalized Losses", min_value=df['normalized-losses'].min(), max_value=df['normalized-losses'].max())
wheel_base = st.slider("Wheel Base", min_value=df['wheel-base'].min(), max_value=df['wheel-base'].max())
length = st.slider("Length", min_value=df['length'].min(), max_value=df['length'].max())
width = st.slider("Width", min_value=df['width'].min(), max_value=df['width'].max())
height = st.slider("Height", min_value=df['height'].min(), max_value=df['height'].max())
curb_weight = st.slider("Curb Weight", min_value=df['curb-weight'].min(), max_value=df['curb-weight'].max())
engine_size = st.slider("Engine Size", min_value=df['engine-size'].min(), max_value=df['engine-size'].max())
bore = st.slider("Bore", min_value=df['bore'].min(), max_value=df['bore'].max())
stroke = st.slider("Stroke", min_value=df['stroke'].min(), max_value=df['stroke'].max())
compression_ratio = st.slider("Compression Ratio", min_value=df['compression-ratio'].min(), max_value=df['compression-ratio'].max())
horsepower = st.slider("Horsepower", min_value=df['horsepower'].min(), max_value=df['horsepower'].max())
peak_rpm = st.slider("Peak RPM", min_value=df['peak-rpm'].min(), max_value=df['peak-rpm'].max())
city_mpg = st.slider("City MPG", min_value=df['city-mpg'].min(), max_value=df['city-mpg'].max())
highway_mpg = st.slider("Highway MPG", min_value=df['highway-mpg'].min(), max_value=df['highway-mpg'].max())

# Create a new DataFrame with the user's input
new_data = {
    'symboling': [symboling],
    'normalized-losses': [normalized_losses],
    'make': [le.transform([make])[0]],  # Encode the 'make' value
    'fuel-type': [fuel_type],
    'aspiration': [aspiration],
    # ... other columns
}

new_df = pd.DataFrame(new_data)

# Make a prediction
prediction = model.predict(new_df)

# Display the prediction
st.write("Predicted Value:", prediction[0])