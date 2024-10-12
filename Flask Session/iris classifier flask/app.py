from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np 

# Load the saved model
with open('models/iris_classifier.pkl', 'rb') as file:
    model = pickle.load(file)

app = Flask(__name__)

# Mapping from numerical prediction to species name
species_mapping = {
    0: 'Iris Setosa',
    1: 'Iris Versicolor',
    2: 'Iris Virginica'
}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get data from the form using request.form
    sepal_length = float(request.form['sepal-length']) 
    sepal_width = float(request.form['sepal-width'])
    petal_length = float(request.form['petal-length'])
    petal_width = float(request.form['petal-width'])

    # Create a 2D array for prediction
    features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

    # Make prediction
    prediction = model.predict(features)

    # Map the prediction to the species name
    predicted_species = species_mapping.get(prediction[0], "Unknown Species")

    # Return prediction as JSON
    return jsonify({'prediction': predicted_species})


if __name__ == '__main__':
    app.run(debug=True)
