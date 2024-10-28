import pandas as pd
import pickle
from flask import Flask, render_template, request

app = Flask(__name__)

def preprocess_input_data(input_data, le):
    
    input_df = pd.DataFrame([input_data])
    
    input_df['sex'] = input_df['sex'].map({'male': 1, 'female': 0})
    
    input_df['island_encoded'] = le.transform(input_df['island'])  
    
    return input_df[['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g', 'sex','year', 'island_encoded']]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    
    with open('Task 4\Decision Tree\decision_tree_app\model\clf.pkl', 'rb') as file:
        clf = pickle.load(file)
    
    with open('Task 4\Decision Tree\decision_tree_app\model\label_encoder.pkl', 'rb') as f:
        le = pickle.load(f)

    input_data = {
        'year': int(request.form['year']),  
        'bill_length_mm': float(request.form['bill_length_mm']),
        'bill_depth_mm': float(request.form['bill_depth_mm']),
        'flipper_length_mm': float(request.form['flipper_length_mm']),
        'body_mass_g': float(request.form['body_mass_g']),
        'island': request.form['island'],
        'sex': request.form['sex']
    }

    processed_data = preprocess_input_data(input_data, le)
    
    prediction = clf.predict(processed_data)
    predicted_species = prediction[0]

    return render_template('results.html', predicted_species=predicted_species)

if __name__ == '__main__':
    app.run(debug=True)
