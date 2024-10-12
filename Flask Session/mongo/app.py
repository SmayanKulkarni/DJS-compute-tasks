from flask import Flask, render_template, request, redirect, url_for
from pymongo import MongoClient
import secrets

app = Flask(__name__)

# Generate a secret key for session management
app.config['SECRET_KEY'] = secrets.token_hex(16)

# MongoDB Atlas connection URI
app.config['MONGO_URI'] = "mongodb+srv://hello:hello1@cluster0.eiqnk.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
client = MongoClient(app.config['MONGO_URI'])
db = client['hello']  # Replace <dbname> with your database name
students_collection = db['students']  # Collection name

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        name = request.form['name']
        sap_id = request.form['sap_id']
        
        # Insert the data into MongoDB
        student_data = {
            'name': name,
            'sap_id': sap_id
        }
        students_collection.insert_one(student_data)
        return redirect(url_for('index'))  # Redirect to the form after submission
    
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
