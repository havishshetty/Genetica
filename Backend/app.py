from flask import Flask, request, jsonify
from flask_cors import CORS  # Import CORS
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import os  # Import os for handling the environment variable for port

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load the Diabetes Prediction Model and Scaler
try:
    with open('Diabetes_Prediction_Model_Test.pkl', 'rb') as diabetes_model_file:
        diabetes_model = pickle.load(diabetes_model_file)

    # Load the dataset for diabetes model and fit the scaler
    diabetes_data = pd.read_csv('preprocessed_diabetes_data.csv')
    diabetes_scaler = StandardScaler()
    numerical_features = ['age', 'bmi', 'HbA1c_level', 'blood_glucose_level']
    diabetes_scaler.fit(diabetes_data[numerical_features])
except Exception as e:
    print(f"Error loading the diabetes model or scaler: {e}")

# Load the Lung Cancer Prediction Model
try:
    lung_cancer_model_file_path = 'lung_cancer_rf_model.pkl'
    with open(lung_cancer_model_file_path, 'rb') as lung_cancer_model_file:
        lung_cancer_model = pickle.load(lung_cancer_model_file)
except Exception as e:
    print(f"Error loading the lung cancer model: {e}")

# Home route
@app.route('/')
def home():
    return "Welcome to the Prediction API!"

# Diabetes prediction route
@app.route('/predict_diabetes', methods=['POST'])
def predict_diabetes():
    try:
        # Extract user inputs from the request JSON
        user_input = request.json

        # Extracting values from user input
        name = user_input['name']
        gender = user_input['gender']
        age = user_input['age']
        hypertension = user_input['hypertension']
        heart_disease = user_input['heart_disease']
        smoking_history = user_input['smoking_history']
        bmi = user_input['bmi']
        HbA1c_level = user_input['HbA1c_level']
        blood_glucose_level = user_input['blood_glucose_level']

        # Convert categorical inputs to numeric
        gender_numeric = 1 if gender == "Male" else 0
        hypertension_numeric = 1 if hypertension == "Yes" else 0
        heart_disease_numeric = 1 if heart_disease == "Yes" else 0
        smoking_history_numeric = {
            "never": 0,
            "current": 1,
            "formerly": 2,
            "No Info": 3,
            "ever": 4,
            "not current": 5
        }[smoking_history]

        # Create feature vector
        inputs = np.array([[age, bmi, HbA1c_level, blood_glucose_level]])
        scaled_inputs = diabetes_scaler.transform(inputs)
        feature_vector = np.concatenate(
            ([gender_numeric, hypertension_numeric, heart_disease_numeric, smoking_history_numeric], 
            scaled_inputs.flatten())).reshape(1, -1)

        # Make prediction using the pre-trained diabetes model
        prediction = diabetes_model.predict(feature_vector)
        result = "Diabetic" if prediction[0] == 1 else "Non-Diabetic"

        # Construct the response JSON
        response = {
            'name': name,
            'result': result  # Only returning the result
        }

        return jsonify(response)

    except Exception as e:
        return jsonify({"error": str(e)})

# Lung Cancer prediction route
@app.route('/predict_lung_cancer', methods=['POST'])
def predict_lung_cancer():
    try:
        # Get data from the request (expects JSON with 'features' array)
        data = request.get_json()
        
        # Extract features and convert them to a NumPy array
        features = np.array(data['features']).reshape(1, -1)
        
        # Make a prediction using the pre-trained lung cancer model
        prediction = lung_cancer_model.predict(features)
        
        # Since this is a classification problem, the result is the predicted class
        predicted_class = int(prediction[0])
        
        # Return the prediction as a JSON response
        return jsonify({'prediction': predicted_class})

    except Exception as e:
        # Return an error message if something goes wrong
        return jsonify({'error': str(e)})

# Start the Flask application with the correct host and port for deployment
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))  # Use the port provided by Render or default to 5000
    app.run(debug=True, host="0.0.0.0", port=port)
