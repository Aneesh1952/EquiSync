import numpy as np
import pandas as pd
import tensorflow as tf
from flask import Flask, request, jsonify
from flask_cors import CORS
from sklearn.preprocessing import MinMaxScaler
import os

app = Flask(__name__)
CORS(app)

# Global variables
model = None
scaler = MinMaxScaler()
features = ['Stride_Length (m)', 'Acceleration (m/s^2)', 'Speed (km/h)', 
           'Heart_Rate (bpm)', 'Oxygen_Level (%)']

def load_model():
    """Load the TensorFlow model with error handling"""
    global model
    try:
        # Specify the custom_objects to handle loading
        model = tf.keras.models.load_model('EquiSync_Model.h5', compile=False)
        # Recompile the model
        model.compile(optimizer='adam', loss='mse')
        return True
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return False

def preprocess_input(data):
    """Preprocess input data with error handling"""
    try:
        # Ensure all features are present
        input_data = [float(data.get(feature, 0)) for feature in features]
        df = pd.DataFrame([input_data], columns=features)
        df_scaled = scaler.fit_transform(df)
        return df_scaled.reshape((1, 1, len(features)))
    except Exception as e:
        raise ValueError(f"Error preprocessing input: {str(e)}")

@app.before_first_request
def initialize():
    """Initialize the model before first request"""
    if not load_model():
        print("Failed to load model during initialization")

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'model_loaded': model is not None})

@app.route('/predict', methods=['POST'])
def predict():
    """Prediction endpoint with comprehensive error handling"""
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 503
    
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No input data provided'}), 400
        
        # Validate input data
        missing_features = [f for f in features if f not in data]
        if missing_features:
            return jsonify({'error': f'Missing features: {missing_features}'}), 400
        
        processed_data = preprocess_input(data)
        prediction = model.predict(processed_data)
        
        return jsonify({
            'status': 'success',
            'Performance_Score': float(prediction[0][0]),
            'input_features': data
        })
    
    except ValueError as ve:
        return jsonify({'error': str(ve)}), 400
    except Exception as e:
        return jsonify({'error': f'Prediction error: {str(e)}'}), 500

if __name__ == '__main__':
    # Use environment variable for port with fallback
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
