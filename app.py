import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TF logging
import numpy as np
import pandas as pd
import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')  # Disable GPU
from flask import Flask, request, jsonify
from flask_cors import CORS
from sklearn.preprocessing import MinMaxScaler
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables
model = None
scaler = MinMaxScaler()
features = ['Stride_Length (m)', 'Acceleration (m/s^2)', 'Speed (km/h)', 
           'Heart_Rate (bpm)', 'Oxygen_Level (%)']

def create_app():
    """Create and configure the Flask application"""
    app = Flask(__name__)
    CORS(app)
    
    # Load model at startup
    global model
    try:
        # Configure TF for CPU
        tf.config.threading.set_intra_op_parallelism_threads(1)
        tf.config.threading.set_inter_op_parallelism_threads(1)
        
        model = tf.keras.models.load_model('EquiSync_Model.h5', compile=False)
        model.compile(optimizer='adam', loss='mse')
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
    
    def preprocess_input(data):
        """Preprocess input data with error handling"""
        try:
            input_data = [float(data.get(feature, 0)) for feature in features]
            df = pd.DataFrame([input_data], columns=features)
            df_scaled = scaler.fit_transform(df)
            return df_scaled.reshape((1, 1, len(features)))
        except Exception as e:
            logger.error(f"Error preprocessing input: {str(e)}")
            raise ValueError(f"Error preprocessing input: {str(e)}")

    @app.route('/health', methods=['GET'])
    def health_check():
        """Health check endpoint"""
        status = 'healthy' if model is not None else 'unhealthy'
        logger.info(f"Health check: {status}")
        return jsonify({
            'status': status,
            'model_loaded': model is not None,
            'environment': 'production'
        })

    @app.route('/predict', methods=['POST'])
    def predict():
        """Prediction endpoint with comprehensive error handling"""
        if model is None:
            logger.error("Model not loaded")
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
            prediction = model.predict(processed_data, verbose=0)
            
            response = {
                'status': 'success',
                'Performance_Score': float(prediction[0][0]),
                'input_features': data
            }
            logger.info(f"Successful prediction: {response['Performance_Score']}")
            return jsonify(response)
        
        except ValueError as ve:
            logger.error(f"Validation error: {str(ve)}")
            return jsonify({'error': str(ve)}), 400
        except Exception as e:
            logger.error(f"Prediction error: {str(e)}")
            return jsonify({'error': f'Prediction error: {str(e)}'}), 500

    return app

# Create the application instance
app = create_app()

if __name__ == '__main__':
    # Get port from environment variable for Render
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
