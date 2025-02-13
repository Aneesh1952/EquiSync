
import numpy as np
import pandas as pd
import tensorflow as tf
from flask import Flask, request, jsonify
from sklearn.preprocessing import MinMaxScaler

# Load trained model
model = tf.keras.models.load_model('EquiSync_Model.h5')

# Initialize Flask app
app = Flask(__name__)

# Define a scaler to match training preprocessing
scaler = MinMaxScaler()
features = ['Stride_Length (m)', 'Acceleration (m/s^2)', 'Speed (km/h)', 'Heart_Rate (bpm)', 'Oxygen_Level (%)']

def preprocess_input(data):
    df = pd.DataFrame([data], columns=features)
    df_scaled = scaler.fit_transform(df)
    return df_scaled.reshape((1, 1, len(features)))

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        processed_data = preprocess_input(data)
        prediction = model.predict(processed_data)
        return jsonify({'Performance_Score': float(prediction[0][0])})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=5000)
