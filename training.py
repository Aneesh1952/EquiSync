
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# Load datasets
biomechanics_df = pd.read_csv('Horse_Biomechanics.csv')
vitals_df = pd.read_csv('Horse_VitalSigns.csv')
performance_df = pd.read_csv('Race_Performance.csv')

# Merge datasets on Horse_ID and Timestamp
merged_df = biomechanics_df.merge(vitals_df, on=['Horse_ID', 'Timestamp']).merge(performance_df, on='Horse_ID')

# Selecting relevant features and target
features = ['Stride_Length (m)', 'Acceleration (m/s^2)', 'Speed (km/h)', 'Heart_Rate (bpm)', 'Oxygen_Level (%)']
target = 'Performance_Score'

# Data preprocessing
scaler = MinMaxScaler()
X = scaler.fit_transform(merged_df[features])
y = merged_df[target].values

# Reshape for LSTM (samples, timesteps, features)
X = X.reshape((X.shape[0], 1, X.shape[1]))
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build LSTM Model
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(1, X.shape[2])),
    Dropout(0.2),
    LSTM(50, return_sequences=False),
    Dropout(0.2),
    Dense(25, activation='relu'),
    Dense(1)
])

model.compile(optimizer='adam', loss='mse')

# Train the model
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=50, batch_size=16)

# Save model
model.save('EquiSync_Model.h5')

print("Model training completed and saved as EquiSync_Model.h5")
