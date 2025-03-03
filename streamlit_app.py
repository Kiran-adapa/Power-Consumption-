import joblib
import numpy as np
import streamlit as st
from sklearn.preprocessing import StandardScaler

# Load the trained model, scaler, and PCA
try:
    model = joblib.load('random_forest_model.pkl')
    scaler = joblib.load('scaled.pkl')
    pca = joblib.load('pca.pkl')  # Load PCA model
except FileNotFoundError:
    st.error("Model, scaler, or PCA file not found. Please check file paths.")
    st.stop()

# Streamlit app title
st.title('Power Consumption Prediction in Wellington, New Zealand')
st.write('Enter various environmental factors to predict power consumption in Zone 1.')

# Define the input fields
st.sidebar.header('Input Features')
Temperature = st.sidebar.number_input("Temperature (°C)", min_value=-10.0, max_value=50.0, value=20.0, step=0.1)
Humidity = st.sidebar.number_input("Humidity (%)", min_value=0.0, max_value=100.0, value=50.0, step=0.1)
WindSpeed = st.sidebar.number_input("Wind Speed (km/h)", min_value=0.0, max_value=100.0, value=10.0, step=0.1)
general_diffuse_flows = st.sidebar.number_input("General Diffuse Flows", min_value=0.0, max_value=100.0, value=30.0, step=0.1)
diffuse_flows = st.sidebar.number_input("Diffuse Flows", min_value=0.0, max_value=100.0, value=20.0, step=0.1)
Air_Quality_Index = st.sidebar.number_input("Air Quality Index (PM)", min_value=0.0, max_value=500.0, value=50.0, step=0.1)
Cloudiness = st.sidebar.number_input("Cloudiness (%)", min_value=0.0, max_value=100.0, value=50.0, step=0.1)

# Prediction button
if st.sidebar.button("Predict"):
    try:
        # Prepare input for prediction
        input_data = np.array([[Temperature, Humidity, WindSpeed, general_diffuse_flows, diffuse_flows, Air_Quality_Index, Cloudiness]])
        
        # Apply scaling
        scaled_input = scaler.transform(input_data)

        # Apply PCA transformation
        input_pca = pca.transform(scaled_input)

        # Make prediction
        prediction = model.predict(input_pca)

        # Display prediction result
        st.success(f"Predicted Power Consumption: {float(prediction[0]):.2f} kWh")

    except Exception as e:
        st.error(f"An error occurred during prediction: {str(e)}")
