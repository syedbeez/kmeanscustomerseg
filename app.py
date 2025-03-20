import streamlit as st
import pickle
import numpy as np

# Load the saved models
with open('kmeans_model.pkl', 'rb') as file:
    kmeans = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

with open('encoder.pkl', 'rb') as file:
    encoder = pickle.load(file)

st.title("Customer Segmentation with KMeans")

# User Inputs
quantity = st.number_input("Quantity", min_value=1, max_value=500, value=10)
unit_price = st.number_input("Unit Price", min_value=0.1, max_value=500.0, value=5.0)
country = st.text_input("Country (e.g., United Kingdom)")

if st.button("Predict Cluster"):
    try:
        country_encoded = encoder.transform([country])[0]
        total_price = quantity * unit_price
        user_input = np.array([[quantity, unit_price, country_encoded, total_price]])
        user_input_scaled = scaler.transform(user_input)
        
        cluster = kmeans.predict(user_input_scaled)[0]
        st.success(f"The customer belongs to Cluster: {cluster}")

    except Exception as e:
        st.error("Error: Ensure the country name is correctly spelled and exists in the dataset.")
