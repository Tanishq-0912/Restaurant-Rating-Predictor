import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.title("🍽️ Restaurant Rating Predictor")

# Load model and preprocessing objects
try:
    model = joblib.load("restaurant_model.pkl")
    scaler = joblib.load("scaler.pkl")
    le_city = joblib.load("city_encoder.pkl")
    le_cuisine = joblib.load("cuisine_encoder.pkl")
    st.success("✅ Model and encoders loaded successfully")
except Exception as e:
    st.error(f"❌ Error loading files: {e}")
    st.stop()

# User Input Form
st.subheader("🔧 Enter Restaurant Details")

average_cost = st.number_input("💸 Average Cost for Two", min_value=50, max_value=5000, value=500)
price_range = st.selectbox("📊 Price Range (1 = Low, 4 = High)", [1, 2, 3, 4])
table_booking = st.radio("🪑 Has Table Booking?", ["Yes", "No"])
online_delivery = st.radio("📦 Has Online Delivery?", ["Yes", "No"])
votes = st.number_input("🗳️ Number of Votes", min_value=0, value=100)
city = st.text_input("🏙️ City", "New Delhi")
cuisine = st.text_input("🍽️ Cuisine", "North Indian")

# Predict Button
if st.button("🔍 Predict Rating"):
    try:
        # Convert categorical
        table_booking = 1 if table_booking == "Yes" else 0
        online_delivery = 1 if online_delivery == "Yes" else 0

        # Encode city and cuisine
        city_encoded = le_city.transform([city])[0] if city in le_city.classes_ else -1
        cuisine_encoded = le_cuisine.transform([cuisine])[0] if cuisine in le_cuisine.classes_ else -1

        if city_encoded == -1 or cuisine_encoded == -1:
            st.warning("⚠️ Unknown city or cuisine — model might not predict accurately.")
        
        # Final input array
        user_data = np.array([[average_cost, price_range, table_booking,
                               online_delivery, votes, city_encoded, cuisine_encoded]])

        # Scale input
        user_data_scaled = scaler.transform(user_data)

        # Predict
        prediction = model.predict(user_data_scaled)[0]
        st.success(f"⭐ Predicted Rating: {round(prediction, 2)} / 5")
    
    except Exception as e:
        st.error(f"❌ Error during prediction: {e}")
