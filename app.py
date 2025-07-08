import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.title("🍽️ Restaurant Rating Predictor")

# Load all saved components
try:
    model = joblib.load("restaurant_model.pkl")
    scaler = joblib.load("scaler.pkl")
    le_city = joblib.load("city_encoder.pkl")
    le_cuisine = joblib.load("cuisine_encoder.pkl")
    st.success("✅ Model and encoders loaded successfully")
except Exception as e:
    st.error(f"❌ Error loading files: {e}")
    st.stop()

