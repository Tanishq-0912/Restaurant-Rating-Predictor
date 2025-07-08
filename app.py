import streamlit as st
import pandas as pd
import numpy as np
import pickle

st.title("🍽️ Restaurant Rating Predictor")

# Load model and encoders using pickle
try:
    with open("restaurant_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open("city_encoder.pkl", "rb") as f:
        le_city = pickle.load(f)
    with open("cuisine_encoder.pkl", "rb") as f:
        le_cuisine = pickle.load(f)

    st.success("✅ Model and encoders loaded successfully")
except Exception as e:
    st.error(f"❌ Error loading model or encoders: {e}")
    st.stop()
