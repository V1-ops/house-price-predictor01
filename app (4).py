import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load model
model = joblib.load("house_price_xgb.pkl")

st.set_page_config(page_title="🏡 House Price Predictor", layout="centered")

st.title("🏡 House Price Predictor")
st.write("Enter details below to estimate house price")

# Sidebar Inputs
area = st.slider("Area (sqft)", 500, 5000, 1200)
rooms = st.slider("Number of Rooms", 1, 10, 3)
age = st.slider("Age of House (years)", 0, 100, 10)
location = st.selectbox("Location", ["Downtown", "Suburban", "Countryside"])

# Convert location into numerical feature
location_map = {"Downtown": 2, "Suburban": 1, "Countryside": 0}
loc_value = location_map[location]

# Create input DataFrame
input_data = pd.DataFrame([[area, rooms, age, loc_value]],
                          columns=["Area", "Rooms", "Age", "Location"])

# Prediction
if st.button("Predict Price"):
    prediction = model.predict(input_data)[0]
    st.success(f"💰 Estimated Price: ${prediction*1000:,.2f}")

    st.metric("House Area", f"{area} sqft")
    st.metric("Rooms", rooms)
    st.metric("Age", f"{age} years")

