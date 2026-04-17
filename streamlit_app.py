import streamlit as st
import numpy as np
import pickle

# load model
model = pickle.load(open("car_model.pkl", "rb"))

st.title("🚗 Car Price Prediction")

# ---------------- INPUTS ---------------- #

Kilometers_Driven = st.number_input("Kilometers Driven", value=50000)
Seats = st.number_input("Seats", value=5)
mileage_num = st.number_input("Mileage (km/l)", value=18.0)
car_age = st.number_input("Car Age (years)", value=5)

fuel = st.selectbox("Fuel Type", ["petrol", "diesel", "lpg"])
transmission = st.selectbox("Transmission", ["manual", "automatic"])
owner = st.selectbox("Owner Type", ["first", "second", "third", "fourth"])

# ---------------- PREDICT BUTTON ---------------- #

if st.button("Predict Price"):

    # ---------- ENCODING ---------- #
    Fuel_Type_Diesel = 1 if fuel == "diesel" else 0
    Fuel_Type_LPG = 1 if fuel == "lpg" else 0
    Fuel_Type_Petrol = 1 if fuel == "petrol" else 0

    Transmission_Manual = 1 if transmission == "manual" else 0

    Owner_Type_Second = 1 if owner == "second" else 0
    Owner_Type_Third = 1 if owner == "third" else 0
    Owner_Type_Fourth = 1 if owner == "fourth" else 0

    # ---------- FEATURE ARRAY (MUST MATCH MODEL) ---------- #
    features = [[
        Kilometers_Driven,
        Seats,
        mileage_num,
        car_age,
        Fuel_Type_Diesel,
        Fuel_Type_LPG,
        Fuel_Type_Petrol,
        Transmission_Manual,
        Owner_Type_Fourth,
        Owner_Type_Second,
        Owner_Type_Third
    ]]

    features = np.array(features)

    # ---------- SAFETY CHECK ---------- #
    if np.isnan(features).any():
        st.error("Invalid input detected. Please check values.")
    else:
        try:
            prediction = model.predict(features)[0]
            st.success(f"Estimated Price: {round(prediction,2)} Lakhs")
        except Exception as e:
            st.error(f"Error: {e}")
