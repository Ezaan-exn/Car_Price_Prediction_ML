import streamlit as st
import pickle
import pandas as pd

# Load model + encoders
model = pickle.load(open("car_price_model.pkl", "rb"))
fuel_le = pickle.load(open("fuel_encoder.pkl", "rb"))
trans_le = pickle.load(open("trans_encoder.pkl", "rb"))
owner_le = pickle.load(open("owner_encoder.pkl", "rb"))

st.title("🚗 Car Price Predictor")

# Inputs
km = st.number_input("Kilometers Driven", 0)
seats = st.number_input("Seats", 1)
mileage = st.number_input("Mileage (km/l)", 0.0)
age = st.number_input("Car Age", 0)

fuel = st.selectbox("Fuel Type", fuel_le.classes_)
trans = st.selectbox("Transmission", trans_le.classes_)
owner = st.selectbox("Owner Type", owner_le.classes_)

# Predict
if st.button("Predict Price"):

    input_df = pd.DataFrame([{
        "Kilometers_Driven": km,
        "Seats": seats,
        "mileage_num": mileage,
        "Fuel_Type": fuel_le.transform([fuel])[0],
        "Transmission": trans_le.transform([trans])[0],
        "Owner_Type": owner_le.transform([owner])[0],
        "car_age": age
    }])

    prediction = model.predict(input_df)[0]

    st.success(f"💰 Estimated Price: ₹ {round(prediction, 2)} Lakhs")