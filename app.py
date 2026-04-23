import streamlit as st
import pickle
import pandas as pd


data = pd.read_excel("Car4u_used_cars_data.xlsx")

# Load model + encoders
model = pickle.load(open("car_price_model.pkl", "rb"))
fuel_le = pickle.load(open("fuel_encoder.pkl", "rb"))
trans_le = pickle.load(open("trans_encoder.pkl", "rb"))
owner_le = pickle.load(open("owner_encoder.pkl", "rb"))

# 🔗 Generate car link
def get_car_link(car_name):
    base_url = "https://www.cardekho.com/search"
    return f"{base_url}?q={car_name.replace(' ', '+')}"

# 🚗 Recommendation function
def recommend_cars(pred_price, data, fuel, trans, n=5):
    lower = pred_price - 2
    upper = pred_price + 2

    filtered = data[
        (data['Price'] >= lower) & 
        (data['Price'] <= upper) &
        (data['Fuel_Type'] == fuel) &
        (data['Transmission'] == trans)
    ]

    if filtered.empty:
        filtered = data[(data['Price'] >= lower) & (data['Price'] <= upper)]

    filtered = filtered.copy()
    filtered['diff'] = abs(filtered['Price'] - pred_price)

    filtered = filtered.sort_values(by='diff')

    return filtered.head(n)
    
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

    # 🚗 RECOMMENDATION SECTION
    st.markdown("## 🚗 Here are some recommendations for you:")

    recs = recommend_cars(prediction, data, fuel, trans)

    if recs.empty:
        st.warning("No similar cars found")
    else:
        for _, row in recs.iterrows():
            car_link = get_car_link(row['Model'])

            st.markdown(f"""
---
### 🚘 {row['Model']}
💰 Price: ₹ {row['Price']} Lakhs  
⛽ Fuel: {row['Fuel_Type']}  
⚙ Transmission: {row['Transmission']}  
📍 KM Driven: {row['Kilometers_Driven']}

👉 [Check this car online]({car_link})
""")