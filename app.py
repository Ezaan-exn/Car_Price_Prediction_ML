import streamlit as st
import pickle
import pandas as pd

# =========================
# LOAD MODEL + ENCODERS
# =========================
model = pickle.load(open("car_price_model.pkl", "rb"))
fuel_le = pickle.load(open("fuel_encoder.pkl", "rb"))
trans_le = pickle.load(open("trans_encoder.pkl", "rb"))
owner_le = pickle.load(open("owner_encoder.pkl", "rb"))

# Load dataset for recommendations
data = pd.read_excel("Car4u_used_cars_data.xlsx")


# =========================
# RECOMMENDATION FUNCTION
# =========================
def recommend_cars(pred_price, data, n=5):
    lower = pred_price - 2
    upper = pred_price + 2

    filtered = data[(data['Price'] >= lower) & (data['Price'] <= upper)]

    filtered = filtered.copy()
    filtered['diff'] = abs(filtered['Price'] - pred_price)

    return filtered.sort_values(by='diff').head(n)


def get_car_link(name):
    return f"https://www.cardekho.com/search?q={name.replace(' ', '+')}"


# =========================
# UI
# =========================
st.title("🚗 Used Car Price Predictor")

st.markdown("Enter car details to predict price and get recommendations")

# Inputs
km = st.number_input("Kilometers Driven", 0)
seats = st.number_input("Seats", 1)
mileage = st.number_input("Mileage (km/l)", 0.0)
age = st.number_input("Car Age", 0)

fuel = st.selectbox("Fuel Type", fuel_le.classes_)
trans = st.selectbox("Transmission", trans_le.classes_)
owner = st.selectbox("Owner Type", owner_le.classes_)


# =========================
# PREDICTION
# =========================
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

    # =========================
    # RECOMMENDATIONS
    # =========================
    st.markdown("## 🚗 Here are some recommendations for you:")

    recs = recommend_cars(prediction, data)

    if recs.empty:
        st.warning("No similar cars found")
    else:
        for _, row in recs.iterrows():

            # 🔴 IMPORTANT: change column name if needed
            car_name = row.get('Brand','Model')

            link = get_car_link(car_name)

            st.markdown(f"""
---
### 🚘 {car_name}
💰 Price: ₹ {row['Price']} Lakhs  
📍 KM Driven: {row['Kilometers_Driven']}

👉 [Check this car online]({link})
""")
