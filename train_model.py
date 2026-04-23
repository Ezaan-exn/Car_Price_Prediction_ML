# =========================
# 1. IMPORT LIBRARIES
# =========================
import pandas as pd
import numpy as np
import pickle
from datetime import datetime

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score


# =========================
# 2. LOAD DATA
# =========================
df = pd.read_excel("Car4u_used_cars_data.xlsx")


# =========================
# 3. FEATURE ENGINEERING
# =========================
df['car_age'] = datetime.now().year - df['Year']


# =========================
# 4. SELECT IMPORTANT COLUMNS
# =========================
df = df[['Kilometers_Driven', 'Seats', 'mileage_num',
         'Fuel_Type', 'Transmission', 'Owner_Type',
         'car_age', 'Price']]


# =========================
# 5. HANDLE MISSING VALUES
# =========================
df = df.dropna()


# =========================
# 6. ENCODE CATEGORICAL DATA (SAVE ENCODERS)
# =========================
fuel_le = LabelEncoder()
trans_le = LabelEncoder()
owner_le = LabelEncoder()

df['Fuel_Type'] = fuel_le.fit_transform(df['Fuel_Type'])
df['Transmission'] = trans_le.fit_transform(df['Transmission'])
df['Owner_Type'] = owner_le.fit_transform(df['Owner_Type'])


# =========================
# 7. SPLIT DATA
# =========================
X = df.drop("Price", axis=1)
y = df["Price"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# =========================
# 8. TRAIN MODEL
# =========================
model = RandomForestRegressor(
    n_estimators=100,
    max_depth=10,
    random_state=42
)

model.fit(X_train, y_train)


# =========================
# 9. EVALUATE MODEL
# =========================
y_pred = model.predict(X_test)
score = r2_score(y_test, y_pred)

print("Model R2 Score:", score)


# =========================
# 10. SAVE MODEL + ENCODERS
# =========================
pickle.dump(model, open("car_price_model.pkl", "wb"))
pickle.dump(fuel_le, open("fuel_encoder.pkl", "wb"))
pickle.dump(trans_le, open("trans_encoder.pkl", "wb"))
pickle.dump(owner_le, open("owner_encoder.pkl", "wb"))

print("✅ Model and encoders saved successfully")
