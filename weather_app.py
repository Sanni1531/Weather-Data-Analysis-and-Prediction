# app.py

import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt

st.title("🌦️ Weather Temperature Prediction App")

# Load dataset
df = pd.read_csv("weather.csv")
df['Formatted Date'] = pd.to_datetime(df['Formatted Date'], utc=True)
df.set_index('Formatted Date', inplace=True)
df['Day'] = df.index.day
df['Month'] = df.index.month
df['Year'] = df.index.year

# Drop missing
df.dropna(subset=['Temperature (C)'], inplace=True)

# Features and Target
features = ['Day', 'Month', 'Year', 'Humidity', 'Pressure (millibars)', 'Wind Speed (km/h)']
X = df[features]
y = df['Temperature (C)']

# Train model
model = RandomForestRegressor()
model.fit(X, y)

# Sidebar inputs
st.sidebar.header("📅 Select a Future Date")
input_date = st.sidebar.date_input("Date", datetime.today())

# Predict
input_features = pd.DataFrame([[
    input_date.day,
    input_date.month,
    input_date.year,
    df['Humidity'].mean(),
    df['Pressure (millibars)'].mean(),
    df['Wind Speed (km/h)'].mean()
]], columns=features)

prediction = model.predict(input_features)[0]
st.subheader(f"🌡️ Predicted Temperature on {input_date}:")
st.metric(label="Predicted Temperature (°C)", value=f"{prediction:.2f} °C")

# Plot trends
st.subheader("📈 Monthly Average Temperature Trends")
monthly_avg = df['Temperature (C)'].resample('M').mean()
fig, ax = plt.subplots()
monthly_avg.plot(ax=ax)
ax.set_ylabel("Temperature (°C)")
ax.set_xlabel("Date")
ax.set_title("Monthly Temperature Trends")
st.pyplot(fig)
