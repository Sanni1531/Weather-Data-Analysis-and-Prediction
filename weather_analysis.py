# weather_analysis.py

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from datetime import datetime
import numpy as np

# Load dataset
file_path = 'weather.csv'  # Change this to your dataset filename
df = pd.read_csv(file_path)
print("\u2705 Dataset Loaded Successfully")
print(df.head())
print(df.info())

# Parse date and extract features
df['Formatted Date'] = pd.to_datetime(df['Formatted Date'], utc=True)
df.set_index('Formatted Date', inplace=True)
df['Day'] = df.index.day
df['Month'] = df.index.month
df['Year'] = df.index.year

# Drop rows with missing target
df = df.dropna(subset=['Temperature (C)'])

# Define features and target
features = ['Day', 'Month', 'Year', 'Humidity', 'Pressure (millibars)', 'Wind Speed (km/h)']
X = df[features]
y = df['Temperature (C)']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestRegressor()
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
print(f"\U0001F4CA RMSE: {rmse:.2f} ")

# Forecast for future date
future_date = datetime(2025, 7, 16)
future_features = pd.DataFrame([[
    future_date.day,
    future_date.month,
    future_date.year,
    df['Humidity'].mean(),
    df['Pressure (millibars)'].mean(),
    df['Wind Speed (km/h)'].mean()
]], columns=features)

future_temp = model.predict(future_features)[0]
print(f"\U0001F4C5 Predicted Temperature on {future_date.date()}: {future_temp:.2f} °C")

# Plot
plt.figure(figsize=(12, 6))
df['Temperature (C)'].resample('M').mean().plot()
plt.title("Temperature Trends")
plt.ylabel("Temperature (C)")
plt.xlabel("Date")
plt.tight_layout()
plt.show() 