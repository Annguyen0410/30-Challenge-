import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

data = {
    'Day': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
    'Temperature': [30, 32, 34, 31, 29, 28, 35, 33, 30, 31, 33, 34, 32, 30, 29, 27, 30, 32, 33, 31],
    'Humidity': [60, 62, 64, 58, 55, 57, 65, 63, 59, 61, 63, 65, 62, 60, 58, 56, 59, 61, 63, 60],
    'Wind Speed': [10, 12, 8, 11, 9, 10, 13, 12, 10, 11, 12, 9, 11, 10, 8, 10, 12, 11, 9, 10],
    'Precipitation': [0, 0, 1, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0],
    'Next Day Temperature': [32, 34, 31, 29, 28, 35, 33, 30, 31, 33, 34, 32, 30, 29, 27, 30, 32, 33, 31, 30]
}

df = pd.DataFrame(data)
original_df_for_context = df.copy()

print("Historical Weather Data (First 5 rows):")
print(df.head())

df['Lag1_Temperature'] = df['Temperature'].shift(1)
df['Lag1_Humidity'] = df['Humidity'].shift(1)
df['Interaction_Temp_Hum'] = df['Temperature'] * df['Humidity']
df['Day_Cycle_Sin'] = np.sin(2 * np.pi * df['Day'] / 7)
df['Day_Cycle_Cos'] = np.cos(2 * np.pi * df['Day'] / 7)
df['Rolling_Mean_Temp_3D_Lag1'] = df['Temperature'].shift(1).rolling(window=3, min_periods=1).mean()
df['Rolling_Std_Temp_3D_Lag1'] = df['Temperature'].shift(1).rolling(window=3, min_periods=1).std().fillna(0) # fillna for std of single point
df['Temp_Diff_1D'] = df['Temperature'] - df['Lag1_Temperature']
df['Is_Precipitating_Today'] = (df['Precipitation'] > 0).astype(int)

df.dropna(inplace=True)
df.reset_index(drop=True, inplace=True)

print("\nProcessed Weather Data with Engineered Features (First 5 rows):")
print(df.head())

feature_columns = [
    'Temperature', 'Humidity', 'Wind Speed', 'Precipitation',
    'Lag1_Temperature', 'Lag1_Humidity', 'Interaction_Temp_Hum',
    'Day_Cycle_Sin', 'Day_Cycle_Cos', 'Rolling_Mean_Temp_3D_Lag1',
    'Rolling_Std_Temp_3D_Lag1', 'Temp_Diff_1D', 'Is_Precipitating_Today'
]
X = df[feature_columns]
y = df['Next Day Temperature']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"\nMean Squared Error: {mse:.2f}")
print(f"R^2 Score: {r2:.2f}")

print("\nModel Coefficients:")
coefficients = pd.Series(model.coef_, index=feature_columns)
print(coefficients)

plt.figure(figsize=(12, 7))
plt.plot(y_test.values, label="Actual Temperatures", marker='o', linestyle='-')
plt.plot(y_pred, label="Predicted Temperatures", marker='x', linestyle='--')
plt.title("Actual vs Predicted Temperatures (Test Set)")
plt.xlabel("Test Sample Index")
plt.ylabel("Temperature (°C)")
plt.legend()
plt.grid(True)
plt.show()

today_temperature = 30
today_humidity = 60
today_wind_speed = 10
today_precipitation = 0

last_recorded_day_number = original_df_for_context['Day'].iloc[-1]
todays_day_number = last_recorded_day_number + 1

yesterdays_temperature = original_df_for_context['Temperature'].iloc[-1]
yesterdays_humidity = original_df_for_context['Humidity'].iloc[-1]
day_before_yesterdays_temp = original_df_for_context['Temperature'].iloc[-2]
two_days_before_yesterdays_temp = original_df_for_context['Temperature'].iloc[-3]

temps_for_rolling_lag1 = [yesterdays_temperature, day_before_yesterdays_temp, two_days_before_yesterdays_temp]

new_data_features = {
    'Temperature': [today_temperature],
    'Humidity': [today_humidity],
    'Wind Speed': [today_wind_speed],
    'Precipitation': [today_precipitation],
    'Lag1_Temperature': [yesterdays_temperature],
    'Lag1_Humidity': [yesterdays_humidity],
    'Interaction_Temp_Hum': [today_temperature * today_humidity],
    'Day_Cycle_Sin': [np.sin(2 * np.pi * todays_day_number / 7)],
    'Day_Cycle_Cos': [np.cos(2 * np.pi * todays_day_number / 7)],
    'Rolling_Mean_Temp_3D_Lag1': [np.mean(temps_for_rolling_lag1)],
    'Rolling_Std_Temp_3D_Lag1': [np.std(temps_for_rolling_lag1) if len(temps_for_rolling_lag1) > 1 else 0],
    'Temp_Diff_1D': [today_temperature - yesterdays_temperature],
    'Is_Precipitating_Today': [1 if today_precipitation > 0 else 0]
}

new_data_df = pd.DataFrame(new_data_features)
new_data_df = new_data_df[feature_columns]

predicted_temperature_next_day = model.predict(new_data_df)
print(f"\nPredicted temperature for the next day (given today's conditions): {predicted_temperature_next_day[0]:.2f}°C")

print("\nFeatures used for the new prediction:")
print(new_data_df.transpose())