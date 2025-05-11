import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

np.random.seed(42)
dates = pd.date_range(start='2025-01-01', periods=200, freq='B')
base_price = 100
trend_factor = 0.3
seasonality_factor = 15
noise_factor = 8
prices = (base_price +
          np.arange(len(dates)) * trend_factor +
          np.sin(np.arange(len(dates)) * 0.05) * seasonality_factor +
          np.random.randn(len(dates)) * noise_factor)
df = pd.DataFrame({'Date': dates, 'Close': prices})
df['Close'] = np.maximum(10, df['Close'].round(2))


df['TimeIndex'] = np.arange(len(df))
df['Lag1'] = df['Close'].shift(1)
df['Lag2'] = df['Close'].shift(2)
df['MA7'] = df['Close'].rolling(window=7).mean()
df['MA21'] = df['Close'].rolling(window=21).mean()
df['Volatility14'] = df['Close'].rolling(window=14).std()

ema12 = df['Close'].ewm(span=12, adjust=False).mean()
ema26 = df['Close'].ewm(span=26, adjust=False).mean()
df['MACD'] = ema12 - ema26
df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()

delta = df['Close'].diff(1)
gain = delta.where(delta > 0, 0)
loss = -delta.where(delta < 0, 0)
avg_gain14 = gain.rolling(window=14, min_periods=1).mean()
avg_loss14 = loss.rolling(window=14, min_periods=1).mean()
rs = avg_gain14 / avg_loss14
df['RSI14'] = 100 - (100 / (1 + rs))
df['RSI14'] = df['RSI14'].fillna(50)


df.dropna(inplace=True)
df_original_dates = df['Date'].copy()

features = ['TimeIndex', 'Lag1', 'Lag2', 'MA7', 'MA21', 'Volatility14', 'MACD', 'MACD_Signal', 'RSI14']
target = 'Close'

X = df[features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)

train_dates = df_original_dates.loc[X_train.index]
test_dates = df_original_dates.loc[X_test.index]

model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1, max_depth=10, min_samples_split=5, min_samples_leaf=3)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Enhanced Stock Price Prediction Model Evaluation:")
print(f"Mean Squared Error: {mse:.2f}")
print(f"Root Mean Squared Error: {rmse:.2f}")
print(f"Mean Absolute Error: {mae:.2f}")
print(f"R^2 Score: {r2:.2f}")

plt.figure(figsize=(16, 8))
plt.plot(train_dates, y_train, label="Training Data Actual Prices", linestyle='-', color='deepskyblue', alpha=0.6)
plt.plot(test_dates, y_test, label="Test Data Actual Prices", marker='o', markersize=5, linestyle='-', color='forestgreen')
plt.plot(test_dates, y_pred, label="Predicted Stock Prices (Random Forest)", marker='x', markersize=5, linestyle='--', color='crimson')

plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator(minticks=10, maxticks=20))
plt.gcf().autofmt_xdate()

plt.title("Advanced Stock Price Prediction: Actual vs Predicted", fontsize=16)
plt.xlabel("Date", fontsize=12)
plt.ylabel("Stock Price ($)", fontsize=12)
plt.legend(fontsize=10)
plt.grid(True, linestyle=':', alpha=0.7)
plt.tight_layout()
plt.show()

last_known_data_all_features = df.iloc[-1:].copy()
future_date_timestamp = df_original_dates.iloc[-1] + pd.Timedelta(days=1)
while future_date_timestamp.dayofweek >= 5: # Skip weekends
    future_date_timestamp += pd.Timedelta(days=1)


future_data_point = pd.DataFrame(columns=features, index=[0])
future_data_point['TimeIndex'] = last_known_data_all_features['TimeIndex'].iloc[0] + 1
future_data_point['Lag1'] = last_known_data_all_features['Close'].iloc[0]
future_data_point['Lag2'] = last_known_data_all_features['Lag1'].iloc[0]

temp_close_series = pd.concat([df['Close'].iloc[-25:], pd.Series([last_known_data_all_features['Close'].iloc[0]])], ignore_index=True)

future_data_point['MA7'] = temp_close_series.rolling(window=7).mean().iloc[-1]
future_data_point['MA21'] = temp_close_series.rolling(window=21).mean().iloc[-1]
future_data_point['Volatility14'] = temp_close_series.rolling(window=14).std().iloc[-1]


current_close = last_known_data_all_features['Close'].iloc[0]
prev_ema12 = last_known_data_all_features['MACD'].iloc[0] + last_known_data_all_features['MACD_Signal'].iloc[0] # Approximation if direct EMA not stored
prev_ema26 = prev_ema12 - last_known_data_all_features['MACD'].iloc[0] # Approximation

multiplier12 = 2 / (12 + 1)
multiplier26 = 2 / (26 + 1)
next_ema12 = (current_close * multiplier12) + (prev_ema12 * (1 - multiplier12)) # Using last EMA from df for MACD feature
next_ema26 = (current_close * multiplier26) + (prev_ema26 * (1 - multiplier26))
future_data_point['MACD'] = next_ema12 - next_ema26

prev_macd_signal = last_known_data_all_features['MACD_Signal'].iloc[0]
multiplier_signal = 2 / (9 + 1)
future_data_point['MACD_Signal'] = (future_data_point['MACD'] * multiplier_signal) + (prev_macd_signal * (1-multiplier_signal))


last_delta = current_close - last_known_data_all_features['Lag1'].iloc[0]
last_gain = max(last_delta, 0)
last_loss = max(-last_delta, 0)

# Requires historical avg_gain and avg_loss. For simplicity, approximate using recent data or take last values.
# This part is complex to do perfectly without full history or carrying state.
# We use the last known RSI value from the dataframe for demonstration if not enough data points.
if len(df) >= 14 :
    recent_gains = gain.iloc[-13:].tolist() + [last_gain]
    recent_losses = loss.iloc[-13:].tolist() + [last_loss]
    avg_gain_future = np.mean(recent_gains)
    avg_loss_future = np.mean(recent_losses)
    rs_future = avg_gain_future / avg_loss_future if avg_loss_future > 0 else np.inf
    future_data_point['RSI14'] = 100 - (100 / (1 + rs_future if np.isfinite(rs_future) else 10000)) # cap rs_future if infinite
else:
    future_data_point['RSI14'] = last_known_data_all_features['RSI14'].iloc[0]


future_data_point.fillna(method='ffill', inplace=True) # Fallback for any remaining NaNs
future_data_point.fillna(method='bfill', inplace=True) # Fallback for any initial NaNs

predicted_price_future = model.predict(future_data_point[features])
print(f"\nPredicted stock price for {future_date_timestamp.date()}: ${predicted_price_future[0]:.2f}")

feature_importances = pd.Series(model.feature_importances_, index=features).sort_values(ascending=False)
print("\nFeature Importances (Random Forest):")
print(feature_importances)

plt.figure(figsize=(10, 6))
feature_importances.plot(kind='bar', color='skyblue')
plt.title('Feature Importances in Random Forest Model')
plt.ylabel('Importance')
plt.xlabel('Features')
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()