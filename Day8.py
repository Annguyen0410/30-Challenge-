import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

housing = fetch_california_housing(as_frame=True)
df = housing.frame

print("California Housing Data (First 5 rows):")
print(df.head())
print("\nData Description:")
print(df.describe())
print("\nChecking for Missing Values:")
print(df.isnull().sum())

X_original = df.drop('MedHouseVal', axis=1)
y = df['MedHouseVal']

def engineer_features(data):
    df_engineered = data.copy()
    df_engineered['Rooms_per_Person'] = df_engineered['AveRooms'] / df_engineered['AveOccup']
    df_engineered['Bedrooms_per_Room'] = df_engineered['AveBedrms'] / df_engineered['AveRooms']
    # Replace inf values that might occur if AveRooms or AveOccup is zero, though unlikely for this dataset
    df_engineered.replace([np.inf, -np.inf], np.nan, inplace=True)
    # Fill any NaNs that might have been created (e.g., division by zero if not handled above) or already exist
    # For this dataset, AveRooms/AveOccup are unlikely to be zero, but this is good practice
    if df_engineered.isnull().sum().sum() > 0:
        print("Warning: NaNs detected after feature engineering. Filling with median.")
        for col in df_engineered.columns:
            if df_engineered[col].isnull().any():
                df_engineered[col].fillna(df_engineered[col].median(), inplace=True)
                
    df_engineered['MedInc_Sq'] = df_engineered['MedInc']**2
    df_engineered['Age_Income_Interaction'] = df_engineered['HouseAge'] * df_engineered['MedInc']
    return df_engineered

X_engineered = engineer_features(X_original)

print("\nEngineered Features (First 5 rows of X):")
print(X_engineered.head())

X_train, X_test, y_train, y_test = train_test_split(X_engineered, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Convert scaled arrays back to DataFrames to retain column names for coefficient display
X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=X_engineered.columns)
X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=X_engineered.columns)

model = LinearRegression()
model.fit(X_train_scaled_df, y_train)

y_pred = model.predict(X_test_scaled_df)

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"\nMean Squared Error (MSE): {mse:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"R^2 Score: {r2:.4f}")

print("\nModel Coefficients:")
print(f"Intercept: {model.intercept_:.4f}")

coef_df = pd.DataFrame(model.coef_, X_engineered.columns, columns=['Coefficient'])
print("\nCoefficients for each feature:")
print(coef_df.sort_values(by='Coefficient', ascending=False))

plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
plt.xlabel("Actual Median House Value")
plt.ylabel("Predicted Median House Value")
plt.title("Actual vs. Predicted Median House Value")
plt.grid(True)
plt.show()

residuals = y_test - y_pred
plt.figure(figsize=(10, 6))
sns.histplot(residuals, kde=True, bins=30)
plt.xlabel("Residuals (Actual - Predicted)")
plt.ylabel("Frequency")
plt.title("Distribution of Residuals")
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 6))
plt.scatter(y_pred, residuals, alpha=0.5)
plt.axhline(y=0, color='k', linestyle='--', lw=2)
plt.xlabel("Predicted Median House Value")
plt.ylabel("Residuals")
plt.title("Residual Plot (Predicted vs. Residuals)")
plt.grid(True)
plt.show()

new_data_dict = {
    'MedInc': [5],
    'HouseAge': [30],
    'AveRooms': [6],
    'AveBedrms': [1],
    'Population': [500],
    'AveOccup': [3],
    'Latitude': [34.05],
    'Longitude': [-118.25]
}
new_data_df = pd.DataFrame(new_data_dict)

new_data_engineered = engineer_features(new_data_df)

# Ensure the order of columns in new_data_engineered matches X_train_scaled_df.columns
# This is important if the scaler or model is sensitive (though typically sklearn handles by name for DFs)
# However, scaler.transform expects the same feature set.
# Our engineer_features function ensures this.
# If there was a mismatch, one would reorder: new_data_engineered = new_data_engineered[X_engineered.columns]

new_data_scaled = scaler.transform(new_data_engineered)

predicted_price = model.predict(new_data_scaled)
print(f"\nPredicted house price for the new data: ${predicted_price[0]:,.2f}")

# Example of a more complex model: Ridge Regression
from sklearn.linear_model import Ridge

ridge_model = Ridge(alpha=1.0) # Alpha is the regularization strength
ridge_model.fit(X_train_scaled_df, y_train)
y_pred_ridge = ridge_model.predict(X_test_scaled_df)

ridge_mse = mean_squared_error(y_test, y_pred_ridge)
ridge_rmse = np.sqrt(ridge_mse)
ridge_mae = mean_absolute_error(y_test, y_pred_ridge)
ridge_r2 = r2_score(y_test, y_pred_ridge)

print(f"\n--- Ridge Regression Model ---")
print(f"Mean Squared Error (MSE): {ridge_mse:.4f}")
print(f"Root Mean Squared Error (RMSE): {ridge_rmse:.4f}")
print(f"Mean Absolute Error (MAE): {ridge_mae:.4f}")
print(f"R^2 Score: {ridge_r2:.4f}")

ridge_coef_df = pd.DataFrame(ridge_model.coef_, X_engineered.columns, columns=['Coefficient'])
print("\nRidge Model Coefficients for each feature:")
print(ridge_coef_df.sort_values(by='Coefficient', ascending=False))