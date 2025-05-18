import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectFromModel
import joblib
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# Load dataset (replace with a real dataset if available)
def load_data():
    """Load or create sample dataset"""
    data = {
        'make': ['Toyota', 'Honda', 'Ford', 'BMW', 'Audi', 'Toyota', 'Honda', 'Ford', 'BMW', 'Audi'],
        'model': ['Corolla', 'Civic', 'F-150', '3 Series', 'A4', 'Camry', 'Accord', 'Mustang', '5 Series', 'A6'],
        'year': [2015, 2017, 2018, 2016, 2015, 2018, 2016, 2015, 2017, 2018],
        'mileage': [50000, 30000, 40000, 60000, 45000, 35000, 55000, 65000, 30000, 20000],
        'fuel_type': ['Gasoline', 'Gasoline', 'Diesel', 'Diesel', 'Gasoline', 'Hybrid', 'Gasoline', 'Gasoline', 'Diesel', 'Gasoline'],
        'transmission': ['Automatic', 'Manual', 'Automatic', 'Automatic', 'Manual', 'Automatic', 'Automatic', 'Manual', 'Automatic', 'Automatic'],
        'engine_size': [1.8, 1.5, 3.5, 2.0, 2.0, 2.5, 2.4, 5.0, 3.0, 3.0],
        'horsepower': [132, 158, 375, 240, 252, 203, 185, 460, 335, 340],
        'num_doors': [4, 4, 2, 4, 4, 4, 4, 2, 4, 4],
        'weight': [2800, 2900, 4500, 3500, 3300, 3200, 3100, 3800, 4000, 3900],
        'price': [15000, 17000, 25000, 27000, 30000, 16000, 18000, 26000, 28000, 31000]
    }
    return pd.DataFrame(data)

def explore_data(df):
    """Exploratory data analysis"""
    print("Dataset information:")
    print(f"Shape: {df.shape}")
    print("\nData types:")
    print(df.dtypes)
    
    print("\nDescriptive statistics:")
    print(df.describe())
    
    print("\nMissing values:")
    print(df.isnull().sum())
    
    # Create correlation matrix visualization
    plt.figure(figsize=(10, 8))
    numeric_df = df.select_dtypes(include=[np.number])
    correlation = numeric_df.corr()
    sns.heatmap(correlation, annot=True, cmap='coolwarm', linewidths=0.5)
    plt.title('Correlation Matrix')
    plt.tight_layout()
    plt.savefig('correlation_matrix.png')
    
    # Feature relationships with target
    num_features = len(numeric_df.columns) - 1  # Excluding price
    plt.figure(figsize=(15, 10))
    plot_idx = 1
    
    for feature in numeric_df.columns:
        if feature != 'price':
            plt.subplot(2, (num_features // 2) + (num_features % 2), plot_idx)
            plt.scatter(df[feature], df['price'], alpha=0.5)
            plt.title(f'{feature} vs price')
            plt.xlabel(feature)
            plt.ylabel('price')
            plot_idx += 1
    
    plt.tight_layout()
    plt.savefig('feature_relationships.png')
    
    # Categorical feature analysis
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        plt.figure(figsize=(10, 6))
        sns.boxplot(x=col, y='price', data=df)
        plt.title(f'Price distribution by {col}')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f'{col}_price_distribution.png')

def feature_engineering(df):
    """Create new features from existing data"""
    # Create a copy to avoid modifying the original dataframe
    df_new = df.copy()
    
    # Calculate car age
    current_year = 2025  # Current year
    df_new['age'] = current_year - df_new['year']
    
    # Create price per mile feature
    df_new['price_per_mile'] = df_new['price'] / df_new['mileage']
    
    # Create luxury brand indicator
    luxury_brands = ['BMW', 'Audi', 'Mercedes', 'Lexus', 'Porsche', 'Tesla']
    df_new['is_luxury'] = df_new['make'].apply(lambda x: 1 if x in luxury_brands else 0)
    
    # Create power-to-weight ratio
    if 'horsepower' in df_new.columns and 'weight' in df_new.columns:
        df_new['power_weight_ratio'] = df_new['horsepower'] / df_new['weight']
    
    # Create combined categorical features
    df_new['make_model'] = df_new['make'] + '_' + df_new['model']
    
    # Log transform of mileage (often has better distribution for modeling)
    df_new['log_mileage'] = np.log1p(df_new['mileage'])
    
    return df_new

def preprocess_data(df, target='price'):
    """Preprocess data for modeling"""
    # Separate features and target
    X = df.drop(target, axis=1)
    y = df[target]
    
    # Identify categorical and numerical columns
    categorical_cols = X.select_dtypes(include=['object']).columns
    numerical_cols = X.select_dtypes(include=[np.number]).columns
    
    # Create preprocessing pipelines
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', RobustScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', pd.get_dummies)
    ])
    
    # Combine preprocessing steps
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_cols),
            ('cat', 'passthrough', categorical_cols)
        ])
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE)
    
    return X_train, X_test, y_train, y_test, preprocessor

def build_models():
    """Build multiple regression models"""
    models = {
        'RandomForest': RandomForestRegressor(random_state=RANDOM_STATE),
        'GradientBoosting': GradientBoostingRegressor(random_state=RANDOM_STATE),
        'ElasticNet': ElasticNet(random_state=RANDOM_STATE)
    }
    
    param_grids = {
        'RandomForest': {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5, 10]
        },
        'GradientBoosting': {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 5, 8]
        },
        'ElasticNet': {
            'alpha': [0.1, 0.5, 1.0],
            'l1_ratio': [0.1, 0.5, 0.9]
        }
    }
    
    return models, param_grids

def train_and_evaluate(X_train, X_test, y_train, y_test, preprocessor, models, param_grids):
    """Train models, tune hyperparameters, and evaluate performance"""
    results = {}
    best_models = {}
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        
        # Create full pipeline with preprocessing
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('model', model)
        ])
        
        # Perform GridSearchCV
        grid_search = GridSearchCV(
            pipeline,
            param_grid={f'model__{key}': value for key, value in param_grids[name].items()},
            cv=5,
            scoring='neg_mean_squared_error',
            n_jobs=-1
        )
        
        # Fit the model
        grid_search.fit(X_train, y_train)
        
        # Get best model
        best_model = grid_search.best_estimator_
        best_models[name] = best_model
        
        # Make predictions
        y_pred = best_model.predict(X_test)
        
        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # Store results
        results[name] = {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'best_params': grid_search.best_params_
        }
        
        # Print results
        print(f"{name} Results:")
        print(f"  MSE: {mse:.2f}")
        print(f"  RMSE: {rmse:.2f}")
        print(f"  MAE: {mae:.2f}")
        print(f"  R²: {r2:.4f}")
        print(f"  Best Parameters: {grid_search.best_params_}")
    
    return results, best_models

def feature_importance(best_model, X):
    """Analyze feature importance for the best model"""
    # Only for tree-based models
    if hasattr(best_model[-1], 'feature_importances_'):
        # Get feature names after preprocessing
        # This is simplified - actual column names would depend on preprocessing
        feature_names = X.columns
        
        # Get feature importances
        importances = best_model[-1].feature_importances_
        
        # Sort importances
        indices = np.argsort(importances)[::-1]
        
        # Plot
        plt.figure(figsize=(12, 8))
        plt.title('Feature Importance')
        plt.bar(range(len(indices)), importances[indices], align='center')
        plt.xticks(range(len(indices)), [feature_names[i] for i in indices], rotation=90)
        plt.tight_layout()
        plt.savefig('feature_importance.png')
        
        # Print importances
        print("\nFeature Importances:")
        for i in indices:
            print(f"{feature_names[i]}: {importances[i]:.4f}")

def save_model(model, filename='best_model.pkl'):
    """Save model to disk"""
    joblib.dump(model, filename)
    print(f"\nModel saved as {filename}")

def create_prediction_function(model, preprocessor, feature_names):
    """Create a function for making predictions on new data"""
    def predict_price(make, model_name, year, mileage, **kwargs):
        # Create a DataFrame with the input data
        input_data = pd.DataFrame({
            'make': [make],
            'model': [model_name],
            'year': [year],
            'mileage': [mileage],
            # Add other features with default values if not provided
            'fuel_type': kwargs.get('fuel_type', 'Gasoline'),
            'transmission': kwargs.get('transmission', 'Automatic'),
            'engine_size': kwargs.get('engine_size', 2.0),
            'horsepower': kwargs.get('horsepower', 200),
            'num_doors': kwargs.get('num_doors', 4),
            'weight': kwargs.get('weight', 3500)
        })
        
        # Apply feature engineering
        input_data = feature_engineering(input_data)
        
        # Make prediction
        prediction = model.predict(input_data)[0]
        
        return prediction
    
    return predict_price

def main():
    # Load data
    df = load_data()
    print("Original dataset:")
    print(df.head())
    
    # Exploratory data analysis
    explore_data(df)
    
    # Feature engineering
    df_engineered = feature_engineering(df)
    print("\nDataset after feature engineering:")
    print(df_engineered.head())
    
    # Preprocess data
    X_train, X_test, y_train, y_test, preprocessor = preprocess_data(df_engineered)
    
    # Build models
    models, param_grids = build_models()
    
    # Train and evaluate models
    results, best_models = train_and_evaluate(X_train, X_test, y_train, y_test, preprocessor, models, param_grids)
    
    # Find best model based on R² score
    best_model_name = max(results, key=lambda x: results[x]['r2'])
    best_model = best_models[best_model_name]
    
    print(f"\nBest model: {best_model_name} with R² = {results[best_model_name]['r2']:.4f}")
    
    # Feature importance analysis
    feature_importance(best_model, X_train)
    
    # Save the best model
    save_model(best_model, f'{best_model_name}_model.pkl')
    
    # Create a prediction function
    predict_price = create_prediction_function(best_model, preprocessor, X_train.columns)
    
    # Example prediction
    example_price = predict_price('Toyota', 'Camry', 2019, 25000, fuel_type='Hybrid')
    print(f"\nExample prediction for 2019 Toyota Camry Hybrid with 25k miles: ${example_price:.2f}")
    
    # Create a simple web interface using Streamlit (code shown but not executed)
    print("\nTo create a web interface, create a file named 'app.py' with the following code:")
    print("""
    import streamlit as st
    import pandas as pd
    import joblib
    
    # Load the model
    model = joblib.load('best_model_name_model.pkl')
    
    # Create the Streamlit app
    st.title('Car Price Prediction App')
    
    col1, col2 = st.columns(2)
    
    with col1:
        make = st.selectbox('Make', ['Toyota', 'Honda', 'Ford', 'BMW', 'Audi'])
        model_name = st.text_input('Model', 'Camry')
        year = st.slider('Year', 2010, 2025, 2020)
        mileage = st.number_input('Mileage', 0, 200000, 30000)
        fuel_type = st.selectbox('Fuel Type', ['Gasoline', 'Diesel', 'Hybrid', 'Electric'])
    
    with col2:
        transmission = st.selectbox('Transmission', ['Automatic', 'Manual'])
        engine_size = st.slider('Engine Size (L)', 1.0, 6.0, 2.0, 0.1)
        horsepower = st.slider('Horsepower', 50, 500, 200)
        num_doors = st.radio('Number of Doors', [2, 4])
        weight = st.slider('Weight (lbs)', 2000, 5000, 3500)
    
    if st.button('Predict Price'):
        # Create input data
        input_data = pd.DataFrame({
            'make': [make],
            'model': [model_name],
            'year': [year],
            'mileage': [mileage],
            'fuel_type': [fuel_type],
            'transmission': [transmission],
            'engine_size': [engine_size],
            'horsepower': [horsepower],
            'num_doors': [num_doors],
            'weight': [weight]
        })
        
        # Feature engineering would be applied here
        
        # Make prediction
        prediction = model.predict(input_data)[0]
        
        st.success(f'Predicted Price: ${prediction:.2f}')
    """)
    
    print("\nRun the app with: streamlit run app.py")

if __name__ == "__main__":
    main()