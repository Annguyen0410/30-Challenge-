import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc, precision_recall_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import xgboost as xgb
import joblib
import shap
import warnings
warnings.filterwarnings('ignore')

# Set plot style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("viridis")

# Load and explore the dataset
print("Loading and exploring the dataset...")
df = pd.read_csv('https://raw.githubusercontent.com/IBM/employee-attrition-aif360/main/data/employee-attrition.csv')

# Enhance with exploratory data analysis (EDA)
def explore_data(df):
    """Perform exploratory data analysis and return insights."""
    print(f"Dataset shape: {df.shape}")
    
    # Check for missing values
    missing_values = df.isnull().sum().sum()
    print(f"Missing values: {missing_values}")
    
    # Check class distribution (attrition)
    attrition_counts = df['Attrition'].value_counts()
    print("\nClass distribution:")
    print(attrition_counts)
    print(f"Attrition rate: {attrition_counts[1] / len(df) * 100:.2f}%")
    
    # Plot attrition distribution
    plt.figure(figsize=(8, 5))
    sns.countplot(x='Attrition', data=df)
    plt.title('Employee Attrition Distribution')
    plt.tight_layout()
    plt.savefig('attrition_distribution.png')
    
    # Visualize key features vs attrition
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    
    # Selected features for visualization
    features_to_plot = ['Age', 'MonthlyIncome', 'YearsAtCompany', 
                        'OverTime', 'JobSatisfaction', 'WorkLifeBalance']
    
    for i, feature in enumerate(features_to_plot):
        if df[feature].dtype in ['int64', 'float64']:
            sns.boxplot(x='Attrition', y=feature, data=df, ax=axes[i])
        else:
            sns.countplot(x=feature, hue='Attrition', data=df, ax=axes[i])
        axes[i].set_title(f'{feature} vs Attrition')
    
    plt.tight_layout()
    plt.savefig('key_features_vs_attrition.png')
    
    # Show correlation matrix for numerical columns
    numerical_df = df.select_dtypes(include=['int64', 'float64'])
    plt.figure(figsize=(14, 10))
    correlation = numerical_df.corr()
    mask = np.triu(correlation)
    sns.heatmap(correlation, annot=False, mask=mask, cmap='coolwarm', 
                linewidths=0.5, vmin=-1, vmax=1)
    plt.title('Correlation Matrix')
    plt.tight_layout()
    plt.savefig('correlation_matrix.png')
    
    return attrition_counts[1] / len(df)

# Run EDA
attrition_rate = explore_data(df)

# Feature Engineering
print("\nPerforming feature engineering...")
def engineer_features(df):
    """Create new features that might help predict attrition."""
    df_new = df.copy()
    
    # Convert categorical Attrition to binary
    df_new['Attrition'] = df_new['Attrition'].map({'Yes': 1, 'No': 0})
    
    # Create composite work-related satisfaction score
    satisfaction_cols = ['JobSatisfaction', 'EnvironmentSatisfaction', 
                         'WorkLifeBalance', 'RelationshipSatisfaction']
    df_new['OverallSatisfaction'] = df_new[satisfaction_cols].mean(axis=1)
    
    # Create career growth indicator
    df_new['YearsSincePromotion_Ratio'] = df_new['YearsSinceLastPromotion'] / (df_new['TotalWorkingYears'] + 1)
    
    # Create compensation vs experience ratio
    df_new['CompensationPerExperience'] = df_new['MonthlyIncome'] / (df_new['TotalWorkingYears'] + 1)
    
    # Create overtime stress indicator
    df_new['OvertimeStress'] = ((df_new['OverTime'] == 'Yes') & 
                              (df_new['WorkLifeBalance'] <= 2)).astype(int)
    
    # Create commute stress indicator
    df_new['CommuteStress'] = (df_new['DistanceFromHome'] > 15).astype(int)
    
    # Drop irrelevant columns
    cols_to_drop = ['EmployeeNumber', 'Over18', 'StandardHours', 'EmployeeCount']
    df_new.drop([col for col in cols_to_drop if col in df_new.columns], axis=1, inplace=True)
    
    # Encode categorical variables
    categorical_cols = df_new.select_dtypes(include=['object']).columns
    label_encoder = LabelEncoder()
    
    for column in categorical_cols:
        df_new[column] = label_encoder.fit_transform(df_new[column])
    
    return df_new

# Apply feature engineering
df_processed = engineer_features(df)
print(f"New dataset shape after engineering: {df_processed.shape}")

# Feature selection
print("\nPerforming feature selection...")
def select_features(df):
    """Select the most important features for modeling."""
    # Split data
    X = df.drop('Attrition', axis=1)
    y = df['Attrition']
    
    # Initial feature importance with Random Forest
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X, y)
    
    # Get feature importances
    feature_imp = pd.DataFrame({
        'Feature': X.columns,
        'Importance': rf.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    # Plot top 15 features
    plt.figure(figsize=(12, 8))
    sns.barplot(x='Importance', y='Feature', data=feature_imp.head(15))
    plt.title('Top 15 Feature Importances')
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    
    # Select top features (keep top 80% of cumulative importance)
    cum_importance = feature_imp['Importance'].cumsum()
    feature_imp['Cumulative_Importance'] = cum_importance
    
    # Find features for top 80% importance
    important_features = feature_imp[feature_imp['Cumulative_Importance'] <= 0.80]['Feature'].tolist()
    
    # Make sure we have at least 10 features
    if len(important_features) < 10:
        important_features = feature_imp['Feature'].head(10).tolist()
    
    print(f"Selected {len(important_features)} features for modeling")
    
    # Always include the engineered features in the selection
    engineered_features = ['OverallSatisfaction', 'YearsSincePromotion_Ratio', 
                          'CompensationPerExperience', 'OvertimeStress', 'CommuteStress']
    
    for feature in engineered_features:
        if feature not in important_features and feature in df.columns:
            important_features.append(feature)
    
    return important_features

# Select important features
important_features = select_features(df_processed)
print("Selected features:", important_features)

# Prepare the dataset for modeling
X = df_processed[important_features]
y = df_processed['Attrition']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Model training with cross-validation and class imbalance handling
print("\nTraining and evaluating models...")
def train_and_evaluate_models(X_train, X_test, y_train, y_test):
    """Train and evaluate multiple models with proper handling of class imbalance."""
    # Define models to try
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Random Forest': RandomForestClassifier(random_state=42),
        'XGBoost': xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    }
    
    # Parameters for grid search
    params = {
        'Logistic Regression': {
            'C': [0.01, 0.1, 1, 10],
            'class_weight': [None, 'balanced']
        },
        'Random Forest': {
            'n_estimators': [100, 200],
            'max_depth': [None, 10, 20],
            'class_weight': [None, 'balanced']
        },
        'XGBoost': {
            'n_estimators': [100, 200],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.1, 0.2],
            'scale_pos_weight': [1, 3, 5]  # For handling class imbalance
        }
    }
    
    # Create pipeline with SMOTE for handling class imbalance
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Apply SMOTE to handle class imbalance
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)
    
    # Train all models
    results = {}
    best_models = {}
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        
        # Grid search with cross-validation
        grid_search = GridSearchCV(
            estimator=model,
            param_grid=params[name],
            cv=cv,
            scoring='roc_auc',
            n_jobs=-1
        )
        
        # Fit on resampled data
        grid_search.fit(X_train_resampled, y_train_resampled)
        
        # Get best model
        best_model = grid_search.best_estimator_
        best_models[name] = best_model
        
        # Predict on test set
        y_pred = best_model.predict(X_test_scaled)
        y_pred_proba = best_model.predict_proba(X_test_scaled)[:, 1]
        
        # Evaluate
        accuracy = accuracy_score(y_test, y_pred)
        conf_mat = confusion_matrix(y_test, y_pred)
        class_rep = classification_report(y_test, y_pred)
        
        # Calculate ROC curve and AUC
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        
        # Store results
        results[name] = {
            'model': best_model,
            'params': grid_search.best_params_,
            'accuracy': accuracy,
            'conf_matrix': conf_mat,
            'class_report': class_rep,
            'roc_auc': roc_auc,
            'fpr': fpr,
            'tpr': tpr,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba
        }
        
        print(f"{name} - Best parameters: {grid_search.best_params_}")
        print(f"{name} - Accuracy: {accuracy:.4f}")
        print(f"{name} - ROC AUC: {roc_auc:.4f}")
    
    # Visualize results
    # ROC curves
    plt.figure(figsize=(10, 8))
    for name, result in results.items():
        plt.plot(result['fpr'], result['tpr'], label=f"{name} (AUC = {result['roc_auc']:.3f})")
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves for Different Models')
    plt.legend(loc='lower right')
    plt.savefig('roc_curves.png')
    
    # Confusion matrices
    fig, axes = plt.subplots(1, len(models), figsize=(18, 5))
    for i, (name, result) in enumerate(results.items()):
        sns.heatmap(result['conf_matrix'], annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Stayed', 'Left'], yticklabels=['Stayed', 'Left'], ax=axes[i])
        axes[i].set_title(f'{name} Confusion Matrix')
        axes[i].set_ylabel('Actual')
        axes[i].set_xlabel('Predicted')
    plt.tight_layout()
    plt.savefig('confusion_matrices.png')
    
    # Find best overall model based on ROC AUC
    best_model_name = max(results.items(), key=lambda x: x[1]['roc_auc'])[0]
    print(f"\nBest overall model: {best_model_name}")
    
    return results, best_models, scaler, best_model_name

# Train and evaluate models
results, best_models, scaler, best_model_name = train_and_evaluate_models(X_train, X_test, y_train, y_test)

# Model explainability with SHAP
print("\nExplaining model predictions with SHAP...")
def explain_model(model_name, model, X_train, X_test):
    """Use SHAP to explain model predictions."""
    # Select a sample of training data for SHAP
    X_sample = X_train.sample(min(100, len(X_train)), random_state=42) 
    
    # Create explainer based on model type
    if model_name == 'XGBoost':
        explainer = shap.TreeExplainer(model)
    else:
        explainer = shap.KernelExplainer(model.predict_proba, X_sample)
    
    # Calculate SHAP values
    shap_values = explainer.shap_values(X_test.iloc[:100])  # Use a subset for visualization
    
    # For non-tree models, we need to handle shap_values differently
    if model_name != 'XGBoost':
        shap_values = shap_values[1]  # For binary classification, class 1
    
    # Summary plot
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_test.iloc[:100], plot_type="bar", show=False)
    plt.tight_layout()
    plt.savefig('shap_feature_importance.png')
    
    # Create waterfall plot for a high-risk employee
    # Find an employee with high attrition probability
    if model_name == 'XGBoost':
        y_pred_proba = model.predict_proba(X_test)[:, 1]
    else:
        y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    high_risk_idx = np.argsort(y_pred_proba)[-1]  # Index of highest attrition probability
    
    plt.figure(figsize=(10, 8))
    shap.waterfall_plot(explainer.expected_value if model_name == 'XGBoost' else explainer.expected_value[1], 
                       shap_values[high_risk_idx], X_test.iloc[high_risk_idx], show=False)
    plt.tight_layout()
    plt.savefig('shap_waterfall_high_risk.png')
    
    return explainer

# Explain best model
best_model = best_models[best_model_name]
explainer = explain_model(best_model_name, best_model, X_train, X_test)

# Create employee risk scoring system
print("\nCreating employee risk scoring system...")
def create_risk_scoring_system(model, X_test, feature_names):
    """Create a risk scoring system that HR departments can use."""
    # Make predictions on test set
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Define risk categories
    risk_categories = ['Low', 'Medium', 'High']
    risk_thresholds = [0.3, 0.6]  # Thresholds for risk categories
    
    # Assign risk categories based on probabilities
    risk_scores = np.digitize(y_pred_proba, risk_thresholds, right=True)
    risk_labels = [risk_categories[score] for score in risk_scores]
    
    # Create a dataframe with employee risk scores
    risk_df = pd.DataFrame({
        'Attrition_Probability': y_pred_proba,
        'Risk_Category': risk_labels
    })
    
    # Display risk distribution
    plt.figure(figsize=(8, 6))
    sns.countplot(x='Risk_Category', data=risk_df, palette=['green', 'orange', 'red'])
    plt.title('Employee Attrition Risk Distribution')
    plt.savefig('risk_distribution.png')
    
    # Create recommendations for each risk category
    recommendations = {
        'Low': "Regular engagement activities",
        'Medium': "Schedule skip-level meetings, review compensation",
        'High': "Immediate intervention needed, conduct stay interviews"
    }
    
    # Add recommendations to dataframe
    risk_df['Recommended_Action'] = risk_df['Risk_Category'].map(recommendations)
    
    # Sample of high-risk employees
    high_risk_employees = risk_df[risk_df['Risk_Category'] == 'High'].head(5)
    print("\nSample of high-risk employees:")
    print(high_risk_employees)
    
    return risk_df

# Create risk scoring system
risk_df = create_risk_scoring_system(best_model, X_test, X.columns)

# Build an interactive prediction function
def predict_attrition_risk(employee_data, model, scaler, feature_names):
    """
    Predict attrition risk for a new employee.
    
    Args:
        employee_data: Dictionary with employee attributes
        model: Trained model
        scaler: Fitted scaler
        feature_names: List of feature names used by the model
    
    Returns:
        risk_probability, risk_category
    """
    # Create a DataFrame for the employee
    employee_df = pd.DataFrame([employee_data])
    
    # Ensure all required features are present
    for feature in feature_names:
        if feature not in employee_df.columns:
            employee_df[feature] = 0  # Default value
    
    # Keep only the features used by the model
    employee_df = employee_df[feature_names]
    
    # Scale the features
    employee_scaled = scaler.transform(employee_df)
    
    # Predict attrition probability
    risk_probability = model.predict_proba(employee_scaled)[0, 1]
    
    # Determine risk category
    if risk_probability < 0.3:
        risk_category = "Low"
    elif risk_probability < 0.6:
        risk_category = "Medium"
    else:
        risk_category = "High"
    
    return risk_probability, risk_category

# Example usage of the prediction function
print("\nDemonstrating prediction function with an example employee...")
example_employee = {
    'Age': 35,
    'DistanceFromHome': 20,
    'MonthlyIncome': 5000,
    'TotalWorkingYears': 10,
    'YearsAtCompany': 5,
    'YearsSinceLastPromotion': 3,
    'OverTime': 1,  # Yes
    'JobSatisfaction': 2,  # Low
    'WorkLifeBalance': 2,  # Low
    'EnvironmentSatisfaction': 3,
    'RelationshipSatisfaction': 3,
    # Engineered features will be calculated
    'OverallSatisfaction': 2.5,
    'YearsSincePromotion_Ratio': 3/11,
    'CompensationPerExperience': 5000/11,
    'OvertimeStress': 1,
    'CommuteStress': 1
}

risk_prob, risk_cat = predict_attrition_risk(example_employee, best_model, scaler, important_features)
print(f"Example employee attrition risk: {risk_prob:.2%} ({risk_cat})")

# Save the model pipeline for deployment
print("\nSaving model artifacts for deployment...")
model_artifacts = {
    'model': best_model,
    'scaler': scaler,
    'feature_names': important_features,
    'model_type': best_model_name
}

joblib.dump(model_artifacts, 'employee_attrition_model.pkl')
print("Model saved as 'employee_attrition_model.pkl'")

# Final summary
print("\n=== Employee Attrition Project Summary ===")
print(f"Dataset size: {df.shape[0]} employees, {df.shape[1]} original features")
print(f"Attrition rate: {attrition_rate * 100:.2f}%")
print(f"Best model: {best_model_name}")
print(f"Model accuracy: {results[best_model_name]['accuracy'] * 100:.2f}%")
print(f"Model ROC AUC: {results[best_model_name]['roc_auc']:.4f}")
print("Generated visualizations:")
print("- attrition_distribution.png")
print("- key_features_vs_attrition.png")
print("- correlation_matrix.png")
print("- feature_importance.png")
print("- roc_curves.png")
print("- confusion_matrices.png")
print("- shap_feature_importance.png")
print("- shap_waterfall_high_risk.png")
print("- risk_distribution.png")
print("\nProject includes:")
print("- Feature engineering with 5 new engineered features")
print("- Model comparison between Logistic Regression, Random Forest, and XGBoost")
print("- Hyperparameter tuning with cross-validation")
print("- Class imbalance handling with SMOTE")
print("- Model explainability with SHAP")
print("- Employee risk scoring system")
print("- Interactive prediction function for new employees")
print("- Model deployment artifacts")