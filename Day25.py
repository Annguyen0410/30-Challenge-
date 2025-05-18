
"""
Enhanced Customer Churn Prediction Model
- Added feature engineering and selection
- Implemented multiple models for comparison
- Added hyperparameter tuning
- Added model evaluation with cross-validation
- Added model interpretation with SHAP
- Added business insights extraction
- Added visualization of results
- Added model persistence
- Added MLflow for experiment tracking
"""

# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectFromModel, RFECV
from sklearn.impute import KNNImputer
from sklearn.decomposition import PCA
import shap
import joblib
import warnings
import mlflow
import mlflow.sklearn

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Set up MLflow tracking
mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("customer_churn_prediction")

# Helper function for visualization
def plot_distribution(df, column, target='churn'):
    plt.figure(figsize=(12, 5))
    
    # Distribution plot
    plt.subplot(1, 2, 1)
    sns.histplot(data=df, x=column, hue=target, multiple="stack", kde=True)
    plt.title(f'Distribution of {column} by Churn Status')
    
    # Box plot
    plt.subplot(1, 2, 2)
    sns.boxplot(data=df, x=target, y=column)
    plt.title(f'Box Plot of {column} by Churn Status')
    
    plt.tight_layout()
    plt.show()

# Helper function for categorical feature analysis
def plot_categorical_analysis(df, column, target='churn'):
    plt.figure(figsize=(10, 6))
    
    # Calculate percentages
    churn_pct = pd.crosstab(df[column], df[target], normalize='index') * 100
    
    # Plot
    ax = churn_pct[1].sort_values(ascending=False).plot(kind='bar', color='salmon')
    plt.title(f'Churn Rate by {column}')
    plt.ylabel('Churn Rate (%)')
    plt.xlabel(column)
    
    # Add percentage labels on top of bars
    for i, p in enumerate(ax.patches):
        ax.annotate(f'{p.get_height():.1f}%', 
                    (p.get_x() + p.get_width() / 2., p.get_height()), 
                    ha='center', va='center', 
                    xytext=(0, 10), 
                    textcoords='offset points')
    
    plt.tight_layout()
    plt.show()

# Custom class for feature engineering
class FeatureEngineer:
    def __init__(self):
        pass
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X_new = X.copy()
        
        # Create ratio features
        if 'monthly_charges' in X_new.columns and 'total_charges' in X_new.columns:
            # Average tenure in months
            X_new['avg_monthly_charges'] = X_new['total_charges'] / X_new['tenure'].replace(0, 1)
            
            # Charge to tenure ratio
            X_new['charge_tenure_ratio'] = X_new['monthly_charges'] / X_new['tenure'].replace(0, 1)
        
        # Create interaction features
        if 'tenure' in X_new.columns and 'monthly_charges' in X_new.columns:
            X_new['tenure_charges_interaction'] = X_new['tenure'] * X_new['monthly_charges']
        
        # Segment customers based on tenure
        if 'tenure' in X_new.columns:
            X_new['tenure_group'] = pd.cut(X_new['tenure'], 
                                          bins=[0, 12, 24, 36, 48, 60, 72], 
                                          labels=['0-1 year', '1-2 years', '2-3 years', 
                                                 '3-4 years', '4-5 years', '5+ years'])
            # One-hot encode the tenure group
            X_new = pd.get_dummies(X_new, columns=['tenure_group'], drop_first=True)
        
        # Service usage intensity (count of services used)
        service_columns = [col for col in X_new.columns if col.startswith('service_')]
        if service_columns:
            X_new['service_count'] = X_new[service_columns].sum(axis=1)
        
        return X_new

# Load the dataset
print("Loading dataset...")
df = pd.read_csv('customer_churn.csv')
print(f"Dataset loaded with {df.shape[0]} rows and {df.shape[1]} columns")

# Exploratory Data Analysis
print("\n=== Exploratory Data Analysis ===")
print("\nDataset Sample:")
print(df.head())

print("\nDataset Information:")
print(df.info())

print("\nStatistical Summary:")
print(df.describe())

print("\nMissing Values:")
print(df.isnull().sum())

print("\nTarget Distribution:")
if 'churn' in df.columns:
    print(df['churn'].value_counts(normalize=True) * 100)

# Visualize the data
print("\n=== Data Visualization ===")
print("Generating key visualizations...")

# Correlation heatmap for numerical features
numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
plt.figure(figsize=(12, 10))
corr_matrix = df[numeric_cols].corr()
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', cmap='coolwarm', 
            linewidths=0.5, vmin=-1, vmax=1)
plt.title('Correlation Heatmap of Numerical Features')
plt.tight_layout()
plt.show()

# Data Preprocessing
print("\n=== Data Preprocessing ===")

# Handle missing values
print("Handling missing values...")
# For numerical features: Use KNN imputation instead of simple dropping
num_imputer = KNNImputer(n_neighbors=5)
df_numeric = df.select_dtypes(include=['int64', 'float64'])
df[df_numeric.columns] = num_imputer.fit_transform(df_numeric)

# For categorical features: Use mode imputation
cat_cols = df.select_dtypes(include=['object']).columns
for col in cat_cols:
    df[col].fillna(df[col].mode()[0], inplace=True)

# Convert categorical columns to numeric using one-hot encoding
print("Encoding categorical features...")
df_encoded = pd.get_dummies(df, columns=['gender', 'contract_type', 'payment_method'], drop_first=True)

# Feature Engineering
print("Applying feature engineering...")
feature_engineer = FeatureEngineer()
df_engineered = feature_engineer.transform(df_encoded)

# Define features (X) and target (y)
X = df_engineered.drop('churn', axis=1)  # Features
y = df_engineered['churn']               # Target variable

# Feature selection using Random Forest importance
print("Performing feature selection...")
selector = SelectFromModel(RandomForestClassifier(n_estimators=100, random_state=42), threshold='median')
selector.fit(X, y)
selected_features = X.columns[selector.get_support()]
print(f"Selected {len(selected_features)} features out of {X.shape[1]}")
X_selected = selector.transform(X)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=42, stratify=y)

# Prepare scalers
standard_scaler = StandardScaler()
robust_scaler = RobustScaler()

# Model Development and Evaluation
print("\n=== Model Development and Evaluation ===")
print("Training and evaluating multiple models...")

# Function to evaluate models
def evaluate_model(model, X_train, X_test, y_train, y_test, model_name):
    # Start MLflow run
    with mlflow.start_run(run_name=model_name):
        # Train the model
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
        conf_matrix = confusion_matrix(y_test, y_pred)
        
        # ROC curve
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        
        # Precision-Recall curve
        precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
        pr_auc = auc(recall, precision)
        
        # Cross-validation
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
        
        # Log parameters
        if hasattr(model, 'get_params'):
            params = model.get_params()
            for param, value in params.items():
                try:
                    mlflow.log_param(param, value)
                except:
                    pass
        
        # Log metrics
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", report['1']['precision'])
        mlflow.log_metric("recall", report['1']['recall'])
        mlflow.log_metric("f1_score", report['1']['f1-score'])
        mlflow.log_metric("roc_auc", roc_auc)
        mlflow.log_metric("pr_auc", pr_auc)
        mlflow.log_metric("cv_mean_accuracy", np.mean(cv_scores))
        
        # Log the model
        mlflow.sklearn.log_model(model, model_name)
        
        # Print results
        print(f"\n{model_name} Results:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"ROC AUC: {roc_auc:.4f}")
        print(f"PR AUC: {pr_auc:.4f}")
        print(f"CV Mean Accuracy: {np.mean(cv_scores):.4f} (±{np.std(cv_scores):.4f})")
        print(f"Classification Report:\n")
        print(classification_report(y_test, y_pred))
        
        # Plot confusion matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['Not Churned', 'Churned'],
                    yticklabels=['Not Churned', 'Churned'])
        plt.title(f'Confusion Matrix - {model_name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.show()
        
        # Plot ROC curve
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {model_name}')
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.show()
        
        return model, accuracy, roc_auc

# 1. Logistic Regression with Grid Search
print("\nTraining Logistic Regression with Grid Search...")
param_grid_lr = {
    'C': [0.01, 0.1, 1, 10, 100],
    'penalty': ['l1', 'l2'],
    'solver': ['liblinear', 'saga']
}
lr_model = GridSearchCV(
    LogisticRegression(random_state=42, max_iter=1000),
    param_grid_lr,
    cv=5,
    scoring='roc_auc',
    n_jobs=-1
)
lr_model, lr_accuracy, lr_auc = evaluate_model(lr_model, X_train, X_test, y_train, y_test, "Logistic_Regression")

# 2. Random Forest with Grid Search
print("\nTraining Random Forest with Grid Search...")
param_grid_rf = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}
rf_model = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid_rf,
    cv=5,
    scoring='roc_auc',
    n_jobs=-1
)
rf_model, rf_accuracy, rf_auc = evaluate_model(rf_model, X_train, X_test, y_train, y_test, "Random_Forest")

# 3. Gradient Boosting with Grid Search
print("\nTraining Gradient Boosting with Grid Search...")
param_grid_gb = {
    'n_estimators': [100, 200],
    'learning_rate': [0.01, 0.1],
    'max_depth': [3, 5]
}
gb_model = GridSearchCV(
    GradientBoostingClassifier(random_state=42),
    param_grid_gb,
    cv=5,
    scoring='roc_auc',
    n_jobs=-1
)
gb_model, gb_accuracy, gb_auc = evaluate_model(gb_model, X_train, X_test, y_train, y_test, "Gradient_Boosting")

# Model Comparison
print("\n=== Model Comparison ===")
models = ['Logistic Regression', 'Random Forest', 'Gradient Boosting']
accuracies = [lr_accuracy, rf_accuracy, gb_accuracy]
aucs = [lr_auc, rf_auc, gb_auc]

plt.figure(figsize=(12, 6))
x = np.arange(len(models))
width = 0.35

ax = plt.subplot(1, 2, 1)
ax.bar(x - width/2, accuracies, width, label='Accuracy')
ax.bar(x + width/2, aucs, width, label='ROC AUC')
ax.set_xticks(x)
ax.set_xticklabels(models)
ax.set_ylim([0, 1])
ax.set_ylabel('Score')
ax.set_title('Model Performance Comparison')
ax.legend()

# Find best model based on AUC
best_model_idx = np.argmax(aucs)
best_model_name = models[best_model_idx]
print(f"\nBest performing model: {best_model_name} with ROC AUC: {aucs[best_model_idx]:.4f}")

# Select the best model based on AUC
if best_model_idx == 0:
    best_model = lr_model
elif best_model_idx == 1:
    best_model = rf_model
else:
    best_model = gb_model

# Model Interpretation with SHAP
print("\n=== Model Interpretation with SHAP ===")
print("Calculating SHAP values for model interpretation...")

# Create SHAP explainer for the best model
if best_model_idx == 0:  # Logistic Regression
    explainer = shap.LinearExplainer(best_model.best_estimator_, X_train)
else:  # Tree-based models
    explainer = shap.TreeExplainer(best_model.best_estimator_)

# Calculate SHAP values
shap_values = explainer.shap_values(X_test)

# Convert to appropriate format if needed
if isinstance(shap_values, list):
    shap_values = shap_values[1]  # For binary classification, use class 1 (churn)

# Plot SHAP summary plot
plt.figure(figsize=(12, 8))
shap.summary_plot(shap_values, X_test, feature_names=selected_features, show=False)
plt.title('Feature Importance (SHAP Values)')
plt.tight_layout()
plt.show()

# Plot SHAP dependence plots for top features
top_features_idx = np.argsort(-np.abs(shap_values).mean(0))[:3]
for i in top_features_idx:
    plt.figure(figsize=(10, 6))
    shap.dependence_plot(i, shap_values, X_test, feature_names=selected_features, show=False)
    plt.title(f'SHAP Dependence Plot for {selected_features[i]}')
    plt.tight_layout()
    plt.show()

# Business Insights & Recommendations
print("\n=== Business Insights & Recommendations ===")
print("Extracting business insights from the model...")

# Identify top features influencing churn
feature_importance = np.abs(shap_values).mean(0)
feature_importance_df = pd.DataFrame({
    'Feature': selected_features,
    'Importance': feature_importance
}).sort_values('Importance', ascending=False)

print("\nTop 10 Features Influencing Customer Churn:")
print(feature_importance_df.head(10))

# Business recommendations based on model findings
print("\nBusiness Recommendations:")
print("1. Focus on improving customer satisfaction in the areas most strongly "
      "associated with churn based on the SHAP analysis.")
print("2. Develop targeted retention campaigns for customer segments with the highest churn risk.")
print("3. Consider adjusting pricing strategies for high-risk customers.")
print("4. Improve service quality in areas that contribute most to churn.")
print("5. Implement a customer health score based on the model to proactively address potential churn.")

# Save the best model
print("\n=== Model Persistence ===")
model_filename = f'customer_churn_{best_model_name.lower().replace(" ", "_")}_model.pkl'
print(f"Saving the best model to {model_filename}...")
joblib.dump(best_model, model_filename)

# Save feature names and scaler
feature_filename = 'customer_churn_feature_names.pkl'
scaler_filename = 'customer_churn_scaler.pkl'
joblib.dump(selected_features, feature_filename)
joblib.dump(standard_scaler, scaler_filename)

print("\n=== Project Complete ===")
print(f"Model successfully trained and saved. Best model: {best_model_name}")
print("The enhanced churn prediction model is now ready for deployment!")