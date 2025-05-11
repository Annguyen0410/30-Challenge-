import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report,
    roc_auc_score, roc_curve, auc, precision_recall_curve, average_precision_score
)
import seaborn as sns
import matplotlib.pyplot as plt
import shap # Ensure shap is installed: pip install shap

# 1. Load Data
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
column_names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
df = pd.read_csv(url, names=column_names)

# 2. Initial Data Exploration
print("--- Diabetes Dataset: Initial Head ---")
print(df.head())
print("\n--- Dataset Information ---")
df.info()
print("\n--- Dataset Statistical Description ---")
print(df.describe())

cols_with_zeros_as_missing = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
df[cols_with_zeros_as_missing] = df[cols_with_zeros_as_missing].replace(0, np.nan)

print(f"\n--- Missing values after replacing 0s with NaN ---")
print(df.isnull().sum())

# 3. Exploratory Data Analysis (Visualizations)
plt.figure(figsize=(12, 10))
for i, col in enumerate(df.columns[:-1]):
    plt.subplot(3, 3, i + 1)
    sns.histplot(df[col], kde=True, bins=20)
    plt.title(col)
plt.tight_layout()
plt.suptitle("Distribution of Features", y=1.02, fontsize=16)
plt.show()

plt.figure(figsize=(10, 8))
correlation_matrix = df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
plt.title('Correlation Matrix of Features')
plt.show()

# sns.pairplot(df, hue='Outcome', diag_kind='kde') # Can be time-consuming for many features
# plt.suptitle("Pairplot of Features by Outcome", y=1.02)
# plt.show()

# 4. Data Preprocessing
X = df.drop('Outcome', axis=1)
y = df['Outcome']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

numerical_features = X.columns
preprocessor = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

X_train_processed_df = pd.DataFrame(X_train_processed, columns=numerical_features)
X_test_processed_df = pd.DataFrame(X_test_processed, columns=numerical_features)


# 5. Model Training (Logistic Regression with GridSearchCV)
log_reg_model = LogisticRegression(random_state=42, max_iter=3000)

param_grid = {
    'C': [0.001, 0.01, 0.1, 1, 10, 100],
    'solver': ['liblinear', 'saga'],
    'penalty': ['l1', 'l2', 'elasticnet'], # Saga supports elasticnet
    'l1_ratio': np.linspace(0.1, 0.9, 5) if 'elasticnet' in ['l1', 'l2', 'elasticnet'] else [None] # Only for saga + elasticnet
}

# Adjust param_grid for solvers
param_grid_adjusted = []
for C_val in param_grid['C']:
    for solver_val in param_grid['solver']:
        current_params = {'C': [C_val], 'solver': [solver_val]}
        if solver_val in ['liblinear', 'saga']:
            current_params['penalty'] = ['l1', 'l2']
            if solver_val == 'saga': # Saga also supports elasticnet
                 current_params['penalty'].append('elasticnet')

        if solver_val == 'saga' and 'elasticnet' in current_params.get('penalty', []):
            current_params['l1_ratio'] = np.linspace(0.1, 0.9, 3).tolist() # Reduced for brevity
        else:
            current_params['l1_ratio'] = [None] # l1_ratio only used with elasticnet

        # Filter out invalid combinations for liblinear
        if solver_val == 'liblinear' and 'elasticnet' in current_params.get('penalty', []):
            continue # liblinear does not support elasticnet

        # Remove l1_ratio if penalty is not elasticnet
        if current_params.get('penalty') != ['elasticnet']:
             current_params.pop('l1_ratio', None)

        # Create specific grids
        if solver_val == 'liblinear':
            param_grid_adjusted.append({'C': [C_val], 'solver': ['liblinear'], 'penalty': ['l1', 'l2']})
        elif solver_val == 'saga':
            param_grid_adjusted.append({'C': [C_val], 'solver': ['saga'], 'penalty': ['l1', 'l2']})
            param_grid_adjusted.append({'C': [C_val], 'solver': ['saga'], 'penalty': ['elasticnet'], 'l1_ratio': np.linspace(0.1, 0.9, 3).tolist()})


print("\n--- Starting GridSearchCV for Logistic Regression ---")
grid_search = GridSearchCV(estimator=log_reg_model, param_grid=param_grid_adjusted, cv=5, scoring='roc_auc', n_jobs=-1, verbose=1)
grid_search.fit(X_train_processed, y_train)

best_model = grid_search.best_estimator_
print(f"\nBest Hyperparameters: {grid_search.best_params_}")
print(f"Best ROC AUC score from GridSearchCV: {grid_search.best_score_:.4f}")

# 6. Model Evaluation
y_pred = best_model.predict(X_test_processed)
y_pred_proba = best_model.predict_proba(X_test_processed)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred, target_names=['Non-Diabetic', 'Diabetic'])
roc_auc = roc_auc_score(y_test, y_pred_proba)
avg_precision = average_precision_score(y_test, y_pred_proba)

print(f"\n--- Model Performance on Test Set ---")
print(f"Accuracy: {accuracy * 100:.2f}%")
print(f"ROC AUC Score: {roc_auc:.4f}")
print(f"Average Precision-Recall Score: {avg_precision:.4f}")

print("\nConfusion Matrix:")
print(conf_matrix)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Non-Diabetic', 'Diabetic'], yticklabels=['Non-Diabetic', 'Diabetic'])
plt.title('Confusion Matrix on Test Set')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

print("\nClassification Report:")
print(class_report)

fpr, tpr, thresholds_roc = roc_curve(y_test, y_pred_proba)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()

precision, recall, thresholds_prc = precision_recall_curve(y_test, y_pred_proba)
plt.figure(figsize=(8, 6))
plt.plot(recall, precision, color='blue', lw=2, label=f'Precision-Recall curve (AP = {avg_precision:.2f})')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend(loc="lower left")
plt.show()

# 7. Feature Importance (Coefficients)
if hasattr(best_model, 'coef_'):
    coefficients = best_model.coef_[0]
    feature_importance_df = pd.DataFrame({'Feature': numerical_features, 'Importance': coefficients})
    feature_importance_df = feature_importance_df.reindex(feature_importance_df.Importance.abs().sort_values(ascending=False).index) # Sort by absolute importance

    plt.figure(figsize=(10, 7))
    sns.barplot(x='Importance', y='Feature', data=feature_importance_df, palette='viridis')
    plt.title('Feature Importance (Logistic Regression Coefficients)')
    plt.xlabel('Coefficient Value (Importance)')
    plt.ylabel('Feature')
    plt.tight_layout()
    plt.show()

# 8. Model Explainability (SHAP)
print("\n--- Calculating SHAP values ---")
explainer = shap.LinearExplainer(best_model, X_train_processed_df, feature_perturbation="interventional") # Using DataFrame for feature names
# If X_train_processed is a numpy array, and you want to provide feature names:
# explainer = shap.LinearExplainer(best_model, X_train_processed, feature_names=numerical_features)

shap_values_train = explainer.shap_values(X_train_processed_df)
shap_values_test = explainer.shap_values(X_test_processed_df)


print("\nSHAP Summary Plot (Global Feature Importance):")
shap.summary_plot(shap_values_test, X_test_processed_df, plot_type="bar", show=False)
plt.title("SHAP Global Feature Importance")
plt.show()

shap.summary_plot(shap_values_test, X_test_processed_df, show=False)
plt.title("SHAP Feature Importance and Impact")
plt.show()

if len(X_test_processed_df) > 0:
    print("\nSHAP Explanation for the first test instance:")
    shap.force_plot(explainer.expected_value, shap_values_test[0,:], X_test_processed_df.iloc[0,:], matplotlib=True, show=False)
    plt.title("SHAP Force Plot for First Test Instance")
    plt.show()

    if len(X_test_processed_df) > 1:
      print("\nSHAP Explanation for the second test instance (if diabetic, or just another instance):")
      idx_diabetic_example = y_test[y_test == 1].index[0] if sum(y_test == 1) > 0 else 1
      # Get the corresponding index in X_test_processed_df
      original_X_test_index = X_test.index.get_loc(idx_diabetic_example)

      shap.force_plot(explainer.expected_value, shap_values_test[original_X_test_index,:], X_test_processed_df.iloc[original_X_test_index,:], matplotlib=True, show=False)
      plt.title(f"SHAP Force Plot for Test Instance {original_X_test_index} (Actual: {y_test.iloc[original_X_test_index]})")
      plt.show()


# 9. Prediction on New Data
new_data_dict = {
    'Pregnancies': [6, 1],
    'Glucose': [148, 89],
    'BloodPressure': [72, 66],
    'SkinThickness': [35, 23],
    'Insulin': [0, 94], # Example with 0 that will be imputed
    'BMI': [33.6, 28.1],
    'DiabetesPedigreeFunction': [0.627, 0.167],
    'Age': [50, 21]
}
new_data_df = pd.DataFrame(new_data_dict)
new_data_df_original_cols = new_data_df.copy() # Keep original for display

print("\n--- Predicting on New Data ---")
print("Original New Data:")
print(new_data_df_original_cols)

new_data_df[cols_with_zeros_as_missing] = new_data_df[cols_with_zeros_as_missing].replace(0, np.nan)
new_data_processed = preprocessor.transform(new_data_df)

predicted_outcome_new = best_model.predict(new_data_processed)
predicted_probability_new = best_model.predict_proba(new_data_processed)

results_df = new_data_df_original_cols.copy()
results_df['Predicted_Outcome'] = ['Diabetic' if p == 1 else 'Non-Diabetic' for p in predicted_outcome_new]
results_df['Probability_Diabetic'] = predicted_probability_new[:, 1]

print("\nPredictions for New Data:")
print(results_df)

if len(new_data_processed) > 0 and shap_values_test is not None: # Ensure shap values were calculated
    print("\nSHAP Explanation for the first new data instance:")
    new_data_processed_df = pd.DataFrame(new_data_processed, columns=numerical_features)
    shap_values_new_data = explainer.shap_values(new_data_processed_df)
    shap.force_plot(explainer.expected_value, shap_values_new_data[0,:], new_data_processed_df.iloc[0,:], matplotlib=True, show=False)
    plt.title("SHAP Force Plot for First New Data Instance")
    plt.show()

print("\n--- Analysis Complete ---")