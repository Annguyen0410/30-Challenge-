import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import KNNImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc, precision_recall_curve
from sklearn.feature_selection import SelectFromModel
from sklearn.decomposition import PCA
from sklearn.calibration import calibration_curve
from imblearn.over_sampling import SMOTE
import shap
import warnings
import joblib
import os
from datetime import datetime

# Configure warning settings
warnings.filterwarnings('ignore')

class HeartDiseasePredictor:
    """
    A comprehensive heart disease prediction system with advanced ML techniques,
    model interpretability, and deployment capabilities.
    """
    
    def __init__(self, data_path, output_folder="model_outputs"):
        """Initialize the heart disease prediction system."""
        self.data_path = data_path
        self.output_folder = output_folder
        
        # Create output folder if it doesn't exist
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        
        # Initialize placeholders
        self.df = None
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.models = {}
        self.best_model = None
        self.feature_names = None
        self.scaler = None
        self.feature_importances = None
        
        # Timestamp for versioning
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
    def load_data(self):
        """Load and display basic information about the dataset."""
        self.df = pd.read_csv(self.data_path)
        print("Dataset Sample:")
        print(self.df.head())
        
        # Display basic information
        print("\nDataset Info:")
        print(f"Number of samples: {self.df.shape[0]}")
        print(f"Number of features: {self.df.shape[1] - 1}")
        
        # Check class distribution
        target_counts = self.df['target'].value_counts()
        print("\nClass Distribution:")
        print(target_counts)
        print(f"Class balance: {target_counts.min() / target_counts.max():.2f}")
        
        # Check for missing values
        missing_values = self.df.isnull().sum()
        print("\nMissing Values:")
        print(missing_values)
        
        return self.df
    
    def explore_data(self):
        """Perform exploratory data analysis with visualizations."""
        # Set up the matplotlib figure
        plt.figure(figsize=(15, 12))
        
        # 1. Distribution of target variable
        plt.subplot(3, 2, 1)
        sns.countplot(x='target', data=self.df)
        plt.title('Distribution of Heart Disease')
        plt.xlabel('Heart Disease (1=Yes, 0=No)')
        
        # 2. Age distribution by target
        plt.subplot(3, 2, 2)
        sns.histplot(data=self.df, x='age', hue='target', kde=True, bins=20)
        plt.title('Age Distribution by Heart Disease Status')
        
        # 3. Correlation matrix of features
        plt.subplot(3, 2, 3)
        corr = self.df.corr()
        sns.heatmap(corr, annot=False, cmap='coolwarm', fmt='.2f')
        plt.title('Correlation Matrix')
        
        # 4. Feature relationships
        plt.subplot(3, 2, 4)
        sns.boxplot(x='target', y='thalach', data=self.df)
        plt.title('Max Heart Rate by Target')
        
        # 5. Age vs. Cholesterol with heart disease status
        plt.subplot(3, 2, 5)
        sns.scatterplot(data=self.df, x='age', y='chol', hue='target', alpha=0.7)
        plt.title('Age vs. Cholesterol by Heart Disease Status')
        
        # 6. Chest pain type distribution by target
        plt.subplot(3, 2, 6)
        sns.countplot(x='cp', hue='target', data=self.df)
        plt.title('Chest Pain Type by Heart Disease Status')
        
        plt.tight_layout()
        plt.savefig(f"{self.output_folder}/eda_plots_{self.timestamp}.png")
        plt.close()
        
        # Create additional feature insights
        plt.figure(figsize=(12, 10))
        
        # Feature importance using correlation with target
        corr_with_target = self.df.corr()['target'].sort_values(ascending=False)
        plt.subplot(2, 1, 1)
        sns.barplot(x=corr_with_target.index[1:], y=corr_with_target.values[1:])
        plt.title('Feature Correlation with Target')
        plt.xticks(rotation=90)
        
        # Gender distribution by heart disease
        plt.subplot(2, 1, 2)
        gender_disease = pd.crosstab(self.df['sex'], self.df['target'])
        gender_disease_pct = gender_disease.div(gender_disease.sum(axis=1), axis=0) * 100
        gender_disease_pct.plot(kind='bar', stacked=False)
        plt.title('Heart Disease Frequency by Gender')
        plt.xlabel('Gender (0=Female, 1=Male)')
        plt.ylabel('Percentage (%)')
        
        plt.tight_layout()
        plt.savefig(f"{self.output_folder}/feature_insights_{self.timestamp}.png")
        plt.close()
        
        print("\nEDA visualizations saved to output folder.")
        
    def create_features(self):
        """
        Create new features that might improve model performance.
        """
        # Make a copy to avoid modifying the original dataframe
        df_features = self.df.copy()
        
        # 1. Age groups (categorize age into meaningful groups)
        df_features['age_group'] = pd.cut(df_features['age'], 
                                        bins=[0, 40, 50, 60, 100], 
                                        labels=['<40', '40-50', '50-60', '>60'])
        
        # 2. Body Mass Index approximation (if height is not available, this is just a proxy)
        if 'weight' in df_features.columns and 'height' in df_features.columns:
            df_features['bmi'] = df_features['weight'] / ((df_features['height']/100) ** 2)
        
        # 3. Blood Pressure Classification
        df_features['bp_category'] = pd.cut(df_features['trestbps'], 
                                          bins=[0, 120, 140, 160, 300], 
                                          labels=['Normal', 'Prehypertension', 'Stage 1', 'Stage 2'])
        
        # 4. Cholesterol level categorization
        df_features['chol_category'] = pd.cut(df_features['chol'], 
                                            bins=[0, 200, 240, 600], 
                                            labels=['Normal', 'Borderline', 'High'])
        
        # 5. Heart rate reserve (uses age and max heart rate)
        df_features['hr_reserve'] = 220 - df_features['age'] - df_features['thalach']
        
        # 6. Resting heart rate efficiency
        df_features['heart_efficiency'] = df_features['thalach'] / df_features['trestbps']
        
        # 7. Advanced age-related risk factor
        df_features['age_risk'] = df_features['age'] * df_features['chol'] / 1000
        
        # 8. Combined risk score based on multiple factors
        df_features['combined_risk'] = (
            df_features['age'] / 10 + 
            df_features['chol'] / 50 + 
            df_features['trestbps'] / 20 + 
            (1 if df_features['fbs'] == 1 else 0) * 2 +
            (1 if df_features['exang'] == 1 else 0) * 2
        )
        
        # Store the enhanced dataframe
        self.df_enhanced = df_features
        
        print("\nFeature Engineering Summary:")
        print(f"Original features: {self.df.shape[1]}")
        print(f"Enhanced features: {self.df_enhanced.shape[1]}")
        print("New features created: age_group, bp_category, chol_category, hr_reserve, heart_efficiency, age_risk, combined_risk")
        
        return self.df_enhanced
    
    def preprocess_data(self, use_enhanced_features=True, handle_imbalance=True):
        """
        Preprocess the data with advanced techniques including handling imbalanced data.
        
        Args:
            use_enhanced_features: Whether to use the engineered features
            handle_imbalance: Whether to apply SMOTE for handling class imbalance
        """
        # Select the appropriate dataframe
        working_df = self.df_enhanced if use_enhanced_features and hasattr(self, 'df_enhanced') else self.df
        
        # Identify categorical and numerical columns
        categorical_cols = working_df.select_dtypes(include=['object', 'category']).columns.tolist()
        numerical_cols = working_df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        
        # Remove the target from feature lists
        if 'target' in numerical_cols:
            numerical_cols.remove('target')
        
        # Separate features and target
        X = working_df.drop('target', axis=1)
        y = working_df['target']
        
        # Store feature names for later use in interpretability
        self.feature_names = X.columns.tolist()
        
        # Split data first to prevent data leakage
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Create preprocessing pipeline
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', Pipeline([
                    ('imputer', KNNImputer(n_neighbors=5)),
                    ('scaler', StandardScaler())
                ]), numerical_cols),
                ('cat', Pipeline([
                    ('onehot', OneHotEncoder(handle_unknown='ignore'))
                ]), categorical_cols)
            ],
            remainder='passthrough'
        )
        
        # Fit and transform the training data
        X_train_processed = preprocessor.fit_transform(X_train)
        X_test_processed = preprocessor.transform(X_test)
        
        # Save the preprocessor for later use in predictions
        self.preprocessor = preprocessor
        
        # Handle class imbalance if needed
        if handle_imbalance:
            smote = SMOTE(random_state=42)
            X_train_processed, y_train = smote.fit_resample(X_train_processed, y_train)
            print("\nApplied SMOTE to handle class imbalance")
            
        # Store the processed datasets
        self.X_train = X_train_processed
        self.X_test = X_test_processed
        self.y_train = y_train
        self.y_test = y_test
        
        # Get transformed feature names (important for interpretability)
        self.get_processed_feature_names(preprocessor, numerical_cols, categorical_cols)
        
        print("\nData Preprocessing Completed:")
        print(f"Training set shape: {X_train_processed.shape}")
        print(f"Testing set shape: {X_test_processed.shape}")
        
        return X_train_processed, X_test_processed, y_train, y_test
    
    def get_processed_feature_names(self, preprocessor, numerical_cols, categorical_cols):
        """Extract feature names after preprocessing for interpretability."""
        # Get transformed feature names
        transformed_feature_names = numerical_cols.copy()
        
        # Add one-hot encoded feature names if available
        if categorical_cols:
            try:
                encoder = preprocessor.named_transformers_['cat'].named_steps['onehot']
                encoded_features = encoder.get_feature_names_out(categorical_cols)
                transformed_feature_names.extend(encoded_features)
            except:
                pass
        
        self.processed_feature_names = transformed_feature_names
    
    def train_models(self):
        """Train multiple machine learning models with hyperparameter tuning."""
        # Initialize models dictionary
        model_configs = {
            'LogisticRegression': {
                'model': LogisticRegression(random_state=42, max_iter=1000),
                'params': {
                    'C': [0.01, 0.1, 1, 10],
                    'penalty': ['l1', 'l2'],
                    'solver': ['liblinear', 'saga']
                }
            },
            'RandomForest': {
                'model': RandomForestClassifier(random_state=42),
                'params': {
                    'n_estimators': [100, 200],
                    'max_depth': [None, 10, 20],
                    'min_samples_split': [2, 5]
                }
            },
            'GradientBoosting': {
                'model': GradientBoostingClassifier(random_state=42),
                'params': {
                    'n_estimators': [100, 200],
                    'learning_rate': [0.05, 0.1],
                    'max_depth': [3, 5]
                }
            },
            'SVM': {
                'model': SVC(random_state=42, probability=True),
                'params': {
                    'C': [0.1, 1, 10],
                    'kernel': ['linear', 'rbf'],
                    'gamma': ['scale', 'auto']
                }
            },
            'NeuralNetwork': {
                'model': MLPClassifier(random_state=42, max_iter=1000),
                'params': {
                    'hidden_layer_sizes': [(50,), (100,), (50, 50)],
                    'activation': ['relu', 'tanh'],
                    'alpha': [0.0001, 0.001]
                }
            }
        }
        
        # Train and evaluate each model
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        results = []
        
        print("\nTraining multiple models with hyperparameter tuning:")
        
        for name, config in model_configs.items():
            print(f"\nTraining {name}...")
            
            # Perform grid search
            grid_search = GridSearchCV(
                estimator=config['model'],
                param_grid=config['params'],
                cv=cv,
                scoring='roc_auc',
                n_jobs=-1
            )
            
            grid_search.fit(self.X_train, self.y_train)
            
            # Store the best model
            best_model = grid_search.best_estimator_
            self.models[name] = best_model
            
            # Evaluate on test set
            y_pred = best_model.predict(self.X_test)
            accuracy = accuracy_score(self.y_test, y_pred)
            
            # Store results
            results.append({
                'model': name,
                'best_params': grid_search.best_params_,
                'best_score': grid_search.best_score_,
                'test_accuracy': accuracy
            })
            
            print(f"  Best parameters: {grid_search.best_params_}")
            print(f"  CV Score (ROC AUC): {grid_search.best_score_:.4f}")
            print(f"  Test Accuracy: {accuracy:.4f}")
        
        # Create ensemble model (voting classifier)
        print("\nCreating ensemble model...")
        estimators = [(name, model) for name, model in self.models.items()]
        
        voting_clf = VotingClassifier(estimators=estimators, voting='soft')
        voting_clf.fit(self.X_train, self.y_train)
        self.models['Ensemble'] = voting_clf
        
        # Evaluate ensemble model
        ensemble_preds = voting_clf.predict(self.X_test)
        ensemble_accuracy = accuracy_score(self.y_test, ensemble_preds)
        
        results.append({
            'model': 'Ensemble',
            'best_params': 'N/A',
            'best_score': 'N/A',
            'test_accuracy': ensemble_accuracy
        })
        
        print(f"Ensemble Model Test Accuracy: {ensemble_accuracy:.4f}")
        
        # Find the best model based on test accuracy
        results_df = pd.DataFrame(results)
        self.model_results = results_df
        best_model_name = results_df.loc[results_df['test_accuracy'].idxmax()]['model']
        self.best_model = self.models[best_model_name]
        
        print(f"\nBest model: {best_model_name} (Test Accuracy: {results_df['test_accuracy'].max():.4f})")
        
        # Save all models
        for name, model in self.models.items():
            joblib.dump(model, f"{self.output_folder}/{name}_model_{self.timestamp}.pkl")
        
        # Save the preprocessor
        joblib.dump(self.preprocessor, f"{self.output_folder}/preprocessor_{self.timestamp}.pkl")
        
        print("\nAll models saved to output folder.")
        
        return self.models, best_model_name
    
    def evaluate_best_model(self):
        """Detailed evaluation of the best model with advanced metrics."""
        if self.best_model is None:
            print("No best model found. Please train models first.")
            return
        
        # Make predictions
        y_pred = self.best_model.predict(self.X_test)
        y_proba = self.best_model.predict_proba(self.X_test)[:, 1]
        
        # Classification report
        print("\nClassification Report:")
        print(classification_report(self.y_test, y_pred))
        
        # Confusion Matrix
        cm = confusion_matrix(self.y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.savefig(f"{self.output_folder}/confusion_matrix_{self.timestamp}.png")
        plt.close()
        
        # ROC Curve
        fpr, tpr, _ = roc_curve(self.y_test, y_proba)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.savefig(f"{self.output_folder}/roc_curve_{self.timestamp}.png")
        plt.close()
        
        # Precision-Recall Curve
        precision, recall, _ = precision_recall_curve(self.y_test, y_proba)
        pr_auc = auc(recall, precision)
        
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, color='blue', lw=2, label=f'PR curve (area = {pr_auc:.2f})')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend(loc="lower left")
        plt.savefig(f"{self.output_folder}/precision_recall_curve_{self.timestamp}.png")
        plt.close()
        
        # Model calibration
        fraction_of_positives, mean_predicted_value = calibration_curve(self.y_test, y_proba, n_bins=10)
        
        plt.figure(figsize=(8, 6))
        plt.plot([0, 1], [0, 1], 'k--', label='Perfectly calibrated')
        plt.plot(mean_predicted_value, fraction_of_positives, 's-', label='Model')
        plt.xlabel('Mean predicted probability')
        plt.ylabel('Fraction of positives')
        plt.title('Calibration Plot')
        plt.legend(loc='best')
        plt.savefig(f"{self.output_folder}/calibration_curve_{self.timestamp}.png")
        plt.close()
        
        print("\nEvaluation metrics and visualizations saved to output folder.")
    
    def explain_model(self):
        """Explain model predictions using SHAP values for interpretability."""
        if self.best_model is None:
            print("No best model found. Please train models first.")
            return
        
        # For tree-based models, use the TreeExplainer
        if hasattr(self.best_model, 'feature_importances_'):
            # Get feature importances
            importances = self.best_model.feature_importances_
            
            # Create feature importance plot
            plt.figure(figsize=(12, 8))
            feature_names = self.processed_feature_names if hasattr(self, 'processed_feature_names') else range(self.X_train.shape[1])
            
            # Sort features by importance
            indices = np.argsort(importances)[::-1]
            
            # Plot feature importances
            plt.bar(range(len(indices[:20])), importances[indices[:20]])
            plt.xticks(range(len(indices[:20])), [feature_names[i] if i < len(feature_names) else f"Feature {i}" for i in indices[:20]], rotation=90)
            plt.title('Feature Importances')
            plt.tight_layout()
            plt.savefig(f"{self.output_folder}/feature_importance_{self.timestamp}.png")
            plt.close()
            
            # Store feature importances for later use
            self.feature_importances = {
                'names': [feature_names[i] if i < len(feature_names) else f"Feature {i}" for i in indices],
                'values': importances[indices]
            }
        
        # SHAP values for model interpretability
        try:
            # Use a subset of the test data for SHAP analysis to keep computation manageable
            X_shap = self.X_test[:100]
            
            # For different model types, use appropriate explainers
            if isinstance(self.best_model, (RandomForestClassifier, GradientBoostingClassifier)):
                explainer = shap.TreeExplainer(self.best_model)
                shap_values = explainer.shap_values(X_shap)
                
                # For binary classification, shap_values is a list with values for each class
                if isinstance(shap_values, list):
                    shap_values = shap_values[1]  # Take values for the positive class
                
                # Summary plot
                plt.figure(figsize=(10, 8))
                shap.summary_plot(shap_values, X_shap, plot_type='bar', show=False)
                plt.tight_layout()
                plt.savefig(f"{self.output_folder}/shap_summary_{self.timestamp}.png")
                plt.close()
                
                # Detailed SHAP dependence plots for top features
                feature_names = self.processed_feature_names if hasattr(self, 'processed_feature_names') else range(X_shap.shape[1])
                mean_abs_shap = np.abs(shap_values).mean(0)
                top_indices = np.argsort(mean_abs_shap)[-5:]  # Top 5 features
                
                for i in top_indices:
                    plt.figure(figsize=(8, 6))
                    feature_name = feature_names[i] if i < len(feature_names) else f"Feature {i}"
                    shap.dependence_plot(i, shap_values, X_shap, show=False)
                    plt.title(f'SHAP Dependence for {feature_name}')
                    plt.tight_layout()
                    plt.savefig(f"{self.output_folder}/shap_dependence_{feature_name}_{self.timestamp}.png")
                    plt.close()
            
            else:
                # For other model types, use KernelExplainer (slower but more general)
                explainer = shap.KernelExplainer(self.best_model.predict_proba, X_shap)
                shap_values = explainer.shap_values(X_shap[:20])  # Reduced sample for computation
                
                # Summary plot for positive class
                plt.figure(figsize=(10, 8))
                shap.summary_plot(shap_values[1], X_shap[:20], show=False)
                plt.tight_layout()
                plt.savefig(f"{self.output_folder}/shap_summary_{self.timestamp}.png")
                plt.close()
            
            print("\nModel interpretability visualizations saved to output folder.")
            
        except Exception as e:
            print(f"Could not generate SHAP explanations due to: {e}")
            print("Continuing with other analyses...")
    
    def make_predictions(self, new_data):
        """
        Make predictions on new data.
        
        Args:
            new_data: DataFrame containing new patient data
        
        Returns:
            Dictionary with prediction results and risk probabilities
        """
        if self.best_model is None:
            print("No model available. Please train models first.")
            return None
        
        # Preprocess new data
        new_data_processed = self.preprocessor.transform(new_data)
        
        # Make prediction
        prediction = self.best_model.predict(new_data_processed)[0]
        probability = self.best_model.predict_proba(new_data_processed)[0][1]
        
        # Define risk levels
        risk_level = "Low"
        if probability > 0.75:
            risk_level = "High"
        elif probability > 0.5:
            risk_level = "Moderate"
        elif probability > 0.25:
            risk_level = "Low-Moderate"
        
        # Generate recommendations based on risk factors
        recommendations = []
        if 'age' in new_data and new_data['age'].values[0] > 50:
            recommendations.append("Regular cardiovascular check-ups recommended due to age factor.")
        
        if 'chol' in new_data and new_data['chol'].values[0] > 200:
            recommendations.append("Consider cholesterol management strategies.")
        
        if 'trestbps' in new_data and new_data['trestbps'].values[0] > 130:
            recommendations.append("Monitor blood pressure regularly.")
        
        if 'fbs' in new_data and new_data['fbs'].values[0] == 1:
            recommendations.append("Manage blood sugar levels.")
        
        # Add general recommendations
        recommendations.append("Maintain a heart-healthy diet rich in fruits, vegetables, and whole grains.")
        recommendations.append("Engage in regular physical activity as approved by your healthcare provider.")
        
        # Create result dictionary
        result = {
            'prediction': int(prediction),
            'diagnosis': "At Risk of Heart Disease" if prediction == 1 else "No Heart Disease",
            'probability': float(probability),
            'risk_level': risk_level,
            'recommendations': recommendations
        }
        
        return result
    
    def run_full_pipeline(self, new_patient_data=None):
        """
        Run the complete heart disease prediction pipeline from data loading to evaluation.
        
        Args:
            new_patient_data: Optional DataFrame for prediction on new data
        """
        print("=== Heart Disease Prediction System ===")
        print("Starting full pipeline execution...\n")
        
        # Step 1: Load data
        self.load_data()
        
        # Step 2: Exploratory Data Analysis
        self.explore_data()
        
        # Step 3: Feature Engineering
        self.create_features()
        
        # Step 4: Preprocess data
        self.preprocess_data(use_enhanced_features=True, handle_imbalance=True)
        
        # Step 5: Train models
        self.train_models()
        
        # Step 6: Evaluate best model
        self.evaluate_best_model()
        
        # Step 7: Model interpretability
        self.explain_model()
        
        # Step 8: Make predictions on new data (if provided)
        if new_patient_data is not None:
            result = self.make_predictions(new_patient_data)
            print("\n=== Prediction Results ===")
            print(f"Diagnosis: {result['diagnosis']}")
            print(f"Risk Probability: {result['probability']:.2f}")
            print(f"Risk Level: {result['risk_level']}")
            print("\nRecommendations:")
            for rec in result['recommendations']:
                print(f"- {rec}")
        
        print("\nPipeline execution completed successfully.")
        return self

# Example execution
if __name__ == "__main__":
    # Initialize the predictor
    heart_predictor = HeartDiseasePredictor('heart_disease.csv')
    
    # Run the full pipeline
    heart_predictor.run_full_pipeline()
    
    # Make predictions on new patient data
    new_patient = pd.DataFrame({
        'age': [58],
        'sex': [1],
        'cp': [2],
        'trestbps': [140],
        'chol': [250],
        'fbs': [1],
        'restecg': [0],
        'thalach': [140],
        'exang': [0],
        'oldpeak': [1.2],
        'slope': [1],
        'ca': [0],
        'thal': [2]
    })
    
    result = heart_predictor.make_predictions(new_patient)
    print("\n=== Example Patient Prediction ===")
    print(f"Diagnosis: {result['diagnosis']}")
    print(f"Risk Probability: {result['probability']:.2f}")
    print(f"Risk Level: {result['risk_level']}")