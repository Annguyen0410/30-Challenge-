import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import string
import nltk
import pickle
import time
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, roc_auc_score, precision_recall_curve
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV
from wordcloud import WordCloud
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# Download necessary NLTK resources
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

class FakeNewsDetector:
    """
    A comprehensive fake news detection system that includes:
    - Data exploration and visualization
    - Text preprocessing and feature engineering
    - Multiple classification models with hyperparameter tuning
    - Performance evaluation and model comparison
    - Model persistence and deployment utilities
    """
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.models = {}
        self.best_model = None
        self.vectorizer = None
        self.lemmatizer = WordNetLemmatizer()
        self.stemmer = PorterStemmer()
        self.stop_words = set(stopwords.words('english'))
        
    def load_data(self, filepath):
        """Load the dataset from a CSV file"""
        try:
            df = pd.read_csv(filepath)
            print(f"Dataset loaded successfully with {df.shape[0]} rows and {df.shape[1]} columns.")
            return df
        except Exception as e:
            print(f"Error loading dataset: {e}")
            return None
    
    def explore_data(self, df):
        """Perform exploratory data analysis on the dataset"""
        # Basic dataset info
        print("\n===== Dataset Information =====")
        print(f"Shape: {df.shape}")
        print("\nColumns:")
        print(df.columns.tolist())
        print("\nMissing values:")
        print(df.isnull().sum())
        
        # Class distribution
        plt.figure(figsize=(8, 6))
        class_dist = df['label'].value_counts()
        class_dist.plot(kind='bar', color=['green', 'red'])
        plt.title('Distribution of News Classes')
        plt.xlabel('Class')
        plt.ylabel('Count')
        plt.xticks(rotation=0)
        plt.tight_layout()
        plt.savefig('class_distribution.png')
        plt.close()
        
        print("\nClass distribution:")
        print(class_dist)
        
        # Text length analysis
        df['text_length'] = df['text'].apply(len)
        plt.figure(figsize=(10, 6))
        sns.histplot(data=df, x='text_length', hue='label', bins=50, alpha=0.7)
        plt.title('Distribution of Text Lengths by Class')
        plt.xlabel('Text Length')
        plt.ylabel('Count')
        plt.tight_layout()
        plt.savefig('text_length_distribution.png')
        plt.close()
        
        # Summary statistics of text length by class
        print("\nText length statistics by class:")
        print(df.groupby('label')['text_length'].describe())
        
        # Generate word clouds for each class
        for label in df['label'].unique():
            self._generate_word_cloud(df[df['label'] == label]['text'], label)
        
        # Most common words by class
        self._most_common_words(df)
        
        return df
    
    def _generate_word_cloud(self, texts, label):
        """Generate and save a word cloud for a specific class"""
        text = ' '.join(texts)
        wordcloud = WordCloud(
            width=800, 
            height=400, 
            background_color='white', 
            max_words=200, 
            collocations=False,
            random_state=self.random_state
        ).generate(text)
        
        plt.figure(figsize=(10, 6))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title(f'Word Cloud for {label.capitalize()} News')
        plt.tight_layout()
        plt.savefig(f'wordcloud_{label}.png')
        plt.close()
    
    def _most_common_words(self, df, n=20):
        """Find most common words for each class"""
        for label in df['label'].unique():
            # Combine all text for this class
            text = ' '.join(df[df['label'] == label]['text'])
            
            # Tokenize and clean
            words = word_tokenize(text.lower())
            words = [word for word in words if word.isalpha() and word not in self.stop_words]
            
            # Count and get most common
            counter = Counter(words)
            most_common = counter.most_common(n)
            
            # Plot
            plt.figure(figsize=(12, 6))
            words, counts = zip(*most_common)
            plt.barh(range(len(words)), counts, align='center')
            plt.yticks(range(len(words)), words)
            plt.title(f'Top {n} Most Common Words in {label.capitalize()} News')
            plt.xlabel('Count')
            plt.tight_layout()
            plt.savefig(f'common_words_{label}.png')
            plt.close()
    
    def preprocess_text(self, df, text_column='text'):
        """Preprocess text data with advanced cleaning techniques"""
        print("\n===== Text Preprocessing =====")
        
        # Create a copy to avoid modifying the original
        processed_df = df.copy()
        
        # Add additional text-based features
        processed_df['text_length'] = processed_df[text_column].apply(len)
        processed_df['word_count'] = processed_df[text_column].apply(lambda x: len(x.split()))
        processed_df['uppercase_count'] = processed_df[text_column].apply(lambda x: sum(1 for c in x if c.isupper()))
        processed_df['uppercase_ratio'] = processed_df['uppercase_count'] / processed_df['text_length']
        processed_df['punctuation_count'] = processed_df[text_column].apply(lambda x: sum(1 for c in x if c in string.punctuation))
        processed_df['punctuation_ratio'] = processed_df['punctuation_count'] / processed_df['text_length']
        processed_df['unique_words'] = processed_df[text_column].apply(lambda x: len(set(x.lower().split())))
        processed_df['unique_ratio'] = processed_df['unique_words'] / processed_df['word_count']
        
        # Display correlation with target
        if 'label' in processed_df.columns:
            # Convert label to numeric if needed
            if processed_df['label'].dtype == 'object':
                processed_df['label_num'] = processed_df['label'].map({'real': 0, 'fake': 1})
            else:
                processed_df['label_num'] = processed_df['label']
                
            # Calculate correlation with engineered features
            corr = processed_df[['text_length', 'word_count', 'uppercase_ratio', 
                                'punctuation_ratio', 'unique_ratio', 'label_num']].corr()['label_num'].drop('label_num')
            
            print("\nCorrelation of engineered features with label:")
            print(corr)
            
            # Plot correlation
            plt.figure(figsize=(10, 6))
            corr.sort_values().plot(kind='bar')
            plt.title('Correlation of Text Features with Label')
            plt.tight_layout()
            plt.savefig('feature_correlation.png')
            plt.close()
        
        # Clean the text
        print("\nCleaning text...")
        processed_df['cleaned_text'] = processed_df[text_column].apply(self._clean_text)
        
        return processed_df
    
    def _clean_text(self, text):
        """Clean a single text with advanced techniques"""
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
        
        # Remove HTML tags
        text = re.sub(r'<.*?>', '', text)
        
        # Remove punctuation
        text = text.translate(str.maketrans('', '', string.punctuation))
        
        # Remove numbers
        text = re.sub(r'\d+', '', text)
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords
        tokens = [word for word in tokens if word not in self.stop_words]
        
        # Lemmatization
        tokens = [self.lemmatizer.lemmatize(word) for word in tokens]
        
        # Join tokens back into text
        cleaned_text = ' '.join(tokens)
        
        return cleaned_text
    
    def feature_extraction(self, train_texts, test_texts, method='tfidf'):
        """Extract features from text using various methods"""
        print(f"\n===== Feature Extraction using {method.upper()} =====")
        
        if method.lower() == 'tfidf':
            # TF-IDF Vectorization
            self.vectorizer = TfidfVectorizer(
                max_features=5000,
                min_df=5,
                max_df=0.7,
                ngram_range=(1, 2),
                sublinear_tf=True
            )
        elif method.lower() == 'count':
            # Count Vectorization
            self.vectorizer = CountVectorizer(
                max_features=5000,
                min_df=5,
                max_df=0.7,
                ngram_range=(1, 2)
            )
        else:
            raise ValueError("Method must be 'tfidf' or 'count'")
        
        # Fit on training data and transform both train and test
        train_features = self.vectorizer.fit_transform(train_texts)
        test_features = self.vectorizer.transform(test_texts)
        
        print(f"Training features shape: {train_features.shape}")
        print(f"Testing features shape: {test_features.shape}")
        
        return train_features, test_features
    
    def build_models(self):
        """Build multiple classification models"""
        print("\n===== Building Classification Models =====")
        
        # Initialize various models
        models = {
            'naive_bayes': MultinomialNB(),
            'logistic_regression': LogisticRegression(random_state=self.random_state),
            'svm': CalibratedClassifierCV(LinearSVC(random_state=self.random_state)),
            'random_forest': RandomForestClassifier(random_state=self.random_state),
            'gradient_boosting': GradientBoostingClassifier(random_state=self.random_state)
        }
        
        # Define hyperparameter grids for each model
        param_grids = {
            'naive_bayes': {
                'alpha': [0.1, 0.5, 1.0]
            },
            'logistic_regression': {
                'C': [0.1, 1.0, 10.0],
                'penalty': ['l2'],
                'solver': ['liblinear', 'saga']
            },
            'svm': {
                'base_estimator__C': [0.1, 1.0, 10.0],
                'base_estimator__loss': ['hinge', 'squared_hinge']
            },
            'random_forest': {
                'n_estimators': [100, 200],
                'max_depth': [None, 10, 20],
                'min_samples_split': [2, 5]
            },
            'gradient_boosting': {
                'n_estimators': [100, 200],
                'learning_rate': [0.1, 0.2],
                'max_depth': [3, 5]
            }
        }
        
        return models, param_grids
    
    def train_models(self, X_train, y_train, models, param_grids, cv=5):
        """Train multiple models with hyperparameter tuning"""
        print("\n===== Training Models with Hyperparameter Tuning =====")
        
        best_models = {}
        
        for name, model in models.items():
            print(f"\nTraining {name}...")
            start_time = time.time()
            
            # Use GridSearchCV for hyperparameter tuning
            grid_search = GridSearchCV(
                estimator=model,
                param_grid=param_grids[name],
                cv=cv,
                scoring='accuracy',
                n_jobs=-1,
                verbose=1
            )
            
            grid_search.fit(X_train, y_train)
            
            # Get the best model
            best_model = grid_search.best_estimator_
            best_models[name] = best_model
            
            # Print results
            print(f"Best parameters: {grid_search.best_params_}")
            print(f"Best cross-validation score: {grid_search.best_score_:.4f}")
            print(f"Training time: {time.time() - start_time:.2f} seconds")
            
            # Store the model
            self.models[name] = best_model
        
        return best_models
    
    def create_ensemble(self, best_models):
        """Create an ensemble model from the best individual models"""
        print("\n===== Creating Ensemble Model =====")
        
        # Create a voting classifier with soft voting
        estimators = [(name, model) for name, model in best_models.items()]
        ensemble = VotingClassifier(estimators=estimators, voting='soft')
        
        self.models['ensemble'] = ensemble
        return ensemble
    
    def evaluate_models(self, X_test, y_test):
        """Evaluate all trained models on the test set"""
        print("\n===== Model Evaluation =====")
        
        results = {}
        best_accuracy = 0
        
        for name, model in self.models.items():
            print(f"\nEvaluating {name}...")
            
            # Make predictions
            y_pred = model.predict(X_test)
            y_prob = None
            
            # Get probabilities if available
            if hasattr(model, "predict_proba"):
                y_prob = model.predict_proba(X_test)[:, 1]
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            report = classification_report(y_test, y_pred)
            
            # Store results
            results[name] = {
                'accuracy': accuracy,
                'report': report,
                'predictions': y_pred,
                'probabilities': y_prob
            }
            
            # Print results
            print(f"Accuracy: {accuracy:.4f}")
            print(f"Classification Report:\n{report}")
            
            # Confusion Matrix
            cm = confusion_matrix(y_test, y_pred)
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.title(f'Confusion Matrix - {name}')
            plt.ylabel('Actual')
            plt.xlabel('Predicted')
            plt.tight_layout()
            plt.savefig(f'confusion_matrix_{name}.png')
            plt.close()
            
            # ROC Curve (if probabilities available)
            if y_prob is not None:
                self._plot_roc_curve(y_test, y_prob, name)
                self._plot_precision_recall_curve(y_test, y_prob, name)
            
            # Track best model
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                self.best_model = model
        
        # Model comparison
        self._compare_models(results)
        
        return results
    
    def _plot_roc_curve(self, y_test, y_prob, model_name):
        """Plot ROC curve for a model"""
        # Compute ROC curve and ROC area
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = roc_auc_score(y_test, y_prob)
        
        # Plot
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
        plt.savefig(f'roc_curve_{model_name}.png')
        plt.close()
    
    def _plot_precision_recall_curve(self, y_test, y_prob, model_name):
        """Plot precision-recall curve for a model"""
        # Compute precision-recall curve
        precision, recall, _ = precision_recall_curve(y_test, y_prob)
        
        # Plot
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, color='blue', lw=2)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'Precision-Recall Curve - {model_name}')
        plt.tight_layout()
        plt.savefig(f'precision_recall_{model_name}.png')
        plt.close()
    
    def _compare_models(self, results):
        """Compare all models and visualize their performance"""
        # Extract accuracies
        accuracies = {name: result['accuracy'] for name, result in results.items()}
        
        # Plot comparison
        plt.figure(figsize=(10, 6))
        plt.bar(accuracies.keys(), accuracies.values())
        plt.xlabel('Model')
        plt.ylabel('Accuracy')
        plt.title('Model Comparison')
        plt.ylim(0.75, 1.0)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('model_comparison.png')
        plt.close()
        
        # Print comparison
        print("\nModel Comparison:")
        for name, accuracy in sorted(accuracies.items(), key=lambda x: x[1], reverse=True):
            print(f"{name}: {accuracy:.4f}")
    
    def get_feature_importance(self, model_name=None):
        """Get feature importance for a specific model or the best model"""
        if model_name is not None and model_name in self.models:
            model = self.models[model_name]
        else:
            model = self.best_model
            model_name = next(name for name, m in self.models.items() if m == model)
        
        # Check if the model has feature importance attributes
        if hasattr(model, 'feature_importances_'):
            # For tree-based models
            importances = model.feature_importances_
            
            # Get feature names from the vectorizer
            feature_names = self.vectorizer.get_feature_names_out()
            
            # Create a DataFrame for visualization
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': importances
            })
            
            # Sort by importance
            importance_df = importance_df.sort_values('importance', ascending=False).head(20)
            
        elif hasattr(model, 'coef_'):
            # For linear models
            coef = model.coef_[0]
            
            # Get feature names from the vectorizer
            feature_names = self.vectorizer.get_feature_names_out()
            
            # Create a DataFrame for visualization
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': np.abs(coef)  # Use absolute values for importance
            })
            
            # Sort by importance
            importance_df = importance_df.sort_values('importance', ascending=False).head(20)
            
        else:
            print(f"Feature importance not available for {model_name}")
            return None
        
        # Plot feature importance
        plt.figure(figsize=(12, 8))
        sns.barplot(x='importance', y='feature', data=importance_df)
        plt.title(f'Top 20 Important Features - {model_name}')
        plt.tight_layout()
        plt.savefig(f'feature_importance_{model_name}.png')
        plt.close()
        
        return importance_df
    
    def save_model(self, model_name=None, filename=None):
        """Save a specific model or the best model to disk"""
        if model_name is not None and model_name in self.models:
            model = self.models[model_name]
            if filename is None:
                filename = f"{model_name}_model.pkl"
        else:
            model = self.best_model
            model_name = next(name for name, m in self.models.items() if m == model)
            if filename is None:
                filename = f"best_model_{model_name}.pkl"
        
        # Save model
        with open(filename, 'wb') as f:
            pickle.dump(model, f)
        
        # Save vectorizer
        vectorizer_filename = f"vectorizer_{model_name}.pkl"
        with open(vectorizer_filename, 'wb') as f:
            pickle.dump(self.vectorizer, f)
        
        print(f"\nModel saved as {filename}")
        print(f"Vectorizer saved as {vectorizer_filename}")
    
    def load_model(self, model_filename, vectorizer_filename):
        """Load a model and its vectorizer from disk"""
        try:
            # Load model
            with open(model_filename, 'rb') as f:
                model = pickle.load(f)
            
            # Load vectorizer
            with open(vectorizer_filename, 'rb') as f:
                self.vectorizer = pickle.load(f)
            
            # Set as best model
            self.best_model = model
            
            print(f"Model loaded from {model_filename}")
            print(f"Vectorizer loaded from {vectorizer_filename}")
            
            return model
        except Exception as e:
            print(f"Error loading model: {e}")
            return None
    
    def predict(self, text, clean=True):
        """Make predictions on new text"""
        if self.best_model is None:
            print("No model available. Train or load a model first.")
            return None
        
        # Clean text if requested
        if clean:
            text = self._clean_text(text)
        
        # Transform text
        text_features = self.vectorizer.transform([text])
        
        # Make prediction
        prediction = self.best_model.predict(text_features)[0]
        
        # Get probability if available
        probability = None
        if hasattr(self.best_model, "predict_proba"):
            probability = self.best_model.predict_proba(text_features)[0]
        
        return prediction, probability
    
    def get_explanation(self, text, clean=True, n_features=10):
        """Explain the prediction by showing the most influential features"""
        if self.best_model is None:
            print("No model available. Train or load a model first.")
            return None
        
        # Clean text if requested
        if clean:
            text = self._clean_text(text)
        
        # Transform text
        text_features = self.vectorizer.transform([text])
        
        # Get feature names
        feature_names = self.vectorizer.get_feature_names_out()
        
        # Get non-zero features in the text
        nonzero_indices = text_features.nonzero()[1]
        nonzero_features = [(feature_names[i], text_features[0, i]) for i in nonzero_indices]
        
        # Sort by importance
        nonzero_features.sort(key=lambda x: x[1], reverse=True)
        
        # Get top features
        top_features = nonzero_features[:n_features]
        
        # Make prediction
        prediction, probability = self.predict(text, clean=False)
        
        return {
            'prediction': prediction,
            'probability': probability,
            'top_features': top_features
        }
    
    def create_pipeline(self, model_name=None):
        """Create a scikit-learn pipeline for easy deployment"""
        if model_name is not None and model_name in self.models:
            model = self.models[model_name]
        else:
            model = self.best_model
            model_name = next(name for name, m in self.models.items() if m == model)
        
        # Create pipeline
        pipeline = Pipeline([
            ('vectorizer', self.vectorizer),
            ('classifier', model)
        ])
        
        print(f"Pipeline created for {model_name}")
        return pipeline


# Example usage
def main():
    # Initialize the detector
    detector = FakeNewsDetector(random_state=42)
    
    # Load and explore data
    df = detector.load_data('fake_news.csv')
    if df is None:
        print("Exiting due to data loading error.")
        return
    
    # Explore data
    df = detector.explore_data(df)
    
    # Preprocess data
    processed_df = detector.preprocess_text(df, text_column='text')
    
    # Split data
    X = processed_df['cleaned_text']
    y = processed_df['label']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Feature extraction
    X_train_features, X_test_features = detector.feature_extraction(X_train, X_test, method='tfidf')
    
    # Build models
    models, param_grids = detector.build_models()
    
    # Train models
    best_models = detector.train_models(X_train_features, y_train, models, param_grids, cv=5)
    
    # Create ensemble
    ensemble = detector.create_ensemble(best_models)
    ensemble.fit(X_train_features, y_train)
    
    # Evaluate models
    results = detector.evaluate_models(X_test_features, y_test)
    
    # Get feature importance
    importance_df = detector.get_feature_importance()
    
    # Save the best model
    detector.save_model()
    
    # Example prediction
    example_text = """
    BREAKING: Scientists discover COVID-19 vaccine causes severe side effects in 80% of recipients.
    A new study from medical researchers reveals shocking findings about the dangerous COVID-19 vaccines.
    Share this with everyone you know before Big Pharma takes this down!
    """
    
    prediction, probability = detector.predict(example_text)
    print(f"\nExample prediction for potentially fake news: {prediction}")
    if probability is not None:
        print(f"Prediction probability: {probability}")
    
    # Get explanation
    explanation = detector.get_explanation(example_text)
    print("\nExplanation:")
    print(f"Prediction: {explanation['prediction']}")
    print("Top features:")
    for feature, value in explanation['top_features']:
        print(f"  {feature}: {value}")
    
    # Web app example code
    print("\nTo create a web interface, create a file named 'app.py' with the following code:")
    print("""
    import streamlit as st
    import pandas as pd
    from fake_news_detector import FakeNewsDetector
    
    # Load the pre-trained model
    detector = FakeNewsDetector()
    detector.load_model('best_model_ensemble.pkl', 'vectorizer_ensemble.pkl')
    
    # Set up the Streamlit app
    st.title('Fake News Detector')
    st.write('Enter a news article to check if it\'s real or fake')
    
    # Input text area
    news_text = st.text_area('News Text', height=200)
    
    # Analysis button
    if st.button('Analyze'):
        if news_text:
            # Get prediction and explanation
            explanation = detector.get_explanation(news_text)
            prediction = explanation['prediction']
            probability = explanation['probability']
            top_features = explanation['top_features']
            
            # Display result
            if prediction == 'fake':
                st.error('⚠️ This news article appears to be FAKE')
            else:
                st.success('✅ This news article appears to be REAL')
            
            # Display probability
            if probability is not None:
                st.write(f'Confidence: {probability[1]:.2%}' if prediction == 'fake' else f'Confidence: {probability[0]:.2%}')
            
            # Display top features
            st.subheader('Top influential words/phrases:')
            for feature, value in top_features:
                st.write(f'- {feature}: {value:.4f}')
        else:
            st.warning('Please enter some text to analyze')
    
    # Show information about the model
    with st.expander('About this model'):
        st.write('''
        This fake news detector uses Natural Language Processing and Machine Learning 
        to identify potentially fake news articles. It analyzes the text content and 
        looks for patterns that are commonly found in fake news.
        
        The model was trained on a dataset of labeled real and fake news articles.
        ''')
    """)
    
    print("\nRun the app with: streamlit run app.py")
    
    # Model deployment advice
    print("\nTo deploy this model as an API:")
    print("""
    1. Create a Flask API:
       - Save the model and vectorizer
       - Create a Flask app that loads the model and provides an endpoint for predictions
       - Deploy to a cloud service like Heroku, AWS, or Google Cloud
       
    2. Create a model card documenting:
       - Model performance metrics
       - Training dataset information
       - Model limitations
       - Ethical considerations
       
    3. Consider implementing:
       - Monitoring for drift in production
       - User feedback collection
       - Regular retraining with new data
    """)


if __name__ == "__main__":
    main()