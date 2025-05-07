import pandas as pd
import numpy as np
import re
import string
import joblib
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

def download_nltk_data():
    try:
        nltk.data.find('corpora/wordnet')
    except nltk.downloader.DownloadError:
        print("Downloading NLTK WordNet...")
        nltk.download('wordnet')
    try:
        nltk.data.find('corpora/stopwords')
    except nltk.downloader.DownloadError:
        print("Downloading NLTK Stopwords...")
        nltk.download('stopwords')
    try:
        nltk.data.find('tokenizers/punkt')
    except nltk.downloader.DownloadError:
        print("Downloading NLTK Punkt...")
        nltk.download('punkt')

download_nltk_data()

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess_text_advanced(text):
    text = str(text).lower()
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = text.strip()
    tokens = nltk.word_tokenize(text)
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words and len(word) > 2]
    return " ".join(tokens)

def calculate_meta_features(df):
    df['message_len'] = df['message'].apply(len)
    df['word_count'] = df['message'].apply(lambda x: len(x.split()))
    df['punctuation_count'] = df['message'].apply(lambda x: sum(1 for char in x if char in string.punctuation))
    df['upper_case_count'] = df['message'].apply(lambda x: sum(1 for char in x if char.isupper()))
    # Avoid division by zero for empty messages if any
    df['punctuation_ratio'] = df['punctuation_count'] / df['message_len'].replace(0, 1)
    df['upper_case_ratio'] = df['upper_case_count'] / df['message_len'].replace(0, 1)
    return df

if __name__ == "__main__":
    try:
        df = pd.read_csv('spam.csv', encoding='latin-1')
    except FileNotFoundError:
        print("Error: 'spam.csv' not found. Please ensure the dataset is in the correct directory.")
        exit()

    df = df[['v1', 'v2']]
    df.columns = ['label', 'message']
    df.dropna(inplace=True)
    df['label'] = df['label'].map({'ham': 0, 'spam': 1})

    df = calculate_meta_features(df)
    df['message_clean'] = df['message'].apply(preprocess_text_advanced)

    meta_features = ['message_len', 'word_count', 'punctuation_ratio', 'upper_case_ratio']
    text_feature = 'message_clean'
    
    X = df[[text_feature] + meta_features]
    y = df['label']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    preprocessor = ColumnTransformer(
        transformers=[
            ('tfidf', TfidfVectorizer(ngram_range=(1, 2), max_df=0.9, min_df=3), text_feature),
            ('scaler', StandardScaler(), meta_features)
        ],
        remainder='passthrough' # Keep other columns (if any) - though we selected only specific ones
    )

    pipeline = Pipeline([
        ('features', preprocessor),
        ('classifier', LogisticRegression(solver='liblinear', random_state=42))
    ])

    param_grid = {
        'features__tfidf__max_features': [3000, 5000, None],
        'features__tfidf__ngram_range': [(1, 1), (1, 2)],
        'classifier__C': [0.1, 1, 10],
        'classifier__penalty': ['l1', 'l2']
    }

    print("Starting GridSearchCV... This may take a while.")
    grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=1)
    grid_search.fit(X_train, y_train)

    print("\nGridSearchCV Results:")
    print(f"Best Score (Accuracy): {grid_search.best_score_ * 100:.2f}%")
    print("Best Parameters:")
    print(grid_search.best_params_)

    best_model = grid_search.best_estimator_

    y_pred = best_model.predict(X_test)
    y_pred_proba = best_model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred, target_names=['Ham', 'Spam'])

    print("\n" + "=" * 60)
    print("Evaluation on Test Set using Best Model")
    print("=" * 60)
    print(f"Classifier: Logistic Regression (tuned)")
    print(f"Features: TF-IDF (tuned) + Scaled Meta-Features")
    print(f"Meta-Features Used: {meta_features}")
    print("-" * 60)
    print(f"Test Set Accuracy: {accuracy * 100:.2f}%")
    print("-" * 60)
    print("Confusion Matrix:")
    print(conf_matrix)
    print("-" * 60)
    print("Classification Report:")
    print(class_report)
    print("-" * 60)

    model_filename = 'spam_classifier_tuned_pipeline.joblib'
    joblib.dump(best_model, model_filename)
    print(f"Optimized model pipeline saved to {model_filename}")

  