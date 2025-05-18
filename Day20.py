import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.probability import FreqDist
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import re
import string
import requests
import os
from collections import Counter
import numpy as np
from textblob import TextBlob
import spacy
import emoji

# Download required NLTK data
nltk_resources = ["vader_lexicon", "stopwords", "punkt", "wordnet", "averaged_perceptron_tagger"]
for resource in nltk_resources:
    try:
        nltk.download(resource, quiet=True)
    except:
        print(f"Could not download {resource}")

# Try to load spaCy model
try:
    nlp = spacy.load("en_core_web_sm")
except:
    print("SpaCy model not available. Installing minimal version.")
    os.system("python -m spacy download en_core_web_sm")
    try:
        nlp = spacy.load("en_core_web_sm")
    except:
        print("Could not load SpaCy model. Some features will be limited.")
        nlp = None

# Initialize the NLTK sentiment analyzer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
sid = SentimentIntensityAnalyzer()

class EmotionDetector:
    def __init__(self):
        """Initialize the emotion detection system."""
        self.lemmatizer = WordNetLemmatizer()
        self.emotion_lexicon = self._load_emotion_lexicon()
        self.emotion_model = None
        self.emotion_labels = ['Joy', 'Sadness', 'Anger', 'Fear', 'Surprise', 'Disgust', 'Neutral']
        self.stop_words = set(stopwords.words('english'))
        
    def _load_emotion_lexicon(self):
        """Load a basic emotion lexicon for rule-based detection."""
        # This is a simplified lexicon - in a real project, you would use a more comprehensive one
        lexicon = {
            'Joy': ['happy', 'joy', 'delighted', 'glad', 'pleased', 'excited', 'thrilled', 'content', 'satisfied', 'cheerful'],
            'Sadness': ['sad', 'unhappy', 'depressed', 'gloomy', 'miserable', 'heartbroken', 'disappointed', 'sorrowful', 'grieved'],
            'Anger': ['angry', 'furious', 'outraged', 'enraged', 'irritated', 'annoyed', 'mad', 'frustrated', 'bitter'],
            'Fear': ['afraid', 'scared', 'frightened', 'terrified', 'anxious', 'nervous', 'worried', 'panicked', 'alarmed'],
            'Surprise': ['surprised', 'amazed', 'astonished', 'shocked', 'stunned', 'startled', 'unexpected', 'wonder'],
            'Disgust': ['disgusted', 'repulsed', 'revolted', 'appalled', 'nauseated', 'detestable', 'offensive', 'loathsome']
        }
        return lexicon
    
    def preprocess_text(self, text, advanced=True):
        """Perform text preprocessing steps."""
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
        
        # Remove HTML tags
        text = re.sub(r'<.*?>', '', text)
        
        # Replace emojis with their descriptions if advanced preprocessing
        if advanced:
            text = self._process_emojis(text)
        
        # Remove punctuation
        text = text.translate(str.maketrans('', '', string.punctuation))
        
        # Tokenize the text
        tokens = word_tokenize(text)
        
        # Remove stopwords and lemmatize if advanced preprocessing
        if advanced:
            tokens = [self.lemmatizer.lemmatize(token) for token in tokens if token not in self.stop_words]
        
        return ' '.join(tokens), tokens
    
    def _process_emojis(self, text):
        """Convert emojis to text descriptions."""
        return emoji.demojize(text)
    
    def extract_features(self, text, tokens):
        """Extract linguistic and emotional features from text."""
        features = {}
        
        # Basic statistics
        features['word_count'] = len(tokens)
        features['avg_word_length'] = sum(len(word) for word in tokens) / len(tokens) if tokens else 0
        features['sentence_count'] = len(sent_tokenize(text))
        
        # NLTK VADER sentiment scores
        vader_scores = sid.polarity_scores(text)
        features.update(vader_scores)
        
        # TextBlob sentiment
        blob = TextBlob(text)
        features['textblob_polarity'] = blob.sentiment.polarity
        features['textblob_subjectivity'] = blob.sentiment.subjectivity
        
        # Lexicon-based emotion word counts
        for emotion, words in self.emotion_lexicon.items():
            count = sum(1 for token in tokens if token in words)
            features[f'{emotion.lower()}_word_count'] = count
            features[f'{emotion.lower()}_word_ratio'] = count / len(tokens) if tokens else 0
        
        # Linguistic features using spaCy if available
        if nlp:
            doc = nlp(text)
            features['noun_count'] = len([token for token in doc if token.pos_ == 'NOUN'])
            features['verb_count'] = len([token for token in doc if token.pos_ == 'VERB'])
            features['adj_count'] = len([token for token in doc if token.pos_ == 'ADJ'])
            features['named_entities'] = len(doc.ents)
        
        return features
    
    def train_model(self, texts, labels):
        """Train a machine learning model for emotion classification."""
        # Create a TF-IDF vectorizer
        vectorizer = TfidfVectorizer(max_features=1000)
        X = vectorizer.fit_transform(texts)
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42)
        
        # Train a RandomForest classifier
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # Evaluate the model
        predictions = model.predict(X_test)
        print("Model Performance:")
        print(classification_report(y_test, predictions))
        
        # Save the model and vectorizer
        self.emotion_model = model
        self.vectorizer = vectorizer
        
        return model
    
    def detect_emotion_rule_based(self, text):
        """Rule-based emotion detection using lexicons and VADER."""
        # Preprocess text
        processed_text, tokens = self.preprocess_text(text)
        
        # Get VADER sentiment scores
        scores = sid.polarity_scores(processed_text)
        
        # Count emotion words in text
        emotion_counts = {}
        for emotion, words in self.emotion_lexicon.items():
            count = sum(1 for token in tokens if token in words)
            emotion_counts[emotion] = count
        
        # Determine primary emotion based on lexicon matches
        max_emotion = max(emotion_counts.items(), key=lambda x: x[1]) if emotion_counts else ('Neutral', 0)
        
        # Use VADER for sentiment intensity
        if max_emotion[1] == 0:  # No emotion words found
            if scores['compound'] >= 0.5:
                primary_emotion = 'Joy'
            elif scores['compound'] <= -0.5:
                primary_emotion = 'Sadness'
            elif scores['neg'] > 0.5:
                primary_emotion = 'Anger'
            elif scores['neu'] > 0.7:
                primary_emotion = 'Neutral'
            else:
                primary_emotion = 'Mixed'
        else:
            primary_emotion = max_emotion[0]
        
        # Calculate intensity based on VADER and word counts
        if primary_emotion == 'Neutral':
            intensity = 0.0
        elif primary_emotion == 'Mixed':
            intensity = abs(scores['compound'])
        else:
            # Normalize the count by total words
            word_ratio = max_emotion[1] / len(tokens) if tokens else 0
            # Combine with VADER for a more robust intensity
            intensity = (abs(scores['compound']) + word_ratio) / 2
            # Cap the intensity at 1.0
            intensity = min(intensity * 1.5, 1.0)
        
        return {
            'primary_emotion': primary_emotion,
            'intensity': intensity,
            'emotion_counts': emotion_counts,
            'sentiment_scores': scores,
            'features': self.extract_features(text, tokens)
        }
    
    def detect_emotion_ml(self, text):
        """Machine learning-based emotion detection."""
        if not self.emotion_model:
            return {"error": "ML model not trained yet"}
        
        # Preprocess text
        processed_text, _ = self.preprocess_text(text)
        
        # Transform text to vector
        text_vector = self.vectorizer.transform([processed_text])
        
        # Predict emotion
        predicted_emotion = self.emotion_model.predict(text_vector)[0]
        
        # Get probabilities for all emotions
        probabilities = self.emotion_model.predict_proba(text_vector)[0]
        emotion_probabilities = {emotion: prob for emotion, prob in zip(self.emotion_labels, probabilities)}
        
        return {
            'primary_emotion': predicted_emotion,
            'confidence': max(probabilities),
            'emotion_probabilities': emotion_probabilities
        }
    
    def detect_emotion(self, text, method='hybrid'):
        """Main emotion detection method that combines different approaches."""
        if not text or len(text.strip()) == 0:
            return {"error": "Empty text provided"}
        
        # Get rule-based detection results
        rule_results = self.detect_emotion_rule_based(text)
        
        # Get ML-based results if model is trained and hybrid method requested
        ml_results = {}
        if method == 'hybrid' and self.emotion_model:
            ml_results = self.detect_emotion_ml(text)
        
        # Combine results or use rule-based if ML not available
        if ml_results and 'error' not in ml_results:
            # Weighted combination
            rule_weight = 0.4
            ml_weight = 0.6
            
            # If rule and ML disagree but ML is confident, favor ML
            if ml_results['confidence'] > 0.7 and rule_results['primary_emotion'] != ml_results['primary_emotion']:
                primary_emotion = ml_results['primary_emotion']
                confidence = ml_results['confidence']
            # If rule and ML agree, boost confidence
            elif rule_results['primary_emotion'] == ml_results['primary_emotion']:
                primary_emotion = ml_results['primary_emotion']
                confidence = (rule_results['intensity'] * rule_weight) + (ml_results['confidence'] * ml_weight)
            # Otherwise weighted decision
            else:
                primary_emotion = ml_results['primary_emotion'] if ml_results['confidence'] > 0.5 else rule_results['primary_emotion']
                confidence = max(rule_results['intensity'], ml_results['confidence'])
                
            result = {
                'primary_emotion': primary_emotion,
                'confidence': confidence,
                'rule_based': rule_results,
                'ml_based': ml_results
            }
        else:
            result = {
                'primary_emotion': rule_results['primary_emotion'],
                'confidence': rule_results['intensity'],
                'rule_based': rule_results
            }
        
        return result
    
    def analyze_emotional_trends(self, texts):
        """Analyze emotional trends across multiple texts."""
        emotions = []
        intensities = []
        
        for text in texts:
            result = self.detect_emotion(text)
            emotions.append(result['primary_emotion'])
            intensities.append(result['confidence'])
        
        # Compile results
        emotion_counts = Counter(emotions)
        avg_intensity = sum(intensities) / len(intensities) if intensities else 0
        
        return {
            'emotion_distribution': dict(emotion_counts),
            'average_intensity': avg_intensity,
            'emotions_by_text': list(zip(emotions, intensities))
        }
    
    def visualize_emotions(self, texts, title="Emotion Analysis"):
        """Create visualizations for emotional analysis."""
        if not texts:
            return "No texts provided for visualization"
        
        results = self.analyze_emotional_trends(texts)
        
        # Create a figure with subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot emotion distribution
        emotions = list(results['emotion_distribution'].keys())
        counts = list(results['emotion_distribution'].values())
        
        # Sort by count
        sorted_data = sorted(zip(emotions, counts), key=lambda x: x[1], reverse=True)
        emotions, counts = zip(*sorted_data) if sorted_data else ([], [])
        
        # Create bar chart
        sns.barplot(x=list(emotions), y=list(counts), ax=ax1, palette='viridis')
        ax1.set_title('Emotion Distribution')
        ax1.set_xlabel('Emotion')
        ax1.set_ylabel('Count')
        plt.setp(ax1.get_xticklabels(), rotation=45, ha='right')
        
        # Plot emotion intensity over texts
        emotions_by_text = results['emotions_by_text']
        text_indices = list(range(1, len(emotions_by_text) + 1))
        intensities = [intensity for _, intensity in emotions_by_text]
        
        # Create line chart
        ax2.plot(text_indices, intensities, marker='o', linestyle='-', color='blue')
        ax2.set_title('Emotion Intensity by Text')
        ax2.set_xlabel('Text Number')
        ax2.set_ylabel('Intensity')
        
        # Add a horizontal line for average intensity
        ax2.axhline(y=results['average_intensity'], color='r', linestyle='--', label=f'Avg: {results["average_intensity"]:.2f}')
        ax2.legend()
        
        plt.tight_layout()
        plt.suptitle(title, fontsize=16)
        plt.subplots_adjust(top=0.9)
        
        return fig
    
    def get_emotion_suggestions(self, text, target_emotion='Joy'):
        """Suggest modifications to text to shift emotion toward target."""
        current_result = self.detect_emotion(text)
        current_emotion = current_result['primary_emotion']
        
        if current_emotion == target_emotion:
            return {
                'status': 'already_achieved',
                'message': f'Text already expresses {target_emotion}',
                'current_emotion': current_emotion,
                'suggestions': []
            }
        
        # Preprocess text
        _, tokens = self.preprocess_text(text, advanced=False)
        
        suggestions = []
        
        # Find words associated with current emotion to potentially replace
        current_emotion_words = []
        for token in tokens:
            for emotion, words in self.emotion_lexicon.items():
                if token in words and emotion == current_emotion:
                    current_emotion_words.append(token)
        
        # Suggest replacements for emotion words
        if current_emotion_words:
            for word in current_emotion_words:
                suggestion = f"Consider replacing '{word}' with a {target_emotion.lower()} word"
                if target_emotion in self.emotion_lexicon:
                    alternatives = self.emotion_lexicon[target_emotion][:3]  # Get a few alternatives
                    if alternatives:
                        suggestion += f" such as: {', '.join(alternatives)}"
                suggestions.append(suggestion)
        
        # Suggest general modifications based on emotion transition
        emotion_transition_suggestions = {
            ('Sadness', 'Joy'): [
                "Use more positive and uplifting language",
                "Focus on positive aspects or silver linings",
                "Include more exclamation marks to convey excitement"
            ],
            ('Anger', 'Joy'): [
                "Replace confrontational language with positive expressions",
                "Focus on solutions rather than problems",
                "Use more collaborative and positive framing"
            ],
            ('Neutral', 'Joy'): [
                "Add more enthusiastic language",
                "Include positive adjectives to describe subjects",
                "Express excitement or gratitude"
            ],
            ('Joy', 'Neutral'): [
                "Reduce exclamation marks",
                "Use more measured language with fewer intensifiers",
                "Balance positive observations with objective statements"
            ],
            ('Anger', 'Neutral'): [
                "Remove confrontational language",
                "Use objective descriptions instead of judgments",
                "Focus on facts rather than emotional reactions"
            ]
            # Could add more transitions as needed
        }
        
        transition_key = (current_emotion, target_emotion)
        if transition_key in emotion_transition_suggestions:
            suggestions.extend(emotion_transition_suggestions[transition_key])
        
        return {
            'status': 'suggestions_available',
            'current_emotion': current_emotion,
            'target_emotion': target_emotion,
            'suggestions': suggestions
        }
    
    def translate_and_analyze(self, text, source_lang='auto', target_lang='en'):
        """Translate non-English text and then analyze emotions."""
        # This is a placeholder for actual translation API integration
        translated_text = text
        
        # For demonstration, we'll just simulate translation for non-English
        if source_lang != 'en' and source_lang != 'auto':
            print(f"[Simulation] Translating from {source_lang} to {target_lang}")
            # In a real implementation, you would use a translation API here
            # translated_text = translation_api.translate(text, source_lang, target_lang)
            
            # For now, just pretend we translated by adding a note
            translated_text = f"[Translated] {text}"
        
        # Analyze the translated text
        emotion_results = self.detect_emotion(translated_text)
        
        return {
            'original_text': text,
            'translated_text': translated_text,
            'emotion_analysis': emotion_results
        }

# Example usage
if __name__ == "__main__":
    # Initialize the emotion detector
    detector = EmotionDetector()
    
    # Sample texts with different emotions
    texts = [
        "I am so happy today! The weather is beautiful, and everything is going well. I feel very positive and motivated!",
        "I'm feeling really down today. Nothing seems to be going right, and I'm struggling to stay positive.",
        "This is absolutely infuriating! I can't believe how they treated me. I'm so angry I could scream.",
        "I'm a bit worried about the upcoming exam. I hope I've prepared enough.",
        "Wow! I didn't expect that at all. What a surprising turn of events!",
        "That's disgusting! I can't stand even looking at it. It makes me sick."
    ]
    
    # Test the emotion detector on each text
    print("\n===== INDIVIDUAL TEXT ANALYSIS =====")
    for i, text in enumerate(texts):
        print(f"\nText {i+1}: {text[:50]}...")
        result = detector.detect_emotion(text)
        print(f"Detected emotion: {result['primary_emotion']} (Confidence: {result['confidence']:.2f})")
    
    # Show how to train a model (with simulated data for this example)
    print("\n===== TRAINING ML MODEL =====")
    # In a real project, you would use a labeled dataset
    # For demonstration, we'll just use our sample texts with assigned labels
    labels = ['Joy', 'Sadness', 'Anger', 'Fear', 'Surprise', 'Disgust']
    detector.train_model(texts, labels)
    
    # Analyze emotional trends
    print("\n===== EMOTIONAL TREND ANALYSIS =====")
    trends = detector.analyze_emotional_trends(texts)
    print("Emotion distribution:", trends['emotion_distribution'])
    print(f"Average intensity: {trends['average_intensity']:.2f}")
    
    # Create visualizations
    print("\n===== CREATING VISUALIZATIONS =====")
    fig = detector.visualize_emotions(texts, "Sample Emotion Analysis")
    
    # Get suggestions for modifying text emotion
    print("\n===== EMOTION MODIFICATION SUGGESTIONS =====")
    sad_text = "I'm feeling really down today. Nothing seems to be going right, and I'm struggling to stay positive."
    suggestions = detector.get_emotion_suggestions(sad_text, target_emotion='Joy')
    print(f"Current emotion: {suggestions['current_emotion']}")
    print("Suggestions to express Joy:")
    for i, suggestion in enumerate(suggestions['suggestions']):
        print(f"  {i+1}. {suggestion}")
    
    # Demonstrate translation and analysis (simulated)
    print("\n===== MULTILINGUAL ANALYSIS =====")
    french_text = "Je suis très heureux aujourd'hui! Le temps est magnifique."
    result = detector.translate_and_analyze(french_text, source_lang='fr')
    print(f"Original: {result['original_text']}")
    print(f"Translated: {result['translated_text']}")
    print(f"Detected emotion: {result['emotion_analysis']['primary_emotion']}")
    
    print("\n===== SYSTEM READY =====")
    print("The enhanced emotion detection system is ready for use.")