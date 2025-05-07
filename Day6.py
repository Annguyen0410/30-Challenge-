import nltk
from nltk.corpus import movie_reviews, stopwords, wordnet
from nltk.classify import NaiveBayesClassifier
from nltk.classify.util import accuracy as nltk_accuracy
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
import random
import string

nltk.download('movie_reviews', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

lemmatizer = WordNetLemmatizer()
stop_words_set = set(stopwords.words('english'))
punctuation_set = set(string.punctuation)
negation_markers_orig = {
    "not", "no", "never", "n't", "cannot", "couldn't", "didn't", "doesn't", 
    "hadn't", "hasn't", "haven't", "isn't", "mightn't", "mustn't", "needn't", 
    "shan't", "shouldn't", "wasn't", "weren't", "won't", "wouldn't"
}

def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN

def preprocess_text_for_features(raw_token_list):
    current_tokens = [word.lower() for word in raw_token_list]

    tokens_with_negation_prefix = []
    negation_active = False
    negation_scope_remaining = 0
    
    for token in current_tokens:
        if token in negation_markers_orig:
            negation_active = True
            negation_scope_remaining = 3 
            continue 
        
        if negation_active and negation_scope_remaining > 0 and token.isalpha():
            tokens_with_negation_prefix.append("NOT_" + token)
            negation_scope_remaining -= 1
        else:
            tokens_with_negation_prefix.append(token)
        
        if not token.isalpha() or token in punctuation_set:
            negation_active = False
            negation_scope_remaining = 0
        
        if negation_scope_remaining == 0: # Ensure negation scope is reset if it runs out
            negation_active = False
            
    current_tokens = tokens_with_negation_prefix
    
    pos_tagged_tokens = pos_tag(current_tokens)
    
    final_processed_tokens = []
    for token, tag in pos_tagged_tokens:
        base_token_for_check = token
        actual_word_for_lemmatization = token
        is_prefixed_negation = False

        if token.startswith("NOT_"):
            base_token_for_check = token[4:]
            if not base_token_for_check: 
                continue
            actual_word_for_lemmatization = base_token_for_check
            is_prefixed_negation = True
        
        if base_token_for_check.lower() in stop_words_set or \
           base_token_for_check in punctuation_set or \
           not base_token_for_check.isalpha():
            continue

        wordnet_pos_for_lemma = get_wordnet_pos(tag)
        lemma = lemmatizer.lemmatize(actual_word_for_lemmatization, pos=wordnet_pos_for_lemma)
        
        if is_prefixed_negation:
            final_processed_tokens.append("NOT_" + lemma)
        else:
            final_processed_tokens.append(lemma)
            
    return final_processed_tokens

def extract_enhanced_features(processed_tokens_list, raw_text_for_punctuation_check=""):
    features = {}
    for token in processed_tokens_list:
        features[f'contains({token})'] = True
    
    for i in range(len(processed_tokens_list) - 1):
        bigram = f'{processed_tokens_list[i]}_{processed_tokens_list[i+1]}'
        features[f'bigram({bigram})'] = True
        
    for i in range(len(processed_tokens_list) - 2):
        trigram = f'{processed_tokens_list[i]}_{processed_tokens_list[i+1]}_{processed_tokens_list[i+2]}'
        features[f'trigram({trigram})'] = True

    if raw_text_for_punctuation_check:
        num_exclamations = raw_text_for_punctuation_check.count('!')
        num_question_marks = raw_text_for_punctuation_check.count('?')
        
        if num_exclamations > 0:
            features['has_exclamation'] = True
        if num_exclamations > 2:
            features['has_multiple_exclamations'] = True
        if num_question_marks > 0:
            features['has_question_mark'] = True
        if raw_text_for_punctuation_check.endswith('!'):
            features['ends_with_exclamation'] = True
        if raw_text_for_punctuation_check.endswith('?'):
            features['ends_with_question'] = True
            
    return features

documents_for_processing = []
for category_label in movie_reviews.categories():
    for file_id in movie_reviews.fileids(category_label):
        words = list(movie_reviews.words(file_id))
        raw_text = movie_reviews.raw(file_id)
        documents_for_processing.append(((words, raw_text), category_label))

random.shuffle(documents_for_processing)

all_featuresets = []
for (doc_words, doc_raw_text), label in documents_for_processing:
    processed_doc_tokens = preprocess_text_for_features(doc_words)
    if processed_doc_tokens or doc_raw_text: # Allow feature extraction if raw_text exists for punctuation
        doc_features = extract_enhanced_features(processed_doc_tokens, doc_raw_text)
        if doc_features: # Ensure some features were extracted
             all_featuresets.append((doc_features, label))


num_docs = len(all_featuresets)
split_index = int(num_docs * 0.8) 

train_set, test_set = all_featuresets[:split_index], all_featuresets[split_index:]

if not train_set:
    print("Error: Training set is empty. Check preprocessing or data loading.")
    exit()

classifier = NaiveBayesClassifier.train(train_set)

if test_set:
    current_accuracy = nltk_accuracy(classifier, test_set)
    print(f"Accuracy: {current_accuracy * 100:.2f}%")
else:
    print("Accuracy: N/A (Test set is empty)")

classifier.show_most_informative_features(20)

def analyze_sentiment_enhanced(input_text_string):
    tokens_from_input = word_tokenize(input_text_string)
    
    processed_input_tokens = preprocess_text_for_features(tokens_from_input)
    
    current_features = extract_enhanced_features(processed_input_tokens, input_text_string)
    
    if not current_features:
        return "neutral (no relevant features found)"

    return classifier.classify(current_features)

example_test_sentences = [
    "This movie is absolutely fantastic! The acting, the story, everything was amazing!",
    "I hated this movie. It was a waste of time and money.",
    "The plot was a bit dull, but the performances were great.",
    "I have mixed feelings about this film. It was okay, not great but not terrible either.",
    "An utterly brilliant masterpiece of cinema.",
    "Completely unwatchable and a total disappointment.",
    "The film had some interesting ideas but failed in execution.",
    "Not the best, not the worst, just average.",
    "This is not good, not enjoyable at all.",
    "I really don't like this movie, it's terrible!!!",
    "Is this supposed to be a good film? I doubt it.",
    "The movie was... well, it was a movie." # Test neutral-ish with few strong signals
]

for text_sentence in example_test_sentences:
    print(f"Sentence: {text_sentence}")
    print(f"Predicted sentiment: {analyze_sentiment_enhanced(text_sentence)}")
    print()