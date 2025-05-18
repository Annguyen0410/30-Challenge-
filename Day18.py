# Import necessary libraries
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer # For lemmatization
import string # For punctuation
from collections import Counter # For frequency counting

# Download necessary NLTK data (only needed once)
# nltk.download("stopwords")
# nltk.download("punkt")
# nltk.download("wordnet")
# nltk.download("omw-1.4") # Open Multilingual Wordnet, needed by WordNetLemmatizer

# Example text for summarization
TEXT_EXAMPLE = """
Artificial Intelligence (AI) refers to the simulation of human intelligence in machines that are programmed to think like humans and mimic their actions. 
The term may also be applied to any machine that exhibits traits associated with a human mind such as learning and problem-solving.

The ideal characteristic of artificial intelligence is its ability to rationalize and take actions that have the best chance of achieving a specific goal. 
A subset of artificial intelligence is machine learning, which refers to the concept that computer programs can automatically learn from and adapt to new data without being assisted by humans. 
Deep learning techniques enable this automatic learning through the absorption of huge amounts of unstructured data such as text, images, or video.
AI is rapidly evolving, impacting various sectors from healthcare to finance. Ethical considerations are paramount as AI becomes more integrated into our daily lives.
"""

class TextSummarizer:
    def __init__(self, language: str = "english"):
        self.stop_words = set(stopwords.words(language))
        self.lemmatizer = WordNetLemmatizer()
        self.punctuation_table = str.maketrans('', '', string.punctuation)

    def _preprocess_text(self, text: str) -> list[str]:
        """Tokenizes, cleans, and lemmatizes words from the text."""
        # Lowercase
        text = text.lower()
        # Tokenize words
        words = word_tokenize(text)
        
        lemmatized_words = []
        for word in words:
            # Remove punctuation from each word
            cleaned_word = word.translate(self.punctuation_table)
            if cleaned_word and cleaned_word not in self.stop_words and cleaned_word.isalpha():
                lemmatized_words.append(self.lemmatizer.lemmatize(cleaned_word))
        return lemmatized_words

    def _calculate_word_frequencies(self, words: list[str]) -> Counter:
        """Calculates the frequency of each lemmatized word."""
        return Counter(words)

    def _score_sentences(self, sentences: list[str], word_frequencies: Counter, positional_weight_factor: float = 0.1) -> dict[str, float]:
        """
        Scores sentences based on word frequencies and positional importance.
        Positional weight gives a slight bonus to earlier sentences.
        """
        sentence_scores: dict[str, float] = {}
        num_total_sentences = len(sentences)

        for i, sentence in enumerate(sentences):
            # Tokenize, clean, and lemmatize words in the current sentence
            sentence_words = self._preprocess_text(sentence) # Re-preprocess to match word_frequencies keys
            
            if not sentence_words: # Skip empty sentences after preprocessing
                continue

            score = 0
            for word in sentence_words:
                if word in word_frequencies:
                    score += word_frequencies[word]
            
            # Normalize score by the number of significant words in the sentence
            # This prevents overly long sentences from dominating just due to length
            if len(sentence_words) > 0:
                normalized_score = score / len(sentence_words)
            else:
                normalized_score = 0 # Avoid division by zero for empty sentences

            # Add positional weight: earlier sentences get a slight boost
            # The weight decreases linearly from positional_weight_factor to 0
            positional_bonus = positional_weight_factor * (1 - (i / num_total_sentences))
            final_score = normalized_score * (1 + positional_bonus)
            
            sentence_scores[sentence] = final_score
            
        return sentence_scores

    def get_keywords(self, text: str, top_n: int = 5) -> list[tuple[str, int]]:
        """Extracts the most frequent significant words as keywords."""
        processed_words = self._preprocess_text(text)
        word_frequencies = self._calculate_word_frequencies(processed_words)
        return word_frequencies.most_common(top_n)

    def summarize(self, text: str, summary_ratio: float = 0.3, min_sentences: int = 1) -> str:
        """
        Generates a frequency-based summary of the text.

        Args:
            text (str): The input text to summarize.
            summary_ratio (float): The desired ratio of summary sentences to original sentences (e.g., 0.3 for 30%).
            min_sentences (int): Minimum number of sentences in the summary.

        Returns:
            str: The generated summary.
        """
        if not text.strip():
            return "Input text is empty."

        # 1. Tokenize text into sentences
        sentences = sent_tokenize(text)
        if not sentences:
            return "No sentences found in the text."

        # 2. Preprocess the entire text to get global word frequencies
        all_processed_words = self._preprocess_text(text)
        if not all_processed_words:
            return "No significant words found after preprocessing."
            
        word_frequencies = self._calculate_word_frequencies(all_processed_words)

        # 3. Score each sentence
        sentence_scores = self._score_sentences(sentences, word_frequencies, positional_weight_factor=0.2)
        
        if not sentence_scores:
             return "Could not score any sentences. Text might be too short or lack significant content."


        # 4. Determine the number of sentences for the summary
        num_summary_sentences = max(min_sentences, int(len(sentences) * summary_ratio))
        # Ensure we don't request more sentences than available
        num_summary_sentences = min(num_summary_sentences, len(sentence_scores))


        # 5. Sort sentences by score and select the top ones
        # We sort the original sentences based on their scores to maintain original order
        # if scores are similar or for reconstructing the summary.
        # However, for extractive summary, the order of highest scored sentences is what we pick.
        
        # Get a list of (sentence, score) pairs
        scored_sentence_tuples = sorted(sentence_scores.items(), key=lambda item: item[1], reverse=True)
        
        # Select the top sentences based on score
        top_sentences_with_scores = scored_sentence_tuples[:num_summary_sentences]

        # To present the summary, it's often better to re-order the selected sentences
        # back to their original appearance order in the text.
        summary_sentences_in_order = []
        for original_sentence in sentences:
            for selected_sentence, score in top_sentences_with_scores:
                if original_sentence == selected_sentence:
                    summary_sentences_in_order.append(selected_sentence)
                    break # Found it, move to next original_sentence

        summary = " ".join(summary_sentences_in_order)
        return summary


# --- Main execution ---
if __name__ == "__main__":
    # Ensure NLTK resources are available
    try:
        nltk.data.find('corpora/stopwords')
    except nltk.downloader.DownloadError:
        print("Downloading stopwords...")
        nltk.download("stopwords", quiet=True)
    try:
        nltk.data.find('tokenizers/punkt')
    except nltk.downloader.DownloadError:
        print("Downloading punkt tokenizer...")
        nltk.download("punkt", quiet=True)
    try:
        nltk.data.find('corpora/wordnet')
    except nltk.downloader.DownloadError:
        print("Downloading wordnet...")
        nltk.download("wordnet", quiet=True)
    try:
        nltk.data.find('corpora/omw-1.4')
    except nltk.downloader.DownloadError:
        print("Downloading omw-1.4...")
        nltk.download("omw-1.4", quiet=True)

    summarizer = TextSummarizer()

    print("Original Text:\n", TEXT_EXAMPLE)

    # Generate summary with default ratio (30%)
    summary_default_ratio = summarizer.summarize(TEXT_EXAMPLE)
    print("\nSummary (approx 30%):\n", summary_default_ratio)

    # Generate summary with a specific ratio (e.g., 50%)
    summary_custom_ratio = summarizer.summarize(TEXT_EXAMPLE, summary_ratio=0.5)
    print("\nSummary (approx 50%):\n", summary_custom_ratio)
    
    # Generate summary with a specific number of sentences (by adjusting ratio carefully or setting min_sentences)
    # To get exactly N sentences, you'd calculate ratio = N / total_sentences
    # For 2 sentences:
    num_sentences_original = len(sent_tokenize(TEXT_EXAMPLE))
    if num_sentences_original > 0:
        target_ratio_for_2_sentences = 2 / num_sentences_original
        summary_2_sentences = summarizer.summarize(TEXT_EXAMPLE, summary_ratio=target_ratio_for_2_sentences, min_sentences=2)
        print("\nSummary (target 2 sentences):\n", summary_2_sentences)
    else:
        print("\nCannot generate 2-sentence summary from empty/short text.")


    # Extract keywords
    keywords = summarizer.get_keywords(TEXT_EXAMPLE, top_n=5)
    print("\nTop Keywords:\n", keywords)

    # Test with very short text
    short_text = "This is a very short text. It has only two sentences."
    print("\nOriginal Short Text:\n", short_text)
    summary_short = summarizer.summarize(short_text, summary_ratio=0.5, min_sentences=1)
    print("\nSummary of Short Text:\n", summary_short)
    
    # Test with empty text
    empty_text = ""
    print("\nOriginal Empty Text:\n", "''")
    summary_empty = summarizer.summarize(empty_text)
    print("\nSummary of Empty Text:\n", summary_empty)