import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from tqdm import tqdm  # For progress bars
import pickle
import os

class BookRecommendationSystem:
    """
    Advanced Book Recommendation System with multiple recommendation algorithms,
    evaluation metrics, and visualization capabilities.
    """
    
    def __init__(self, data=None, data_path=None):
        """
        Initialize the recommendation system with either a dataset or path to data file.
        
        Args:
            data (dict or DataFrame): Data containing user ratings
            data_path (str): Path to CSV file containing rating data
        """
        self.df = None
        self.user_book_matrix = None
        self.user_similarity = None
        self.book_similarity = None
        self.mean_ratings = None
        self.popular_books = None
        
        # Load data if provided
        if data is not None:
            if isinstance(data, dict):
                self.df = pd.DataFrame(data)
            elif isinstance(data, pd.DataFrame):
                self.df = data
        elif data_path is not None:
            self.load_data(data_path)
    
    def load_data(self, file_path):
        """
        Load data from a CSV file.
        
        Args:
            file_path (str): Path to the CSV file containing rating data
        """
        try:
            self.df = pd.read_csv(file_path)
            print(f"Data loaded successfully from {file_path}")
            print(f"Dataset shape: {self.df.shape}")
            return True
        except Exception as e:
            print(f"Error loading data: {str(e)}")
            return False
    
    def preprocess_data(self, min_ratings=1):
        """
        Preprocess the dataset by:
        - Removing users with too few ratings
        - Normalizing ratings (optional)
        - Creating user-book matrices
        
        Args:
            min_ratings (int): Minimum number of ratings required for users to be included
        """
        if self.df is None:
            print("No data available. Please load data first.")
            return
        
        print("Preprocessing data...")
        
        # Create a copy to avoid modifying the original data
        df_processed = self.df.copy()
        
        # Filter out users with too few ratings
        if min_ratings > 1:
            user_counts = df_processed['user_id'].value_counts()
            active_users = user_counts[user_counts >= min_ratings].index
            df_processed = df_processed[df_processed['user_id'].isin(active_users)]
            print(f"Filtered to {len(active_users)} users with at least {min_ratings} ratings")
        
        # Create user-book matrix
        self.user_book_matrix = df_processed.pivot_table(
            index='user_id', 
            columns='book_title', 
            values='rating'
        ).fillna(0)
        
        # Calculate mean ratings for books
        self.mean_ratings = df_processed.groupby('book_title')['rating'].mean().sort_values(ascending=False)
        
        # Calculate popularity of books (number of ratings)
        self.popular_books = df_processed['book_title'].value_counts()
        
        print("Data preprocessing complete.")
        print(f"Processed matrix shape: {self.user_book_matrix.shape}")
        
        return self.user_book_matrix
    
    def compute_similarity_matrices(self, method='cosine'):
        """
        Compute similarity matrices for users and items.
        
        Args:
            method (str): Similarity method ('cosine', 'pearson', 'euclidean')
        """
        if self.user_book_matrix is None:
            print("User-book matrix not available. Run preprocess_data first.")
            return
        
        print(f"Computing similarity matrices using {method} similarity...")
        
        # User similarity
        if method == 'cosine':
            self.user_similarity = cosine_similarity(self.user_book_matrix)
        elif method == 'pearson':
            # Center the ratings (subtract mean for each user)
            user_means = self.user_book_matrix.mean(axis=1)
            centered_matrix = self.user_book_matrix.sub(user_means, axis=0)
            self.user_similarity = np.corrcoef(centered_matrix)
        else:
            # Default to cosine
            self.user_similarity = cosine_similarity(self.user_book_matrix)
            
        self.user_similarity_df = pd.DataFrame(
            self.user_similarity, 
            index=self.user_book_matrix.index, 
            columns=self.user_book_matrix.index
        )
        
        # Book similarity (transpose the matrix to get book-user matrix)
        self.book_similarity = cosine_similarity(self.user_book_matrix.T)
        self.book_similarity_df = pd.DataFrame(
            self.book_similarity,
            index=self.user_book_matrix.columns,
            columns=self.user_book_matrix.columns
        )
        
        print("Similarity matrices computed successfully.")
        return self.user_similarity_df
    
    def user_based_recommend(self, user_id, top_n=5, sim_threshold=0.0):
        """
        Generate recommendations using user-based collaborative filtering.
        
        Args:
            user_id: ID of the user to recommend for
            top_n (int): Number of recommendations to return
            sim_threshold (float): Minimum similarity score for users to be considered
            
        Returns:
            list: Top N recommended books for the user
        """
        if user_id not in self.user_similarity_df.index:
            print(f"User {user_id} not found in the dataset.")
            return []
        
        # Get similarity scores for the user
        similar_users = self.user_similarity_df[user_id].sort_values(ascending=False).drop(user_id)
        
        # Filter users with similarity above threshold
        similar_users = similar_users[similar_users >= sim_threshold]
        
        if len(similar_users) == 0:
            print(f"No users with similarity above {sim_threshold} found.")
            return []
        
        # Get books that the user hasn't rated yet
        user_books = set(self.user_book_matrix.loc[user_id][self.user_book_matrix.loc[user_id] > 0].index)
        unrated_books = set(self.user_book_matrix.columns) - user_books
        
        # Aggregate ratings from similar users, weighted by similarity
        recommended_books = {}
        for sim_user, similarity in similar_users.items():
            for book in unrated_books:
                rating = self.user_book_matrix.loc[sim_user, book]
                if rating > 0:  # Only consider actual ratings (not missing values)
                    recommended_books[book] = recommended_books.get(book, 0) + rating * similarity
        
        # Normalize scores by sum of similarities
        for book in recommended_books:
            book_raters = [u for u in similar_users.index 
                           if self.user_book_matrix.loc[u, book] > 0]
            sim_sum = sum(similar_users[u] for u in book_raters)
            if sim_sum > 0:
                recommended_books[book] /= sim_sum
        
        # Sort books by aggregated score and return top recommendations
        recommended_books = sorted(recommended_books.items(), key=lambda x: x[1], reverse=True)
        
        return recommended_books[:top_n]
    
    def item_based_recommend(self, user_id, top_n=5, min_rating=0):
        """
        Generate recommendations using item-based collaborative filtering.
        
        Args:
            user_id: ID of the user to recommend for
            top_n (int): Number of recommendations to return
            min_rating (float): Minimum rating for items to be considered
            
        Returns:
            list: Top N recommended books for the user
        """
        if user_id not in self.user_book_matrix.index:
            print(f"User {user_id} not found in the dataset.")
            return []
        
        # Get books that the user has rated
        user_ratings = self.user_book_matrix.loc[user_id]
        rated_books = user_ratings[user_ratings > min_rating].index.tolist()
        
        if not rated_books:
            print(f"User {user_id} has no ratings above {min_rating}.")
            return []
        
        # Get books the user hasn't rated
        unrated_books = user_ratings[user_ratings == 0].index.tolist()
        
        # Calculate scores for each unrated book based on its similarity to books the user liked
        book_scores = defaultdict(float)
        book_sim_sums = defaultdict(float)
        
        for book in unrated_books:
            for rated_book in rated_books:
                similarity = self.book_similarity_df.loc[book, rated_book]
                rating = user_ratings[rated_book]
                
                # Weighted sum of ratings and similarities
                book_scores[book] += similarity * rating
                book_sim_sums[book] += abs(similarity)
            
            # Normalize by sum of similarities
            if book_sim_sums[book] > 0:
                book_scores[book] /= book_sim_sums[book]
        
        # Sort by score and return top_n
        recommended_books = sorted(book_scores.items(), key=lambda x: x[1], reverse=True)
        return recommended_books[:top_n]
    
    def hybrid_recommend(self, user_id, top_n=5, user_weight=0.5):
        """
        Generate hybrid recommendations combining user-based and item-based approaches.
        
        Args:
            user_id: ID of the user to recommend for
            top_n (int): Number of recommendations to return
            user_weight (float): Weight for user-based recommendations (0 to 1)
            
        Returns:
            list: Top N recommended books for the user
        """
        # Get recommendations from both approaches
        user_based_recs = dict(self.user_based_recommend(user_id, top_n=top_n*2))
        item_based_recs = dict(self.item_based_recommend(user_id, top_n=top_n*2))
        
        # Normalize scores within each approach
        def normalize_scores(scores_dict):
            if not scores_dict:
                return {}
            max_score = max(scores_dict.values())
            return {k: v/max_score for k, v in scores_dict.items()}
        
        user_based_recs = normalize_scores(user_based_recs)
        item_based_recs = normalize_scores(item_based_recs)
        
        # Combine scores
        all_books = set(user_based_recs.keys()) | set(item_based_recs.keys())
        hybrid_scores = {}
        
        for book in all_books:
            user_score = user_based_recs.get(book, 0)
            item_score = item_based_recs.get(book, 0)
            hybrid_scores[book] = user_weight * user_score + (1 - user_weight) * item_score
        
        # Sort and return top recommendations
        hybrid_recs = sorted(hybrid_scores.items(), key=lambda x: x[1], reverse=True)
        return hybrid_recs[:top_n]
    
    def content_based_recommend(self, user_id, book_features, top_n=5):
        """
        Generate content-based recommendations using book features.
        
        Args:
            user_id: ID of the user to recommend for
            book_features (DataFrame): DataFrame with book_title index and feature columns
            top_n (int): Number of recommendations to return
            
        Returns:
            list: Top N recommended books for the user
        """
        if book_features is None:
            print("No book features provided.")
            return []
            
        if user_id not in self.user_book_matrix.index:
            print(f"User {user_id} not found in the dataset.")
            return []
        
        # Get user's rated books and their ratings
        user_ratings = self.user_book_matrix.loc[user_id]
        rated_books = user_ratings[user_ratings > 0]
        
        if len(rated_books) == 0:
            print(f"User {user_id} has no ratings.")
            return []
        
        # Calculate user profile based on books they've rated
        user_profile = pd.Series(0, index=book_features.columns)
        
        for book, rating in rated_books.items():
            if book in book_features.index:
                user_profile += book_features.loc[book] * rating
        
        # Normalize user profile
        user_profile = user_profile / rated_books.sum()
        
        # Calculate similarity between user profile and all books
        book_scores = {}
        for book in book_features.index:
            if book not in rated_books:
                # Calculate cosine similarity between user profile and book features
                similarity = np.dot(user_profile, book_features.loc[book]) / (
                    np.linalg.norm(user_profile) * np.linalg.norm(book_features.loc[book])
                )
                book_scores[book] = similarity
        
        # Sort and return top recommendations
        recommended_books = sorted(book_scores.items(), key=lambda x: x[1], reverse=True)
        return recommended_books[:top_n]
    
    def popularity_based_recommend(self, top_n=5, min_ratings=2):
        """
        Generate recommendations based on book popularity.
        
        Args:
            top_n (int): Number of recommendations to return
            min_ratings (int): Minimum number of ratings for a book to be considered
            
        Returns:
            list: Top N most popular books
        """
        if self.mean_ratings is None or self.popular_books is None:
            print("Mean ratings or popularity data not available.")
            return []
        
        # Filter books with minimum number of ratings
        popular_with_min = self.popular_books[self.popular_books >= min_ratings]
        
        # Get mean ratings for these books
        popular_books_ratings = self.mean_ratings[popular_with_min.index]
        
        # Sort by rating and return top N
        top_books = popular_books_ratings.sort_values(ascending=False).head(top_n)
        return [(book, rating) for book, rating in top_books.items()]
    
    def evaluate_recommendations(self, test_users=None, n_recommendations=5, method='user'):
        """
        Evaluate the recommendation system using precision, recall, and other metrics.
        
        Args:
            test_users (list): List of user IDs to evaluate on (default: all users)
            n_recommendations (int): Number of recommendations to generate
            method (str): Recommendation method to evaluate ('user', 'item', 'hybrid')
            
        Returns:
            dict: Dictionary of evaluation metrics
        """
        if test_users is None:
            test_users = self.user_book_matrix.index.tolist()
        
        # For a simple evaluation, let's use leave-one-out cross-validation
        precision_at_k = []
        recall_at_k = []
        
        for user in tqdm(test_users, desc="Evaluating recommendations"):
            # Get the user's actual ratings
            actual_ratings = self.user_book_matrix.loc[user]
            liked_books = set(actual_ratings[actual_ratings >= 4].index)
            
            if len(liked_books) == 0:
                continue
            
            # Temporarily remove one liked book for testing
            temp_matrix = self.user_book_matrix.copy()
            test_book = np.random.choice(list(liked_books))
            temp_matrix.loc[user, test_book] = 0
            
            # Generate recommendations
            if method == 'user':
                recommendations = self.user_based_recommend(user, top_n=n_recommendations)
            elif method == 'item':
                recommendations = self.item_based_recommend(user, top_n=n_recommendations)
            elif method == 'hybrid':
                recommendations = self.hybrid_recommend(user, top_n=n_recommendations)
            else:
                print(f"Unknown method: {method}")
                return {}
            
            recommended_books = [book for book, _ in recommendations]
            
            # Calculate precision and recall
            hits = len(set(recommended_books) & {test_book})
            precision = hits / len(recommended_books) if recommended_books else 0
            recall = hits / 1  # Only one test item
            
            precision_at_k.append(precision)
            recall_at_k.append(recall)
        
        metrics = {
            'precision': np.mean(precision_at_k),
            'recall': np.mean(recall_at_k),
            'f1_score': 2 * np.mean(precision_at_k) * np.mean(recall_at_k) / 
                      (np.mean(precision_at_k) + np.mean(recall_at_k)) 
                      if (np.mean(precision_at_k) + np.mean(recall_at_k)) > 0 else 0
        }
        
        return metrics
    
    def visualize_user_similarity(self, user_id=None, top_n=10):
        """
        Visualize the similarity between users.
        
        Args:
            user_id: Optional specific user to visualize similarities for
            top_n (int): Number of most similar users to display
        """
        if self.user_similarity_df is None:
            print("User similarity matrix not available.")
            return
        
        plt.figure(figsize=(12, 8))
        
        if user_id is not None and user_id in self.user_similarity_df.index:
            # Get similarities for specific user
            similarities = self.user_similarity_df[user_id].sort_values(ascending=False).drop(user_id)
            top_similar = similarities.head(top_n)
            
            # Create bar chart
            plt.bar(top_similar.index.astype(str), top_similar.values)
            plt.title(f'Top {top_n} Similar Users to User {user_id}')
            plt.xlabel('User ID')
            plt.ylabel('Similarity Score')
            plt.xticks(rotation=45)
        else:
            # Create heatmap of user similarities
            mask = np.triu(np.ones_like(self.user_similarity_df, dtype=bool))
            sns.heatmap(
                self.user_similarity_df, 
                mask=mask, 
                cmap='viridis', 
                annot=False, 
                square=True,
                cbar_kws={'label': 'Similarity Score'}
            )
            plt.title('User Similarity Heatmap')
        
        plt.tight_layout()
        plt.show()
    
    def visualize_recommendations(self, user_id, method='hybrid', top_n=5):
        """
        Visualize recommendations for a user with their predicted scores.
        
        Args:
            user_id: ID of the user to visualize recommendations for
            method (str): Recommendation method ('user', 'item', 'hybrid')
            top_n (int): Number of recommendations to display
        """
        if method == 'user':
            recommendations = self.user_based_recommend(user_id, top_n=top_n)
        elif method == 'item':
            recommendations = self.item_based_recommend(user_id, top_n=top_n)
        elif method == 'hybrid':
            recommendations = self.hybrid_recommend(user_id, top_n=top_n)
        else:
            print(f"Unknown recommendation method: {method}")
            return
        
        if not recommendations:
            print(f"No recommendations found for User {user_id} using {method} method.")
            return
        
        books, scores = zip(*recommendations)
        
        plt.figure(figsize=(12, 6))
        plt.bar(books, scores, color='cornflowerblue')
        plt.title(f'Top {top_n} Recommended Books for User {user_id} ({method.capitalize()}-based)')
        plt.xlabel('Book Title')
        plt.ylabel('Recommendation Score')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()
    
    def save_model(self, filepath):
        """
        Save the recommendation model to a file.
        
        Args:
            filepath (str): Path to save the model
        """
        model_data = {
            'user_book_matrix': self.user_book_matrix,
            'user_similarity': self.user_similarity_df,
            'book_similarity': self.book_similarity_df,
            'mean_ratings': self.mean_ratings,
            'popular_books': self.popular_books
        }
        
        try:
            with open(filepath, 'wb') as f:
                pickle.dump(model_data, f)
            print(f"Model saved successfully to {filepath}")
            return True
        except Exception as e:
            print(f"Error saving model: {str(e)}")
            return False
    
    def load_model(self, filepath):
        """
        Load a recommendation model from a file.
        
        Args:
            filepath (str): Path to the saved model
        """
        try:
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)
            
            self.user_book_matrix = model_data['user_book_matrix']
            self.user_similarity_df = model_data['user_similarity']
            self.book_similarity_df = model_data['book_similarity']
            self.mean_ratings = model_data['mean_ratings']
            self.popular_books = model_data['popular_books']
            
            # Reconstruct numpy arrays from DataFrames
            self.user_similarity = self.user_similarity_df.values
            self.book_similarity = self.book_similarity_df.values
            
            print(f"Model loaded successfully from {filepath}")
            return True
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            return False


# Function to generate synthetic book features for demo purposes
def generate_book_features(book_titles, n_features=5):
    """
    Generate synthetic features for books for content-based filtering demo.
    
    Args:
        book_titles (list): List of book titles
        n_features (int): Number of features to generate
        
    Returns:
        DataFrame: DataFrame with book titles as index and feature columns
    """
    feature_names = [f'feature_{i}' for i in range(1, n_features+1)]
    features = np.random.rand(len(book_titles), n_features)
    
    # Normalize features
    features = StandardScaler().fit_transform(features)
    
    return pd.DataFrame(features, index=book_titles, columns=feature_names)


# Demo with sample data
if __name__ == "__main__":
    # Sample dataset
    data = {
        'user_id': [1, 1, 1, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 7, 7],
        'book_title': [
            'Book A', 'Book B', 'Book C', 'Book A', 'Book D', 
            'Book B', 'Book C', 'Book E', 'Book A', 'Book C',
            'Book F', 'Book D', 'Book E', 'Book F', 'Book G',
            'Book A', 'Book G', 'Book H', 'Book C', 'Book H'
        ],
        'rating': [5, 3, 4, 4, 5, 5, 3, 4, 3, 2, 5, 4, 3, 5, 2, 4, 3, 5, 4, 5]
    }
    
    # Initialize recommendation system
    recommender = BookRecommendationSystem(data)
    
    # Preprocess data
    recommender.preprocess_data()
    
    # Compute similarity matrices
    recommender.compute_similarity_matrices(method='cosine')
    
    # Generate user-based recommendations
    user_id = 1
    print(f"\nUser-based recommendations for User {user_id}:")
    user_recs = recommender.user_based_recommend(user_id, top_n=3)
    for book, score in user_recs:
        print(f"- {book}: {score:.3f}")
    
    # Generate item-based recommendations
    print(f"\nItem-based recommendations for User {user_id}:")
    item_recs = recommender.item_based_recommend(user_id, top_n=3)
    for book, score in item_recs:
        print(f"- {book}: {score:.3f}")
    
    # Generate hybrid recommendations
    print(f"\nHybrid recommendations for User {user_id}:")
    hybrid_recs = recommender.hybrid_recommend(user_id, top_n=3, user_weight=0.6)
    for book, score in hybrid_recs:
        print(f"- {book}: {score:.3f}")
    
    # Generate synthetic book features for content-based filtering
    book_titles = recommender.user_book_matrix.columns.tolist()
    book_features = generate_book_features(book_titles)
    
    # Generate content-based recommendations
    print(f"\nContent-based recommendations for User {user_id}:")
    content_recs = recommender.content_based_recommend(user_id, book_features, top_n=3)
    for book, score in content_recs:
        print(f"- {book}: {score:.3f}")
    
    # Get popular books
    print("\nMost popular books:")
    popular_books = recommender.popularity_based_recommend(top_n=3)
    for book, score in popular_books:
        print(f"- {book}: {score:.3f}")
    
    # Evaluate recommendations
    print("\nEvaluating recommendation methods...")
    user_metrics = recommender.evaluate_recommendations(method='user', n_recommendations=3)
    item_metrics = recommender.evaluate_recommendations(method='item', n_recommendations=3)
    hybrid_metrics = recommender.evaluate_recommendations(method='hybrid', n_recommendations=3)
    
    print("\nEvaluation Results:")
    print(f"User-based method: Precision={user_metrics['precision']:.3f}, Recall={user_metrics['recall']:.3f}")
    print(f"Item-based method: Precision={item_metrics['precision']:.3f}, Recall={item_metrics['recall']:.3f}")
    print(f"Hybrid method: Precision={hybrid_metrics['precision']:.3f}, Recall={hybrid_metrics['recall']:.3f}")
    
    # Save model
    recommender.save_model('book_recommender_model.pkl')
    
    print("\nRecommendation system demo complete!")