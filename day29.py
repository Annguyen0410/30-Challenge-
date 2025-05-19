# Import Libraries
import pandas as pd
import numpy as np
from surprise import SVD, Dataset, Reader, accuracy, BaselineOnly
from surprise.model_selection import cross_validate, train_test_split, GridSearchCV, KFold
from surprise.dump import dump, load as load_model
import seaborn as sns
import matplotlib.pyplot as plt
from collections import defaultdict
import os
import random
from sklearn.metrics.pairwise import pairwise_distances # For Jaccard

# --- Configuration & Constants ---
RANDOM_STATE = 42
TEST_SIZE = 0.2 # For initial train/test split for hyperparameter tuning and basic eval
N_SPLITS_CV = 3 # For K-Fold cross-validation for P@k, R@k
MIN_USER_RATINGS_FILTER = 10 # Initial data filtering
MIN_MOVIE_RATINGS_FILTER = 5  # Initial data filtering
TOP_N_RECOMMENDATIONS = 10
MODEL_FILENAME = 'svd_recommender_model_v3.pkl'
RELEVANCE_THRESHOLD_FOR_PREC_RECALL = 4.0 # Ratings >= this are considered relevant for P@k, R@k

# --- Seeding for Reproducibility ---
np.random.seed(RANDOM_STATE)
random.seed(RANDOM_STATE)

# --- Helper Functions ---
def get_movie_metadata(movie_id, movies_df, field='title'):
    """Fetches movie metadata (title or genres) given its ID."""
    try:
        if field == 'genres_list':
            genres_str = movies_df.loc[movies_df['movieId'] == movie_id, 'genres'].iloc[0]
            return genres_str.split('|')
        return movies_df.loc[movies_df['movieId'] == movie_id, field].iloc[0]
    except IndexError:
        return "Unknown Movie" if field == 'title' else []

def get_user_genre_preferences(user_id, trainset, movies_df, top_n_genres=3, min_rating_for_pref=4.0):
    """Identifies a user's preferred genres based on their highly-rated movies in the training set."""
    try:
        user_inner_id = trainset.to_inner_uid(user_id)
    except ValueError: return []

    genre_counts = defaultdict(int)
    items_considered = 0
    for item_inner_id, rating in trainset.ur[user_inner_id]:
        if rating >= min_rating_for_pref:
            movie_id = trainset.to_raw_iid(item_inner_id)
            genres = get_movie_metadata(movie_id, movies_df, 'genres_list')
            for genre in genres:
                if genre != "(no genres listed)": genre_counts[genre] += 1
            items_considered +=1
    if not items_considered: return []
    return [genre for genre, _ in sorted(genre_counts.items(), key=lambda x: x[1], reverse=True)[:top_n_genres]]

def jaccard_similarity(list1, list2):
    """Computes Jaccard similarity between two lists (sets of genres)."""
    s1 = set(list1)
    s2 = set(list2)
    if not s1 and not s2: return 0.0 # Or 1.0 if both empty means identical
    intersection = len(s1.intersection(s2))
    union = len(s1.union(s2))
    return intersection / union if union != 0 else 0.0

def get_content_similar_movies(target_movie_id, movies_df, top_n=5):
    """Recommends movies similar to target_movie_id based on genre Jaccard similarity."""
    if target_movie_id not in movies_df['movieId'].values:
        print(f"Movie ID {target_movie_id} not found in movies metadata.")
        return []

    target_genres = set(get_movie_metadata(target_movie_id, movies_df, 'genres_list'))
    if not target_genres or "(no genres listed)" in target_genres:
        print(f"Target movie {get_movie_metadata(target_movie_id, movies_df)} has no listed genres for comparison.")
        return []

    similarities = []
    for _, row in movies_df.iterrows():
        if row['movieId'] == target_movie_id:
            continue
        current_genres = set(row['genres'].split('|'))
        sim = jaccard_similarity(target_genres, current_genres)
        if sim > 0: # Only consider movies with some genre overlap
            similarities.append({'movieId': row['movieId'], 'title': row['title'], 'genres': row['genres'], 'similarity': sim})

    similarities.sort(key=lambda x: x['similarity'], reverse=True)
    return similarities[:top_n]


# --- Core Logic Functions ---

def load_and_preprocess_data(min_user_r, min_movie_r):
    print("Step 1: Loading and Preprocessing Datasets...")
    ratings_df = pd.read_csv('ratings.csv')
    movies_df = pd.read_csv('movies.csv')
    ratings_df.drop('timestamp', axis=1, inplace=True)

    print(f"\nOriginal ratings: {ratings_df.shape[0]} ratings, {ratings_df['userId'].nunique()} users, {ratings_df['movieId'].nunique()} movies.")
    print(f"Original movies metadata: {movies_df.shape[0]} movies.")

    # Filtering
    print(f"\nApplying filters: min {min_user_r} ratings/user, min {min_movie_r} ratings/movie")
    while True: # Iteratively filter until stable
        user_counts = ratings_df['userId'].value_counts()
        movie_counts = ratings_df['movieId'].value_counts()
        
        can_filter_users = (user_counts < min_user_r).any()
        can_filter_movies = (movie_counts < min_movie_r).any()

        if not can_filter_users and not can_filter_movies:
            break

        if can_filter_users:
            active_users = user_counts[user_counts >= min_user_r].index
            ratings_df = ratings_df[ratings_df['userId'].isin(active_users)]
        
        if can_filter_movies:
            # Update movie_counts after user filtering
            movie_counts = ratings_df['movieId'].value_counts()
            active_movies = movie_counts[movie_counts >= min_movie_r].index
            ratings_df = ratings_df[ratings_df['movieId'].isin(active_movies)]
    
    df_filtered = ratings_df
    movies_df_filtered = movies_df[movies_df['movieId'].isin(df_filtered['movieId'].unique())]

    print(f"Filtered ratings: {df_filtered.shape[0]} ratings, {df_filtered['userId'].nunique()} users, {df_filtered['movieId'].nunique()} movies.")
    if df_filtered.empty: raise ValueError("Dataset empty after filtering.")

    reader = Reader(rating_scale=(0.5, 5))
    data = Dataset.load_from_df(df_filtered[['userId', 'movieId', 'rating']], reader)
    return data, df_filtered, movies_df_filtered, pd.read_csv('ratings.csv'), movies_df # Return original too

def perform_eda(ratings_df_full, movies_df_full, ratings_df_filtered):
    print("\nStep 2: Exploratory Data Analysis...")
    # (Using previously defined EDA functions, ensuring clear titles and labels)
    # ... (Rating distribution, ratings per user/movie, genre distribution) ...
    # Example addition: Sparsity
    sparsity = 1.0 - (len(ratings_df_full) / (ratings_df_full['userId'].nunique() * ratings_df_full['movieId'].nunique()))
    print(f"\nSparsity of the original ratings matrix: {sparsity:.4f}")
    if not ratings_df_filtered.empty:
        sparsity_filtered = 1.0 - (len(ratings_df_filtered) / (ratings_df_filtered['userId'].nunique() * ratings_df_filtered['movieId'].nunique()))
        print(f"Sparsity of the filtered ratings matrix: {sparsity_filtered:.4f}")

    # Visualizing impact of filtering (example)
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    sns.histplot(ratings_df_full['userId'].value_counts(), ax=axes[0], bins=50, kde=True, color='blue')
    axes[0].set_title('Ratings per User (Original)')
    axes[0].set_xlim(0, ratings_df_full['userId'].value_counts().quantile(0.95))
    if not ratings_df_filtered.empty:
        sns.histplot(ratings_df_filtered['userId'].value_counts(), ax=axes[1], bins=50, kde=True, color='green')
        axes[1].set_title('Ratings per User (Filtered)')
        axes[1].set_xlim(0, ratings_df_filtered['userId'].value_counts().quantile(0.95))
    plt.tight_layout()
    plt.show()


def train_evaluate_rating_prediction(data, model_filename, retrain_if_model_exists=False):
    print("\nStep 3: Training and Evaluating Rating Prediction Models (RMSE/MAE)...")
    
    # Split data for this evaluation part
    trainset_eval, testset_eval = train_test_split(data, test_size=TEST_SIZE, random_state=RANDOM_STATE)

    # BaselineOnly
    print("\n--- BaselineOnly Model ---")
    baseline_model = BaselineOnly()
    baseline_model.fit(trainset_eval)
    baseline_predictions = baseline_model.test(testset_eval)
    print("BaselineOnly Evaluation:")
    accuracy.rmse(baseline_predictions, verbose=True)
    accuracy.mae(baseline_predictions, verbose=True)

    # SVD
    print("\n--- SVD Model ---")
    svd_model = None
    if os.path.exists(model_filename) and not retrain_if_model_exists:
        print(f"Loading pre-tuned SVD model from {model_filename}...")
        _, svd_model = load_model(model_filename)
    else:
        print("Performing GridSearchCV for SVD...")
        param_grid = {'n_epochs': [20, 30], 'lr_all': [0.005, 0.007], 'reg_all': [0.02, 0.04], 'n_factors': [50, 100]}
        gs = GridSearchCV(SVD, param_grid, measures=['rmse', 'mae'], cv=N_SPLITS_CV, n_jobs=-1, joblib_verbose=1)
        gs.fit(data) # GS fits on the whole 'data' object
        print(f"SVD Best RMSE (from GS): {gs.best_score['rmse']:.4f}")
        print("SVD Best parameters (for RMSE):", gs.best_params['rmse'])
        svd_model = SVD(**gs.best_params['rmse']) # Instantiate with best params
        # We will fit this on the specific trainset_eval for this section's evaluation
        # And later on full_trainset for final recommendations and P@k/R@k
        print(f"Saving tuned SVD model (parameters) to {model_filename}...")
        dump(model_filename, algo=svd_model) # Saves the algorithm object with its parameters

    # Fit the (potentially loaded or newly parameterized) SVD model on trainset_eval
    svd_model.fit(trainset_eval)
    print("\nSVD Model Evaluation on Test Set (for RMSE/MAE):")
    svd_predictions = svd_model.test(testset_eval)
    rmse_svd = accuracy.rmse(svd_predictions, verbose=True)
    accuracy.mae(svd_predictions, verbose=True)
    return svd_model # Return model trained on trainset_eval for immediate error analysis if needed


def evaluate_top_n_recommendations(algo, data, movies_df, n_rec=TOP_N_RECOMMENDATIONS, relevance_threshold=RELEVANCE_THRESHOLD_FOR_PREC_RECALL):
    """
    Evaluates top-N recommendations using Precision@k and Recall@k with K-fold cross-validation.
    `algo` should be an unfitted algorithm instance (e.g., SVD(**params)).
    `data` is the full Dataset object.
    """
    print(f"\nStep 4: Evaluating Top-N Recommendations (Precision@{n_rec}, Recall@{n_rec})...")
    kf = KFold(n_splits=N_SPLITS_CV, random_state=RANDOM_STATE, shuffle=True)
    
    precisions = []
    recalls = []

    for fold_idx, (train_idx, test_idx) in enumerate(kf.split(data)):
        print(f"Fold {fold_idx+1}/{N_SPLITS_CV}...")
        # Construct trainset and testset for this fold
        # Surprise KFold gives indices for its internal raw ratings list
        raw_ratings = data.raw_ratings
        train_raw_ratings = [raw_ratings[i] for i in train_idx]
        test_raw_ratings = [raw_ratings[i] for i in test_idx]

        # Create temporary Dataset objects for this fold
        # Reader must be the same as used for the main 'data' object
        reader = data.reader 
        train_data_fold = Dataset.load_from_folds([(train_raw_ratings, [])], reader) # Pass test part as empty
        trainset_fold = train_data_fold.build_full_trainset() # This is how to get a trainset from raw ratings
        
        # The testset needs to be in the format Surprise expects: list of (uid, iid, rating) tuples
        # We only use the test_raw_ratings to define what's "relevant" for a user
        # Predictions will be made for items NOT in trainset_fold

        algo.fit(trainset_fold)

        # Prepare test data for prediction (items user rated, but not in current train fold)
        # And get ground truth relevant items
        # Group test_raw_ratings by user
        user_true_relevant_items_fold = defaultdict(list)
        for uid, iid, rating, _ in test_raw_ratings:
            if rating >= relevance_threshold:
                user_true_relevant_items_fold[uid].append(iid)
        
        fold_precisions = []
        fold_recalls = []
        
        users_in_fold_test = list(user_true_relevant_items_fold.keys())
        if not users_in_fold_test:
            print(f"  No users with relevant items in test set for fold {fold_idx+1}. Skipping P/R calculation for this fold.")
            continue

        for user_id in users_in_fold_test:
            if not user_true_relevant_items_fold[user_id]: # No relevant items for this user in this fold's test set
                continue

            # Get items rated by user in trainset_fold to exclude them from prediction candidates
            try:
                user_inner_id = trainset_fold.to_inner_uid(user_id)
                rated_in_train_fold_raw_iids = {trainset_fold.to_raw_iid(inner_iid) for inner_iid, _ in trainset_fold.ur[user_inner_id]}
            except ValueError: # User not in trainset_fold (e.g. all their ratings went to test)
                rated_in_train_fold_raw_iids = set()

            # Predict for all items in movies_df NOT in rated_in_train_fold_raw_iids
            items_to_predict_for = movies_df[~movies_df['movieId'].isin(rated_in_train_fold_raw_iids)]['movieId'].unique()
            
            predictions_for_user = []
            for movie_id_to_pred in items_to_predict_for:
                # Ensure movie_id_to_pred is known to the trainset_fold (i.e., has an inner id)
                # If not, SVD might predict global average.
                # We only care about movies the model *can* give a specific prediction for.
                # However, for a fair eval, predict for all potential candidates.
                pred_obj = algo.predict(user_id, movie_id_to_pred)
                predictions_for_user.append((pred_obj.iid, pred_obj.est))
            
            predictions_for_user.sort(key=lambda x: x[1], reverse=True)
            recommended_item_ids = [iid for iid, est in predictions_for_user[:n_rec]]

            # Calculate P@k, R@k
            true_relevant = set(user_true_relevant_items_fold[user_id])
            recommended_set = set(recommended_item_ids)
            
            num_hit = len(true_relevant.intersection(recommended_set))
            
            if recommended_set: # Avoid division by zero if no recommendations made (shouldn't happen with SVD)
                precision_at_k = num_hit / len(recommended_set)
            else:
                precision_at_k = 0.0
            
            if true_relevant: # Avoid division by zero if no relevant items in test set for user
                recall_at_k = num_hit / len(true_relevant)
            else: # This case should be filtered by `if not user_true_relevant_items_fold[user_id]:`
                recall_at_k = 0.0 
                
            fold_precisions.append(precision_at_k)
            fold_recalls.append(recall_at_k)

        if fold_precisions: precisions.append(np.mean(fold_precisions))
        if fold_recalls: recalls.append(np.mean(fold_recalls))
        print(f"  Fold {fold_idx+1} Avg Precision@{n_rec}: {np.mean(fold_precisions):.4f} | Avg Recall@{n_rec}: {np.mean(fold_recalls):.4f}")

    avg_precision = np.mean(precisions) if precisions else 0.0
    avg_recall = np.mean(recalls) if recalls else 0.0
    print(f"\nAverage Precision@{n_rec} across folds: {avg_precision:.4f}")
    print(f"Average Recall@{n_rec} across folds: {avg_recall:.4f}")
    return avg_precision, avg_recall


def analyze_prediction_error_vs_user_activity(model, testset, ratings_df_filtered):
    print("\nStep 5: Analyzing Prediction Error vs. User Activity...")
    if not testset: return

    predictions = model.test(testset) # model should be trained on the corresponding trainset for this testset
    user_errors = defaultdict(list)
    for pred in predictions:
        user_errors[pred.uid].append(abs(pred.r_ui - pred.est))
    
    avg_user_error = {uid: np.mean(errors) for uid, errors in user_errors.items()}
    
    user_activity = ratings_df_filtered['userId'].value_counts().to_dict()
    
    data_for_plot = []
    for uid, avg_err in avg_user_error.items():
        if uid in user_activity:
            data_for_plot.append({'userId': uid, 'avg_error': avg_err, 'num_ratings': user_activity[uid]})
    
    if not data_for_plot:
        print("Not enough data to plot error vs activity.")
        return

    error_df = pd.DataFrame(data_for_plot)
    
    # Create bins for number of ratings
    bins = [0, MIN_USER_RATINGS_FILTER+10, 50, 100, 200, error_df['num_ratings'].max()]
    labels = [f'{bins[i]}-{bins[i+1]-1}' for i in range(len(bins)-1)]
    error_df['rating_group'] = pd.cut(error_df['num_ratings'], bins=bins, labels=labels, right=False)
    
    plt.figure(figsize=(12, 7))
    sns.boxplot(x='rating_group', y='avg_error', data=error_df, palette="coolwarm")
    plt.title('Average Prediction Error (RMSE) vs. Number of User Ratings')
    plt.xlabel('Number of Ratings by User (Grouped)')
    plt.ylabel('Average Absolute Error for User')
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()


def generate_recommendations_showcase(model, trainset, movies_df, ratings_df_full, example_user_id=None, new_user_simulated_prefs=None):
    print("\nStep 6: Generating Recommendations Showcase...")
    
    # Pick example user if none provided
    if example_user_id is None and trainset.n_users > 0:
        all_raw_uids = [trainset.to_raw_uid(inner_uid) for inner_uid in trainset.all_users()]
        example_user_id = random.choice(all_raw_uids) if all_raw_uids else None
    
    if example_user_id:
        print(f"\n--- Recommendations for Existing User ID: {example_user_id} ---")
        user_prefs = get_user_genre_preferences(example_user_id, trainset, movies_df)
        print(f"  User's inferred preferred genres (from high ratings): {user_prefs if user_prefs else 'None identified'}")

        # Get CF recommendations
        # We need to get items not rated by the user in the trainset
        user_inner_id = trainset.to_inner_uid(example_user_id)
        rated_movie_ids_raw = {trainset.to_raw_iid(inner_id) for inner_id, _ in trainset.ur[user_inner_id]}
        
        all_movie_ids_in_db = movies_df['movieId'].unique()
        unrated_movie_ids = [m_id for m_id in all_movie_ids_in_db if m_id not in rated_movie_ids_raw]

        predictions = []
        for movie_id in unrated_movie_ids:
            predictions.append(model.predict(example_user_id, movie_id))

        # Sort by estimate
        predictions.sort(key=lambda p: p.est, reverse=True)
        
        print("\n  Top CF Recommendations (SVD):")
        for i, pred in enumerate(predictions[:TOP_N_RECOMMENDATIONS]):
            title = get_movie_metadata(pred.iid, movies_df)
            genres = get_movie_metadata(pred.iid, movies_df, 'genres_list')
            print(f"  {i+1}. {title} (Pred: {pred.est:.2f}, Genres: {', '.join(genres)})")

        # Hybrid-like: Boost/filter based on user_prefs (simple post-processing)
        print("\n  CF Recommendations Re-ranked/Boosted by Preferred Genres:")
        boosted_predictions = []
        for pred in predictions:
            boost = 0
            movie_genres = get_movie_metadata(pred.iid, movies_df, 'genres_list')
            if user_prefs:
                for pref_genre in user_prefs:
                    if pref_genre in movie_genres:
                        boost += 0.2 # Arbitrary boost factor
            
            boosted_est = np.clip(pred.est + boost, trainset.rating_scale[0], trainset.rating_scale[1])
            boosted_predictions.append({'pred': pred, 'boosted_est': boosted_est, 'genres': movie_genres})

        boosted_predictions.sort(key=lambda x: x['boosted_est'], reverse=True)
        for i, item in enumerate(boosted_predictions[:TOP_N_RECOMMENDATIONS]):
            title = get_movie_metadata(item['pred'].iid, movies_df)
            print(f"  {i+1}. {title} (Boosted Pred: {item['boosted_est']:.2f}, Original Pred: {item['pred'].est:.2f}, Genres: {', '.join(item['genres'])})")
    else:
        print("No example user ID provided or trainset is empty.")

    # Cold Start User (with simulated genre preferences)
    print(f"\n--- Recommendations for NEW User (Cold Start) ---")
    if new_user_simulated_prefs is None: new_user_simulated_prefs = ["Action", "Sci-Fi"] # Example
    print(f"  Simulated new user preferred genres: {new_user_simulated_prefs}")
    
    # Recommend popular movies from these genres
    candidate_movies = []
    for _, row in movies_df.iterrows():
        movie_genres = row['genres'].split('|')
        if any(pref_g in movie_genres for pref_g in new_user_simulated_prefs):
            candidate_movies.append(row['movieId'])
    
    if candidate_movies:
        # Get average ratings for these candidates
        movie_stats = ratings_df_full[ratings_df_full['movieId'].isin(candidate_movies)].groupby('movieId')['rating'].agg(['count', 'mean'])
        # Filter for movies with a minimum number of ratings for reliability
        popular_in_genre = movie_stats[movie_stats['count'] >= MIN_MOVIE_RATINGS_FILTER].sort_values(by=['mean', 'count'], ascending=[False, False])
        
        print(f"\n  Top Popular Movies from Preferred Genres ({', '.join(new_user_simulated_prefs)}):")
        count = 0
        for movie_id, stats_row in popular_in_genre.head(TOP_N_RECOMMENDATIONS).iterrows():
            title = get_movie_metadata(movie_id, movies_df)
            genres = get_movie_metadata(movie_id, movies_df, 'genres_list')
            print(f"  {count+1}. {title} (Avg Rating: {stats_row['mean']:.2f} from {stats_row['count']} ratings, Genres: {', '.join(genres)})")
            count += 1
            if count >= TOP_N_RECOMMENDATIONS: break
        if count == 0: print("  No sufficiently popular movies found for these genres.")
    else:
        print("  No movies found matching the new user's preferred genres. Falling back to global popular.")
        # (Insert global popular logic here if needed as a final fallback)


def showcase_item_item_similarity(movies_df_all_meta, example_movie_id=None):
    print("\nStep 7: Item-Item Similarity Showcase ('Because you watched X')...")
    if example_movie_id is None: # Pick a popular movie
        movie_counts = pd.read_csv('ratings.csv')['movieId'].value_counts()
        if not movie_counts.empty:
            example_movie_id = movie_counts.index[0] 
        else:
            print("Could not pick an example movie ID.")
            return
            
    target_title = get_movie_metadata(example_movie_id, movies_df_all_meta)
    target_genres = get_movie_metadata(example_movie_id, movies_df_all_meta, 'genres_list')
    print(f"\nMovies similar to: '{target_title}' (ID: {example_movie_id}, Genres: {target_genres})")
    
    similar_movies = get_content_similar_movies(example_movie_id, movies_df_all_meta, top_n=5)
    if similar_movies:
        for i, movie in enumerate(similar_movies):
            print(f"  {i+1}. {movie['title']} (Genres: {movie['genres']}, Similarity: {movie['similarity']:.2f})")
    else:
        print("  No similar movies found based on genres.")


def discuss_further_improvements_v3():
    print("\nStep 8: Potential Further Improvements & Discussion Points for CV (v3):")
    print("- True Hybrid Models: Factorization Machines (e.g., libFM, LightFM), Deep models incorporating content features directly into embeddings.")
    print("- Sequence-Aware & Session-Based Models: For scenarios where order of interaction matters (e.g., e-commerce, music). (BERT4Rec, GRU4Rec).")
    print("- Advanced Cold Start: Meta-learning (MAML), feature mapping for new items/users to existing latent space.")
    print("- Graph-Based Models: GNNs (Graph Convolutional Networks, PinSage) to model complex relations in user-item graph.")
    print("- Evaluation Metrics for Top-N: NDCG@k, MAP@k for ranked lists. Beyond-accuracy: serendipity, novelty, diversity (e.g., intra-list similarity).")
    print("- Explainability: Methods like SHAP or LIME for SVD-like models (harder), or rule-based explanations for simpler ones.")
    print("- Fairness, Bias, and Transparency: Techniques to detect and mitigate popularity bias, demographic bias. Ensuring model decisions are understandable.")
    print("- Deployment & MLOps: Dockerizing the model, API endpoints (Flask/FastAPI), monitoring performance drift, retraining pipelines.")
    print("- Online Evaluation: A/B testing frameworks for comparing models in a live environment.")


# --- Main Script ---
def main():
    # Step 1: Load and preprocess data
    data, df_filtered, movies_df_filtered, ratings_df_full, movies_df_full = load_and_preprocess_data(
        MIN_USER_RATINGS_FILTER, MIN_MOVIE_RATINGS_FILTER
    )

    # Step 2: EDA
    perform_eda(ratings_df_full, movies_df_full, df_filtered)

    # Step 3: Train and evaluate rating prediction (RMSE/MAE)
    # This model (svd_eval_model) is trained on a split of 'data' for RMSE/MAE eval
    # The parameters found (or loaded) are used later for P@k/R@k and final recs
    svd_eval_model = train_evaluate_rating_prediction(data, MODEL_FILENAME, retrain_if_model_exists=False)

    # Step 4: Evaluate Top-N Recommendations (Precision@k, Recall@k)
    # We need an unfitted SVD instance with the best parameters from GS or loaded model
    if os.path.exists(MODEL_FILENAME):
        _, loaded_algo_obj = load_model(MODEL_FILENAME) # This is an SVD algo instance, possibly fitted
        # Ensure we use an *unfitted* instance with the correct parameters for KFold
        best_svd_params = loaded_algo_obj.get_params()['params']
    else: # Should have run GS in train_evaluate_rating_prediction if model didn't exist
        # This assumes gs_best_params were stored/accessible if model wasn't saved yet
        # For simplicity, let's assume svd_eval_model holds the right params after GS
        best_svd_params = svd_eval_model.get_params()['params']

    svd_algo_for_top_n = SVD(**best_svd_params) # Unfitted, with best params
    evaluate_top_n_recommendations(svd_algo_for_top_n, data, movies_df_filtered) # Use filtered movies for consistency

    # Step 5: Analyze prediction error vs. user activity
    # For this, we need predictions on a testset from a model trained on corresponding trainset
    # We can reuse svd_eval_model (trained on trainset_eval) and testset_eval from train_evaluate_rating_prediction
    _, testset_for_error_analysis = train_test_split(data, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    analyze_prediction_error_vs_user_activity(svd_eval_model, testset_for_error_analysis, df_filtered)

    # Step 6: Generate Recommendations Showcase
    # For final recommendations, train the model on the full filtered dataset ('full_trainset')
    print("\nTraining final SVD model on full filtered dataset for recommendation showcase...")
    full_trainset = data.build_full_trainset()
    final_recommendation_model = SVD(**best_svd_params)
    final_recommendation_model.fit(full_trainset)
    generate_recommendations_showcase(final_recommendation_model, full_trainset, movies_df_filtered, ratings_df_full)

    # Step 7: Item-Item Similarity Showcase
    showcase_item_item_similarity(movies_df_full) # Use full movies metadata for broader item similarity

    # Step 8: Discussion
    discuss_further_improvements_v3()

    # Create requirements.txt
    try:
        import pkg_resources
        installed_packages = {pkg.key for pkg in pkg_resources.working_set}
        required_packages = ['pandas', 'numpy', 'scikit-surprise', 'seaborn', 'matplotlib', 'scikit-learn']
        with open('requirements.txt', 'w') as f:
            for pkg_name in required_packages:
                if pkg_name in installed_packages:
                    version = pkg_resources.get_distribution(pkg_name).version
                    f.write(f"{pkg_name}=={version}\n")
                else: # For surprise, which might be scikit-surprise
                    if pkg_name == 'scikit-surprise' and 'scikit-surprise' in installed_packages:
                         version = pkg_resources.get_distribution('scikit-surprise').version
                         f.write(f"scikit-surprise=={version}\n")
                    else:
                        f.write(f"{pkg_name}\n") # If not found, just list name
        print("\nGenerated requirements.txt")
    except Exception as e:
        print(f"\nCould not generate requirements.txt: {e}")


if __name__ == '__main__':
    main()