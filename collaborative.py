# collaborative.py
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import joblib
import os

def build_collaborative_model():
    """Builds User-Based Collaborative Filtering model"""
    print("🚀 Building Collaborative Filtering Model...")
    
    # 1. Load the data
    print("📥 Loading ratings and movies data...")
    ratings = pd.read_csv('data/ratings.csv')
    movies = pd.read_csv('data/popular_movies.csv')
    
    # Filter only popular movies (that exist in our popular_movies list)
    popular_movie_ids = set(movies['movieId'].values)
    ratings = ratings[ratings['movieId'].isin(popular_movie_ids)]
    print(f"   ✅ Filtered to {len(ratings)} ratings on popular movies")
    
    # 2. Create User-Item Matrix
    # Rows = Users, Columns = Movies, Values = Ratings
    print("📊 Creating User-Item Matrix...")
    user_item_matrix = ratings.pivot_table(
        index='userId',
        columns='movieId',
        values='rating'
    ).fillna(0)
    
    print(f"   ✅ Matrix shape: {user_item_matrix.shape} (Users x Movies)")
    
    # 3. Compute User Similarity
    print("📐 Computing User Similarity...")
    user_similarity = cosine_similarity(user_item_matrix)
    print(f"   ✅ User similarity matrix shape: {user_similarity.shape}")
    
    # 4. Save everything we need
    os.makedirs('models', exist_ok=True)
    joblib.dump(user_similarity, 'models/user_similarity.pkl')
    joblib.dump(user_item_matrix, 'models/user_item_matrix.pkl')
    
    # Create user ID to index mapping
    user_id_to_index = {uid: idx for idx, uid in enumerate(user_item_matrix.index)}
    joblib.dump(user_id_to_index, 'models/user_id_to_index.pkl')
    
    print("\n💾 Collaborative model saved successfully!")
    print("   - models/user_similarity.pkl")
    print("   - models/user_item_matrix.pkl")
    print("   - models/user_id_to_index.pkl")
    
    return user_similarity, user_item_matrix, user_id_to_index, movies

def get_collaborative_recommendations(user_id, user_similarity, user_item_matrix, 
                                      user_id_to_index, movies, top_n=10):
    """Returns top N movie recommendations for a user based on similar users"""
    
    # Check if user exists
    if user_id not in user_id_to_index:
        return f"❌ User {user_id} not found in the database."
    
    # Get the index of the user
    user_idx = user_id_to_index[user_id]
    
    # Get similarity scores for this user with all other users
    similarity_scores = list(enumerate(user_similarity[user_idx]))
    
    # Sort by similarity (descending) and take top 50 similar users
    similar_users = sorted(similarity_scores, key=lambda x: x[1], reverse=True)[1:51]
    
    # Get movies that the target user has NOT rated
    user_ratings = user_item_matrix.iloc[user_idx]
    unseen_movies = user_ratings[user_ratings == 0].index
    
    # Calculate weighted score for each unseen movie
    movie_scores = {}
    for movie_id in unseen_movies:
        weighted_sum = 0
        similarity_sum = 0
        
        for sim_user_idx, similarity in similar_users:
            sim_user_rating = user_item_matrix.iloc[sim_user_idx][movie_id]
            if sim_user_rating > 0:
                weighted_sum += similarity * sim_user_rating
                similarity_sum += similarity
        
        if similarity_sum > 0:
            movie_scores[movie_id] = weighted_sum / similarity_sum
    
    # Sort movies by predicted score
    sorted_movies = sorted(movie_scores.items(), key=lambda x: x[1], reverse=True)
    top_movie_ids = [movie_id for movie_id, score in sorted_movies[:top_n]]
    
    # Get movie details
    available_movies_ids = set(movies['movieId'].values)
    valid_movie_ids = [mid for mid in top_movie_ids if mid in available_movies_ids]
    
    if not valid_movie_ids:
        return "❌ No recommendations available for this user."
    
    recommended = movies[movies['movieId'].isin(top_movie_ids)][['movieId','clean_title', 'year', 'genres']]
    recommended = recommended.rename(columns={
        'clean_title': 'Title',
        'year': 'Year',
        'genres': 'Genres'
    })
    
    # Maintain the order based on scores
    recommended['movieId'] = recommended['movieId'].astype(int)
    recommended = recommended.set_index('movieId').loc[top_movie_ids].reset_index()
    
    recommended = recommended.set_index('movieId').loc[valid_movie_ids].reset_index()

    return recommended

if __name__ == "__main__":
    # Build the model
    user_similarity, user_item_matrix, user_id_to_index, movies = build_collaborative_model()
    
    # Test it out!
    print("\n" + "="*50)
    print("👤 TESTING COLLABORATIVE FILTERING")
    print("="*50)
    
    # Test with a few user IDs
    test_users = [1, 5, 10]
    
    for user_id in test_users:
        print(f"\n🔍 Recommendations for User {user_id}:")
        recs = get_collaborative_recommendations(
            user_id, user_similarity, user_item_matrix, 
            user_id_to_index, movies, top_n=5
        )
        if isinstance(recs, str):
            print(recs)
        else:
            print(recs[['Title', 'Year', 'Genres']].to_string(index=False))
        print("-" * 40)