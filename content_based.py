# content_based.py
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import joblib
import os

def build_similarity_matrix():
    """Builds and saves the TF-IDF and Cosine Similarity matrices"""
    print("🚀 Building Content-Based Recommendation Model...")
    
    # 1. Load the preprocessed data
    print("📥 Loading processed data...")
    movies = pd.read_csv('data/popular_movies.csv')
    
    # 2. Create the TF-IDF Vectorizer
    # stop_words='english' removes common words like 'the', 'and', 'a'
    print("🔢 Creating TF-IDF Matrix...")
    tfidf = TfidfVectorizer(stop_words='english')
    
    # Fit and transform the 'soup' column (title + genres + year)
    tfidf_matrix = tfidf.fit_transform(movies['soup'])
    print(f"   ✅ Matrix shape: {tfidf_matrix.shape} (Movies x Unique Words)")
    
    # 3. Compute Cosine Similarity
    print("📐 Computing Cosine Similarity...")
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    print(f"   ✅ Similarity matrix shape: {cosine_sim.shape}")
    
    # 4. Create a mapping from movie title to index
    # This helps us quickly find the row number of a movie when a user searches for it
    indices = pd.Series(movies.index, index=movies['clean_title']).drop_duplicates()
    
    # 5. Save the models to the 'models/' folder
    os.makedirs('models', exist_ok=True)
    joblib.dump(cosine_sim, 'models/cosine_sim.pkl')
    joblib.dump(indices, 'models/movie_indices.pkl')
    
    print("\n💾 Models saved successfully!")
    print("   - models/cosine_sim.pkl")
    print("   - models/movie_indices.pkl")
    
    return cosine_sim, indices, movies

def get_recommendations(title, cosine_sim, indices, movies, top_n=10):
    """Returns top N similar movies for a given title"""
    
    # Check if movie exists in our dataset
    if title not in indices.index:
        return f"❌ Movie '{title}' not found in the database."
    
    # 1. Get the index of the movie that matches the title
    idx = indices[title]
    
    # 2. Get the pairwise similarity scores of all movies with that movie
    # Enumerate adds an index to each score: (index, score)
    sim_scores = list(enumerate(cosine_sim[idx]))
    
    # 3. Sort the movies based on the similarity scores (descending)
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    
    # 4. Get the scores of the top N most similar movies 
    # We skip the first one (index 0) because it's the movie itself (similarity = 1.0)
    sim_scores = sim_scores[1:top_n + 1]
    
    # 5. Get the actual movie indices from our similarity scores
    movie_indices = [i[0] for i in sim_scores]
    
    # 6. Return the top N most similar movies
    recommended_movies = movies[['clean_title', 'year', 'genres']].iloc[movie_indices]
    recommended_movies.columns = ['Title', 'Year', 'Genres']
    
    return recommended_movies

if __name__ == "__main__":
    # Build the model
    cosine_sim, indices, movies = build_similarity_matrix()
    
    # Test it out!
    print("\n" + "="*50)
    print("🎬 TESTING THE RECOMMENDATION ENGINE")
    print("="*50)
    
    test_movies = ["Toy Story", "The Matrix", "Inception"]
    
    for movie in test_movies:
        print(f"\n🔍 If you liked '{movie}', you might also like:")
        recs = get_recommendations(movie, cosine_sim, indices, movies, top_n=5)
        if isinstance(recs, str):
            print(recs)
        else:
            print(recs.to_string(index=False))
        print("-" * 40)