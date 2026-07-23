# preprocess_data.py
import pandas as pd
import numpy as np
import re

def load_data():
    """Load movies and ratings datasets"""
    movies = pd.read_csv('data/movies.csv')
    ratings = pd.read_csv('data/ratings.csv')
    return movies, ratings

def clean_movie_titles(movies):
    """Extract year from title and clean it"""
    # Extract year using regex: "Toy Story (1995)" -> "Toy Story"
    movies['year'] = movies['title'].str.extract(r'\((\d{4})\)', expand=False)
    movies['clean_title'] = movies['title'].str.replace(r'\s*\(\d{4}\)', '', regex=True)
    return movies

def split_genres(movies):
    """Split genres into separate columns"""
    # Get all unique genres
    all_genres = set()
    for genres in movies['genres']:
        if genres != '(no genres listed)':
            all_genres.update(genres.split('|'))
    
    all_genres = sorted(list(all_genres))
    print(f"🎭 Found {len(all_genres)} unique genres: {all_genres}")
    
    # Create binary columns for each genre
    for genre in all_genres:
        movies[genre] = movies['genres'].apply(
            lambda x: 1 if genre in x else 0
        )
    
    return movies, all_genres

def create_combined_features(movies):
    """Combine title, genres, and year into a single feature string"""
    # For content-based filtering, we need a "soup" of features
    movies['soup'] = (
        movies['clean_title'] + ' ' +
        movies['genres'].str.replace('|', ' ') + ' ' +
        movies['year'].fillna('')
    )
    return movies

def get_movie_stats(movies, ratings):
    """Calculate useful statistics"""
    # Count ratings per movie
    rating_counts = ratings.groupby('movieId').size().reset_index(name='rating_count')
    
    # Merge with movies
    movies_with_stats = movies.merge(rating_counts, on='movieId', how='left')
    movies_with_stats['rating_count'] = movies_with_stats['rating_count'].fillna(0)
    
    # Calculate average rating per movie
    avg_ratings = ratings.groupby('movieId')['rating'].mean().reset_index(name='avg_rating')
    movies_with_stats = movies_with_stats.merge(avg_ratings, on='movieId', how='left')
    
    return movies_with_stats

def main():
    print("🚀 Starting data preprocessing...")
    
    # Load data
    movies, ratings = load_data()
    print(f"✅ Loaded {len(movies)} movies and {len(ratings)} ratings")
    
    # Clean titles
    movies = clean_movie_titles(movies)
    print("✅ Cleaned movie titles and extracted years")
    
    # Split genres
    movies, all_genres = split_genres(movies)
    print("✅ Split genres into binary columns")
    
    # Create combined features
    movies = create_combined_features(movies)
    print("✅ Created combined feature strings")
    
    # Get statistics
    movies = get_movie_stats(movies, ratings)
    print("✅ Calculated movie statistics")
    
    # Save processed data
    movies.to_csv('data/movies_processed.csv', index=False)
    print("\n💾 Saved processed data to data/movies_processed.csv")
    
    # Show sample
    print("\n📊 Sample of processed data:")
    print(movies[['movieId', 'clean_title', 'year', 'avg_rating', 'rating_count']].head(10))
    
    # Filter movies with at least 10 ratings for better recommendations
    popular_movies = movies[movies['rating_count'] >= 10]
    print(f"\n🎯 Movies with 10+ ratings: {len(popular_movies)}")
    
    popular_movies.to_csv('data/popular_movies.csv', index=False)
    print("💾 Saved popular movies to data/popular_movies.csv")

if __name__ == "__main__":
    main()