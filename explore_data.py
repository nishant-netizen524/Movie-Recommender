# explore_data.py
import pandas as pd
import numpy as np

# Load the datasets
movies = pd.read_csv('data/movies.csv')
ratings = pd.read_csv('data/ratings.csv')

# Quick look at the data
print("🎬 Movies Dataset:")
print(movies.head())
print(f"\nShape: {movies.shape}")

print("\n⭐ Ratings Dataset:")
print(ratings.head())
print(f"\nShape: {ratings.shape}")

# Check for missing values
print("\n🔍 Missing Values in Movies:")
print(movies.isnull().sum())

print("\n🔍 Missing Values in Ratings:")
print(ratings.isnull().sum())

# Understand the genres
print("\n🎭 Sample Genres:")
print(movies['genres'].value_counts().head(10))

# Basic statistics
print("\n📊 Rating Statistics:")
print(ratings['rating'].describe())