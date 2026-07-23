# app.py
import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Page configuration
st.set_page_config(
    page_title="Movie Recommender",
    page_icon="🎬",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #FF4B4B;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<p class="main-header">🎬 Movie Recommendation Engine</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Powered by Content-Based & Collaborative Filtering</p>', unsafe_allow_html=True)

# Load data and models (cached for performance)
@st.cache_resource
def load_data():
    movies = pd.read_csv('data/popular_movies.csv')
    return movies

@st.cache_resource
def load_content_based_model():
    cosine_sim = joblib.load('models/cosine_sim.pkl')
    indices = joblib.load('models/movie_indices.pkl')
    return cosine_sim, indices

@st.cache_resource
def load_collaborative_model():
    user_similarity = joblib.load('models/user_similarity.pkl')
    user_item_matrix = joblib.load('models/user_item_matrix.pkl')
    user_id_to_index = joblib.load('models/user_id_to_index.pkl')
    return user_similarity, user_item_matrix, user_id_to_index

# Load everything
movies = load_data()
cosine_sim, indices = load_content_based_model()
user_similarity, user_item_matrix, user_id_to_index = load_collaborative_model()

# Sidebar for navigation
st.sidebar.header("⚙️ Settings")
approach = st.sidebar.radio(
    "Choose Recommendation Approach:",
    ["🎯 Content-Based (Movie Similarity)", "👥 Collaborative (User-Based)"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("""
### 📊 About This Project
- **Dataset**: MovieLens (9,742 movies)
- **Popular Movies**: 5,879 (10+ ratings)
- **Techniques**: TF-IDF, Cosine Similarity
- **Built with**: Python, Scikit-Learn, Streamlit
""")

# ============ CONTENT-BASED RECOMMENDATIONS ============
if approach == "🎯 Content-Based (Movie Similarity)":
    st.header("🎯 Find Similar Movies")
    st.info("💡 Tell us a movie you love, and we'll recommend similar ones based on genre, title, and year!")
    
    # Movie search
    movie_list = movies['clean_title'].values
    selected_movie = st.selectbox(
        "🔍 Search for a movie:",
        options=movie_list,
        index=0
    )
    
    # Number of recommendations
    top_n = st.slider("Number of recommendations:", 5, 20, 10)
    
    if st.button("🎬 Get Recommendations", use_container_width=True):
        with st.spinner("Finding similar movies..."):
            # Get movie index
            if selected_movie not in indices.index:
                st.error(f"❌ Movie '{selected_movie}' not found in our database.")
            else:
                idx = indices[selected_movie]
                sim_scores = list(enumerate(cosine_sim[idx]))
                sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
                sim_scores = sim_scores[1:top_n + 1]
                movie_indices = [i[0] for i in sim_scores]
                
                recommended = movies.iloc[movie_indices][['clean_title', 'year', 'genres', 'avg_rating', 'rating_count']]
                recommended = recommended.rename(columns={
                    'clean_title': 'Title',
                    'year': 'Year',
                    'genres': 'Genres',
                    'avg_rating': 'Avg Rating',
                    'rating_count': 'Rating Count'
                })
                
                # Display the selected movie info
                st.subheader(f"📽️ You selected: **{selected_movie}**")
                selected_info = movies[movies['clean_title'] == selected_movie].iloc[0]
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Year", selected_info['year'])
                with col2:
                    st.metric("Avg Rating", f"{selected_info['avg_rating']:.2f} ⭐")
                with col3:
                    st.metric("Total Ratings", f"{int(selected_info['rating_count'])}")
                
                st.markdown(f"**Genres:** {selected_info['genres']}")
                
                # Display recommendations
                st.subheader(f"🎬 Top {top_n} Similar Movies")
                
                # Display as cards
                for i, row in recommended.iterrows():
                    with st.container():
                        col1, col2, col3, col4 = st.columns([3, 1, 1, 1])
                        with col1:
                            st.markdown(f"**{row['Title']}** ({row['Year']})")
                            st.caption(row['Genres'])
                        with col2:
                            st.markdown(f"⭐ {row['Avg Rating']:.2f}")
                        with col3:
                            st.markdown(f"👥 {int(row['Rating Count'])} ratings")
                        st.divider()

# ============ COLLABORATIVE RECOMMENDATIONS (WITH HYBRID FALLBACK) ============
else:
    st.header("👥 Personalized Recommendations (Hybrid)")
    st.info("💡 Enter a User ID (1 to 610). We'll use collaborative filtering first, then fall back to content-based if needed!")
    
    # User ID input
    user_id = st.number_input(
        "Enter User ID:",
        min_value=1,
        max_value=610,
        value=1,
        step=1
    )
    
    top_n = st.slider("Number of recommendations:", 5, 20, 10)
    
    if st.button("🎬 Get Recommendations", use_container_width=True):
        with st.spinner("Finding your perfect movies..."):
            if user_id not in user_id_to_index:
                st.error(f"❌ User {user_id} not found in our database.")
            else:
                user_idx = user_id_to_index[user_id]
                similarity_scores = list(enumerate(user_similarity[user_idx]))
                similar_users = sorted(similarity_scores, key=lambda x: x[1], reverse=True)[1:51]
                
                # Get unseen movies
                user_ratings = user_item_matrix.iloc[user_idx]
                unseen_movies = user_ratings[user_ratings == 0].index
                
                # Calculate predicted ratings (Collaborative Filtering)
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
                
                # Sort and get top candidates
                sorted_movies = sorted(movie_scores.items(), key=lambda x: x[1], reverse=True)
                top_movie_ids = [mid for mid, score in sorted_movies[:top_n * 2]]  # Get extra candidates
                
                # Filter to only valid movie IDs
                available_movie_ids = set(movies['movieId'].values)
                valid_movie_ids = [mid for mid in top_movie_ids if mid in available_movie_ids]
                
                # ============ HYBRID FALLBACK LOGIC ============
                fallback_used = False
                if len(valid_movie_ids) < top_n:
                    fallback_used = True
                    st.info(f"💡 Collaborative filtering found {len(valid_movie_ids)} movies. Adding content-based suggestions...")
                    
                    # Get user's top-rated movies (their favorites)
                    user_rated = user_ratings[user_ratings > 0].sort_values(ascending=False)
                    top_rated_movie_ids = user_rated.head(3).index.tolist()
                    
                    # For each favorite movie, find similar ones using content-based
                    additional_movie_ids = []
                    for fav_movie_id in top_rated_movie_ids:
                        fav_movie_info = movies[movies['movieId'] == fav_movie_id]
                        if not fav_movie_info.empty:
                            fav_title = fav_movie_info.iloc[0]['clean_title']
                            if fav_title in indices.index:
                                idx = indices[fav_title]
                                sim_scores = list(enumerate(cosine_sim[idx]))
                                sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
                                sim_scores = sim_scores[1:8]  # Get 7 similar movies per favorite
                                
                                for sim_idx, score in sim_scores:
                                    sim_movie_id = movies.iloc[sim_idx]['movieId']
                                    # Only add if not already in list and not already rated by user
                                    if (sim_movie_id not in valid_movie_ids and 
                                        sim_movie_id not in additional_movie_ids and
                                        sim_movie_id not in user_ratings[user_ratings > 0].index):
                                        additional_movie_ids.append(sim_movie_id)
                    
                    # Combine collaborative + content-based results
                    valid_movie_ids = valid_movie_ids + additional_movie_ids
                    valid_movie_ids = valid_movie_ids[:top_n]  # Limit to requested number
                
                # ============ DISPLAY RESULTS ============
                if not valid_movie_ids:
                    st.warning("⚠️ No recommendations available for this user.")
                else:
                    recommended = movies[movies['movieId'].isin(valid_movie_ids)][['movieId', 'clean_title', 'year', 'genres', 'avg_rating']]
                    recommended = recommended.rename(columns={
                        'clean_title': 'Title',
                        'year': 'Year',
                        'genres': 'Genres',
                        'avg_rating': 'Avg Rating'
                    })
                    
                    # Maintain order based on scores
                    recommended = recommended.set_index('movieId').loc[valid_movie_ids].reset_index()
                    
                    # Display user info
                    st.subheader(f"👤 Recommendations for User {user_id}")
                    user_rated_count = (user_ratings > 0).sum()
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Movies Rated by You", int(user_rated_count))
                    with col2:
                        avg_user_rating = user_ratings[user_ratings > 0].mean()
                        st.metric("Your Avg Rating", f"{avg_user_rating:.2f} ⭐")
                    with col3:
                        st.metric("Recommendations Found", len(valid_movie_ids))
                    
                    # Show which approach was used
                    if fallback_used:
                        st.success("✅ Hybrid approach used: Collaborative + Content-Based filtering combined!")
                        
                        # Show user's favorite movies (for context)
                        st.subheader("🎯 Based on your favorites:")
                        top_rated_display = movies[movies['movieId'].isin(user_ratings[user_ratings > 0].sort_values(ascending=False).head(3).index)]
                        for _, row in top_rated_display.iterrows():
                            st.caption(f"• {row['clean_title']} ({row['year']}) - {row['genres']}")
                    else:
                        st.success("✅ Pure collaborative filtering worked perfectly!")
                    
                    # Display recommendations
                    st.subheader(f"🎬 Top {len(valid_movie_ids)} Movies for You")
                    for i, row in recommended.iterrows():
                        with st.container():
                            col1, col2 = st.columns([4, 1])
                            with col1:
                                st.markdown(f"**{row['Title']}** ({row['Year']})")
                                st.caption(row['Genres'])
                            with col2:
                                st.markdown(f"⭐ {row['Avg Rating']:.2f}")
                            st.divider()
    

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>Built with ❤️ by Nishant</p>
    <p><a href='https://github.com/nishant-netizen524/Movie-Recommender' target='_blank'>View Source Code on GitHub</a></p>
</div>
""", unsafe_allow_html=True)