# 🎬 Movie Recommendation Engine (Hybrid System)

A production-grade recommendation system that combines **Content-Based Filtering** and **Collaborative Filtering** with intelligent fallback logic. Built to solve the real-world "cold-start problem" faced by companies like Netflix and Amazon.


![Content-Based Demo](assets/demo_content-based.png)
*Content-Based Filtering: Finding similar movies to "The Matrix"*

![Collaborative Demo](assets/demo_collaborative-based.png)
*Collaborative Filtering: Personalized recommendations for User 1*

![Hybrid Fallback](assets/demo_hybrid-fallback.png)
*Hybrid System: Intelligent fallback when collaborative data is insufficient*

## 🌟 Key Features

### 🎯 Content-Based Filtering
- Uses **TF-IDF Vectorization** to convert movie features (title, genres, year) into numerical vectors
- Computes **Cosine Similarity** to find movies with similar characteristics
- Example: "If you liked The Matrix, here are similar sci-fi action movies"

### 👥 Collaborative Filtering (User-Based)
- Builds a **User-Item Matrix** from 100,000+ ratings
- Identifies users with similar taste patterns
- Predicts ratings for unseen movies using weighted similarity scores
- Example: "Users who rated like you also loved Inception"

### 🚀 Hybrid Fallback System (Production-Ready)
- **Intelligent fallback logic**: If collaborative filtering lacks sufficient data, the system automatically falls back to content-based recommendations
- **Cold-start problem solved**: New users or users with sparse ratings still get quality recommendations
- **Context-aware**: Shows users which approach was used and why

## 🛠️ Tech Stack

| Component | Technology |
| :--- | :--- |
| **Language** | Python 3.x |
| **ML Libraries** | Scikit-Learn (TF-IDF, Cosine Similarity) |
| **Data Processing** | Pandas, NumPy |
| **Web Framework** | Streamlit |
| **Model Serialization** | Joblib |
| **Dataset** | MovieLens Latest Small (9,742 movies, 100,836 ratings) |

## 📊 System Architecture

### 🔄 Recommendation Flow

```mermaid
graph TD
    A[User Input] --> B{Movie or User ID?}
    B -->|Movie| C[Content-Based Filtering]
    B -->|User ID| D[Collaborative Filtering]
    C --> E[TF-IDF + Cosine Similarity]
    D --> F[User Similarity Matrix]
    E --> G{Enough Results?}
    F --> G
    G -->|Yes| H[Display Recommendations]
    G -->|No| I[Hybrid Fallback]
    I --> J[Content-Based on User Favorites]
    J --> H


## 📦 Installation & Usage

### 1. Clone the repository
```bash
git clone https://github.com/nishant-netizen524/Movie-Recommender.git
cd Movie_Recommender
