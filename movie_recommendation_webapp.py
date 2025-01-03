# -*- coding: utf-8 -*-
"""
Created on Fri Jan  3 22:29:23 2025

@author: deepi
"""
import streamlit as st
import numpy as np
import pandas as pd
import difflib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load dataset
@st.cache_data  # Updated caching function for data-related operations
def load_data():
    return pd.read_csv("C:/Users/deepi/Downloads/ML PROJECTS STUFFS/movies.csv")

movies_df = load_data()

# Preprocess data
selected_features = ['genres', 'keywords', 'tagline', 'cast', 'director']
for feature in selected_features:
    movies_df[feature] = movies_df[feature].fillna('')

# Combine features
combined_features = (
    movies_df['genres'] + ' ' +
    movies_df['keywords'] + ' ' +
    movies_df['tagline'] + ' ' +
    movies_df['cast'] + ' ' +
    movies_df['director']
)

# Convert text data into feature vectors
vectorizer = TfidfVectorizer()
feature_vector = vectorizer.fit_transform(combined_features)

# Calculate similarity scores
similarity = cosine_similarity(feature_vector)

# Streamlit UI
st.title("🎥 Movie Recommendation System")

# Input from user
movie_name = st.text_input("Enter your favorite movie name:")

if movie_name:
    # Get a list of all movie titles
    list_of_all_titles = movies_df['title'].tolist()

    # Find the best match for the movie name
    find_close_match = difflib.get_close_matches(movie_name, list_of_all_titles)

    if find_close_match:
        close_match = find_close_match[0]
        st.write(f"Best match: **{close_match}**")

        # Get index of the movie
        index_of_the_movie = movies_df[movies_df.title == close_match]['index'].values[0]

        # Get similarity scores
        similarity_score = list(enumerate(similarity[index_of_the_movie]))

        # Sort movies based on similarity score
        sorted_similar_movies = sorted(similarity_score, key=lambda x: x[1], reverse=True)

        # Display the top 20 recommended movies
        st.subheader("Movies you might like:")
        i = 1
        for movie in sorted_similar_movies:
            index = movie[0]
            title_from_index = movies_df[movies_df.index == index]['title'].values[0]
            if i < 20:
                st.write(f"{i}. {title_from_index}")
                i += 1
    else:
        st.error("Sorry, no close match found!")


