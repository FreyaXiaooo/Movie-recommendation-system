from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
import pandas as pd
import json 
import re
import numpy as np 

"""
#calculate lsa values

df = pd.read_csv('data/final_dataset.csv')
body = df['text']

vectorizer = TfidfVectorizer(min_df=1,stop_words='english')
bag_of_words = vectorizer.fit_transform(body)

svd = TruncatedSVD(n_components=2000)
lsa = svd.fit_transform(bag_of_words)

# Create a DataFrame to store LSA values with corresponding IDs
lsa_df = pd.DataFrame(subtitle_ids, columns=['Subtitle_ID'])
lsa_df = pd.concat([lsa_df, pd.DataFrame(lsa, columns=[f'LSA_{i+1}' for i in range(2000)])], axis=1)

# Save the DataFrame to a CSV file
lsa_df.to_csv('data/lsa_values.csv', index=False)
"""

# Read movie data and subtitle data
merged_data = pd.read_csv("data/final_dataset.csv")

# Extract textual features of movie names, subtitles, cast, crew, and keyword types
vectorizer = TfidfVectorizer(stop_words='english', use_idf=True, smooth_idf=True,  max_df=0.6, min_df=0.4, max_features=1000, norm='l2', sublinear_tf=True)
features = vectorizer.fit_transform(merged_data['title'] + ' ' + merged_data['cast'] + ' ' + merged_data['crew'] + ' ' + merged_data['keywords'] + ' ' + merged_data['text'])

# Calculate similarity matrix
similarity_matrix = cosine_similarity(features)

def recommend_movies(movie_title, top_n=10):
    # Find the index of the input movie
    movie_index = merged_data[merged_data['title'] == movie_title].index[0]
    
    # Get the indices of the most similar movies to the input movie
    similar_movies_indices = similarity_matrix[movie_index].argsort()[::-1][1:top_n+1]
    
    # Get the information of the recommended movies
    recommended_movies = merged_data.iloc[similar_movies_indices]['title']
    
    return recommended_movies

# Test the recommendation function
movie_title = input("Please enter the movie title:")
recommended_movies = recommend_movies(movie_title)

print(f"Based on the movie《{movie_title}》,the following movies are recommended based on movie name, subtitles, cast, crew, and keywords:")
print(recommended_movies)