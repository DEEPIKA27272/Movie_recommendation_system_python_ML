# -*- coding: utf-8 -*-
"""Movie_recommendation_system_ML_python.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1GoPbdZtp5e-5UELK5SMr6_yvehOkfCni
"""

#importing required libraries
import numpy as np
import pandas as pd
import difflib
from sklearn.feature_extraction.text import TfidfVectorizer # convert text to integer
from sklearn.metrics.pairwise import cosine_similarity

"""##Data preprocessing"""

#loading the dataset
movies_df=pd.read_csv("/content/movies.csv")

movies_df.head() #printing first 5 datas from the dataset

#checking the number of rows and columns
movies_df.shape

#selecting the relevant features for recommendation
selected_features=['genres','keywords','tagline','cast','director']
print(selected_features)

#replacing the null values with null string

for feature in selected_features:
  movies_df[feature]=movies_df[feature].fillna('')

#combining all the 5 selected features

combined_features=movies_df['genres']+' '+movies_df['keywords']+' '+movies_df['tagline']+' '+movies_df['cast']+' '+movies_df['director']

print(combined_features)

"""#converting text data into feature vector"""

vectorizer=TfidfVectorizer()

feature_vector=vectorizer.fit_transform(combined_features)

print(feature_vector)

"""#Performing cosine simalirity to get similarity score"""

similarity=cosine_similarity(feature_vector)

print(similarity)

print(similarity.shape) #In output first value represents the index value and the sceond value is the similarity score

#getting the movie name from the user
movie_name=input('enter the favourite movie name:')

#creating a list with all the movie names given in the dataset
list_of_all_titles=movies_df['title'].tolist()
print(list_of_all_titles)

#finding the best match given by user
find_close_match=difflib.get_close_matches(movie_name,list_of_all_titles)
print(find_close_match)

close_match=find_close_match[0]
print(close_match)

"""##Finding index of the movie"""

index_of_the_movie=movies_df[movies_df.title==close_match]['index'].values[0]
print(index_of_the_movie)

similarity_score=list(enumerate(similarity[index_of_the_movie]))
print(similarity_score)

len(similarity_score)

#checking the higher similarity score for that sortng the movies based on the similarity score
sorted_similar_movies=sorted(similarity_score ,key=lambda x:x[1] ,reverse=True)
print(sorted_similar_movies)

"""##The name of similar movies based on thier index"""

print('Movies suggested:\n')
i=1
for movie in sorted_similar_movies:
  index=movie[0]
  title_from_index=movies_df[movies_df.index==index]['title'].values[0]
  if(i<20):
    print(i,'.',title_from_index)
    i+=1

"""#movie recommendation system"""

movie_name=input('enter the favourite movie name:')

list_of_all_titles=movies_df['title'].tolist()

find_close_match=difflib.get_close_matches(movie_name,list_of_all_titles)

close_match=find_close_match[0]

index_of_the_movie=movies_df[movies_df.title==close_match]['index'].values[0]

similarity_score=list(enumerate(similarity[index_of_the_movie]))

sorted_similar_movies=sorted(similarity_score ,key=lambda x:x[1] ,reverse=True)

print('Movies suggested:\n')
i=1
for movie in sorted_similar_movies:
  index=movie[0]
  title_from_index=movies_df[movies_df.index==index]['title'].values[0]
  if(i<20):
    print(i,'.',title_from_index)
    i+=1