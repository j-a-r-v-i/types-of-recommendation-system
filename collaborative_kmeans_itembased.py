# -*- coding: utf-8 -*-
"""
Created on Sat Dec  1 12:04:14 2018

@author: archit bansal
"""
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors

ratings=pd.read_csv('ratings.csv',sep=",")
movies=pd.read_csv('movies.csv',sep=",",encoding='latin-1')

merged=ratings.merge(movies,left_on='movieId',right_on='movieId',sort=True)

merged.rename(columns={'rating_user':'user_rating'},inplace=True)
merged=merged[['userId','title','rating']]

movieRatings=merged.pivot_table(index=['title'],columns=['userId'],values='rating')

movieRatings.fillna(0,inplace=True)

model_knn=NearestNeighbors(algorithm='brute',metric='cosine')

model_knn.fit(movieRatings.values)

distances,indices=model_knn.kneighbors((movieRatings.iloc[100, :]).values.reshape(1,-1),n_neighbors=7)

for i in range(0,len(distances.flatten())):
    if (i==0):
        print('Recomendations for {0}'.format(movieRatings.index[100]))
    else:
        print('{0} : {1}, with distance of {2}'.format(i,movieRatings.index[indices.flatten()[i]],distances.flatten()[i]))
