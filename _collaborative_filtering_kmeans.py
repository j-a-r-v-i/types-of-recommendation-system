# -*- coding: utf-8 -*-
"""
Created on Sat Nov 17 19:11:53 2018

@author: archit bansal
"""

import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
ratings=pd.read_csv('ratings.csv',sep=",")
movies=pd.read_csv('movies.csv',sep=",",encoding='latin-1')
merged=ratings.merge(movies,left_on='movieId',right_on='movieId',sort=True)
merged=merged[['userId','title','rating']]
movieRatings=merged.pivot_table(index=['userId'],columns=['title'],values='rating')

movieRatings.fillna(0,inplace=True)

model_knn=NearestNeighbors(algorithm='brute',metric='cosine')

model_knn.fit(movieRatings.values)

user=2

distances,indices=model_knn.kneighbors((movieRatings.iloc[user-1, :]).values.reshape(1,-1),n_neighbors=7)

movieRatings=movieRatings.T
import operator

best=[]
for i in indices.flatten():
    if(i!=user-1):
        max_score=movieRatings.loc[:,i+1].max()
        best.append(movieRatings[movieRatings.loc[:,i+1]==max_score].index.tolist())
user_seen_movies=movieRatings[movieRatings.loc[:,user]>0].index.tolist()
for i in range(len(best)):
    for j in best[i]:
        if(j in user_seen_movies):
            best[i].remove(j)
                
most_common={}
for i in range(len(best)):
    for j in best[i]:
        if j in most_common:
            most_common[j]+=1
        else:
            most_common[j]=1
sorted_list=sorted(most_common.items(),key=operator.itemgetter(1),reverse=True)
print(sorted_list[:5])
    

                
            
    
