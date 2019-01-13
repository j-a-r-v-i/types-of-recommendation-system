# -*- coding: utf-8 -*-
"""
Created on Sat Dec  1 11:20:21 2018

@author: archit bansal
"""



import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
ratings=pd.read_csv('ratings.csv',sep=",")
movies=pd.read_csv('movies.csv',sep=",",encoding='latin-1')
merged=ratings.merge(movies,left_on='movieId',right_on='movieId',sort=True)
merged=merged[['userId','title','rating']]
movieRatings=merged.pivot_table(index=['title'],columns=['userId'],values='rating')

movieRatings.fillna(0,inplace=True)
item_similarity=cosine_similarity(movieRatings)
item_sim_df=pd.DataFrame(item_similarity,index=movieRatings.index,columns=movieRatings.index)



def sim_movies_to(title):
    count=1
    print("similar movies to {} are:".format(title))
    for item in item_sim_df.sort_values(by=title,ascending=False).index[1:11]:
        print("NO.{} : {}".format(count,item))
        count+=1
           
sim_movies_to("Terminator 2: Judgment Day (1991)")            
    
