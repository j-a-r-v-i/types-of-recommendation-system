# -*- coding: utf-8 -*-
"""
Created on Sat Dec  1 15:51:06 2018

@author: archit bansal
"""

import pandas as pd
import numpy as np

movies=pd.read_csv('movies.csv',sep=",",encoding='latin-1')
movies=pd.concat([movies,movies['genres'].str.get_dummies(sep='|')],axis=1)
del movies['genres']
del movies['(no genres listed)']

selected=[1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
X=movies.iloc[:,2:21]

from sklearn.neighbors import NearestNeighbors

nbrs=NearestNeighbors(n_neighbors=5).fit(X)

print(nbrs.kneighbors([selected]))