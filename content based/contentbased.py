# -*- coding: utf-8 -*-
"""
Created on Sat Dec  1 13:20:32 2018

@author: archit bansal
"""

import pandas as pd
import numpy as np


ratings=pd.read_csv('ratings.csv',sep=",")
movies=pd.read_csv('movies.csv',sep=",",encoding='latin-1')

ratings.loc[ratings['rating']<=3,"rating"]=0
ratings.loc[ratings['rating']>3,"rating"]=1

merged=ratings.merge(movies,left_on='movieId',right_on='movieId',sort=True)
merged=merged[['userId','title','genres','rating']]
merged=pd.concat([merged,merged['genres'].str.get_dummies(sep='|')],axis=1)
del merged['genres']
del merged['(no genres listed)']
rating=merged["rating"]
merged=merged.drop(['rating'],axis=1)
merged=merged.join(rating)


X=merged.iloc[:,2:21].values
y=merged.iloc[:,21].values

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Fitting classifier to the Training set
from sklearn.ensemble import RandomForestClassifier
classifier=RandomForestClassifier(n_estimators=500,criterion="entropy",random_state=0)
classifier.fit(X_train,y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

totalMovieIds=movies['movieId'].unique()

def nonratedmovies(userId):
    ratedmovies=ratings['movieId'].loc[ratings['userId']==userId]
    non_ratedmovies=np.setdiff1d(totalMovieIds,ratedmovies.values)
    non_ratedmoviesDF=pd.DataFrame(non_ratedmovies,columns=['movieId'])
    non_ratedmoviesDF['userId']=userId
    non_ratedmoviesDF['prediction']=0
    active_user_nonratedmovies=non_ratedmoviesDF.merge(movies,left_on='movieId',right_on='movieId',sort=True)
    active_user_nonratedmovies=pd.concat([active_user_nonratedmovies,active_user_nonratedmovies['genres'].str.get_dummies(sep='|')],axis=1)
    del active_user_nonratedmovies['genres']
    del active_user_nonratedmovies['(no genres listed)']
    del active_user_nonratedmovies['title']
    return(active_user_nonratedmovies)
    
active_user_nonratedmoviesDF=nonratedmovies(30)

df=active_user_nonratedmoviesDF.iloc[:,3:].values
y_pred2=classifier.predict(df)

active_user_nonratedmoviesDF["prediction"]=y_pred2
recommend=active_user_nonratedmoviesDF[['movieId','prediction']]

recommend=recommend.loc[recommend['prediction']==1]




