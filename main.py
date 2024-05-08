from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import pickle
from sklearn import preprocessing


import numpy as np
import pandas as pd 
import streamlit as st



data = pd.read_csv("MLFinalData (1) (4).csv")
    

X = data.iloc[:, :-1].values
Y= data.iloc[:, 5].values

mm_scaler = preprocessing.MinMaxScaler()
X_mm = mm_scaler.fit_transform(X)


x_train,x_test,y_train,y_test=train_test_split(X_mm,Y)
knn_model=KNeighborsClassifier(n_neighbors=1) 
knn_model=knn_model.fit(x_train,y_train)

pickle.dump(knn_model,open('knn_model.pkl','wb'))
