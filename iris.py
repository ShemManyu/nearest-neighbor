# -*- coding: utf-8 -*-
"""
Created on Mon Jul 31 23:10:03 2017

@author: Shem`
"""
import numpy as np
import pandas as pd
#import the classical iris dataset
from sklearn.datasets import load_iris
iris_dataset = load_iris()

#In order to know whether the model generalises well, we split the data into 75% train and 25% test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(iris_dataset['data'], iris_dataset['target'], random_state=0) #Random state parameter ensures a fixed seed thus the line will always have the same outcome

#create a dataframe from X_data
#label the cols using the strings in iris_dataset.feature_names
iris_dataframe = pd.DataFrame(X_train, columns=iris_dataset.feature_names)

#Looking at the data. create a scatter matrix from the dataframe,color by y_train.
#grr = pd.scatter_matrix(iris_dataframe, c=y_train, figsize=(8, 8), marker='o', hist_kwds={'bins': 20}, s=30, alpha=.8) #cmap=mglearn.cm3)

#build a model using k-nearest neigbor
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(X_train, y_train)

#test the model
y_pred = knn.predict(X_test)
print("Test score: {:.2f}".format(np.mean(y_pred == y_test)))