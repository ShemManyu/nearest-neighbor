# -*- coding: utf-8 -*-
"""
Created on Tue Aug  1 00:57:06 2017

@author: Shem`
"""

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
breast_cancer_dataset = load_breast_cancer()

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(breast_cancer_dataset['data'], breast_cancer_dataset['target'], random_state = 0)

from sklearn.neighbors import KNeighborsClassifier
accuracy, n = 0, 0
for i in range(2, 30, 2):
    knn = KNeighborsClassifier(n_neighbors = i)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    if np.mean(y_pred == y_test) > accuracy:
        accuracy = np.mean(y_pred == y_test)
        print(np.mean(y_pred == y_test))
        n = i
        print(n)


#Fits data with various values of n in a range and outputs value of n with the best accuracy 
    
    
