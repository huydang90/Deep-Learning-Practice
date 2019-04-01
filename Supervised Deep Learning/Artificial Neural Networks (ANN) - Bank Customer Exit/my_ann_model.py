#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 13 10:07:22 2019

@author: dangngochuy
"""

#import libraries

import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 

#importing the dataset
dataset = pd.read_csv("Churn_Modelling.csv")
X = dataset.iloc[:, 3:-1].values
y = dataset.iloc[:, -1].values

#Encode categorical variable 
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X1 = LabelEncoder()
X[:, 1] = labelencoder_X1.fit_transform(X[:,1])
labelencoder_X2 = LabelEncoder()
X[:, 2] = labelencoder_X2.fit_transform(X[:,1])
onehotencoder_X = OneHotEncoder(categorical_features = [1])
X = onehotencoder_X.fit_transform(X).toarray()
X = X[:, 1:]

#Split data into training set and test set 
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

#Feature scaling 
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

#Make the ANN

#import the keras libraries and packages
import keras
from keras.models import Sequential 
from keras.layers import Dense

#Initialize the ANN 
classifier = Sequential()

#Add the input layer and the first hidden layer 
classifier.add(Dense(units = 6, kernel_initializer = "uniform", activation = "relu", input_dim = 11))

#Add second hidden 
classifier.add(Dense(units = 6, kernel_initializer = "uniform", activation = "relu"))

#Add the output layer 
classifier.add(Dense(units = 1, kernel_initializer = "uniform", activation = "sigmoid"))

#compiling the ANN
classifier.compile(optimizer = "adam", loss = "binary_crossentropy", metrics = ["accuracy"])

#Fit the ANN to the training set 
classifier.fit(x = X_train, y= y_train, batch_size = 10, epochs = 100)

#Part 3: Making the predictions and evaluate the ANN

#Predict with the test set 
y_pred = classifier.predict(X_test)
y_pred = (y_pred >0.5)

#Make confusion matrix 
from sklearn.metrics import confusion_matrix 
cm = confusion_matrix(y_test, y_pred)

#Make prediction on a new customer 
new_prediction = classifier.predict(sc_X.transform(np.array([[0,0, 600, 1, 40, 3, 60000, 2, 1, 1, 50000]])))
new_prediction = (new_prediction > 0.5)

#Part 4: Evaluate, Improve and Tune the ANN

#Evaluate the ANN
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
def build_classifier(): 
    classifier = Sequential()
    classifier.add(Dense(units = 6, kernel_initializer = "uniform", activation = "relu", input_dim = 11))
    classifier.add(Dense(units = 6, kernel_initializer = "uniform", activation = "relu"))
    classifier.add(Dense(units = 1, kernel_initializer = "uniform", activation = "sigmoid"))
    classifier.compile(optimizer = "adam", loss = "binary_crossentropy", metrics = ["accuracy"])
    return classifier
classifier = KerasClassifier(build_fn = build_classifier, batch_size = 10, epochs = 100)
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10, n_jobs = -1)
means = accuracies.mean()
variance = accuracies.std()

#Improve the ANN
#

#Tune the ANN
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
def build_classifier(optimizer): 
    classifier = Sequential()
    classifier.add(Dense(units = 6, kernel_initializer = "uniform", activation = "relu", input_dim = 11))
    classifier.add(Dense(units = 6, kernel_initializer = "uniform", activation = "relu"))
    classifier.add(Dense(units = 1, kernel_initializer = "uniform", activation = "sigmoid"))
    classifier.compile(optimizer = optimizer, loss = "binary_crossentropy", metrics = ["accuracy"])
    return classifier
classifier = KerasClassifier(build_fn = build_classifier)
parameters = {"batch_size": [25, 32],
              "epochs": [100, 500],
              "optimizer": ["adam", "rmsprop"]}
grid_search = GridSearchCV(estimator = classifier, 
                           param_grid = parameters, 
                           scoring = "accuracy", 
                           cv = 10)
grid_search = grid_search.fit(X_train, y_train)
best_parameters = grid_search.best_params_
best_accuracy = grid_search.best_score_


