#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 21 22:49:44 2019

@author: dangngochuy
"""

#Part 1: Creat the SOM

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 21 21:42:48 2019

@author: dangngochuy
"""

#import libraries
import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 

#import the dataset 
dataset = pd.read_csv("Credit_Card_Applications.csv")
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:, -1].values

#Feature scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0,1))
X = sc.fit_transform(X)


rows = 10
cols = 10


#Train the SOM
#2 options, made it from scratch or use code from another 
from minisom import MiniSom
som = MiniSom(x = 10, y = 10, input_len = 15, sigma = 1.0, learning_rate = 0.5)
som.random_weights_init(X)
som.train_random(data = X, num_iteration = 100)

#Visualize the results
from pylab import bone, pcolor, colorbar, plot, show
bone()
pcolor(som.distance_map().T)
colorbar()
markers = ['o', 's']
colors = ['r', 'g']
for i, x in enumerate(X): 
    w = som.winner(x)
    plot(w[0] + 0.5,
         w[1] + 0.5,
         markers[y[i]],
         markeredgecolor = colors[y[i]], 
         markerfacecolor = "None", 
         markersize = 10, 
         markeredgewidth = 2)
show()

#finding the frauds
mappings = som.win_map(X)
frauds = mappings[(1,7)]
frauds = sc.inverse_transform(frauds)

#Part 2: create supervised learning model 

#create matrix of features 
customers = dataset.iloc[:, 1:].values


#create the dependent variable
is_fraud = np.zeros(len(dataset))
for i in range(len(dataset)):
    if dataset.iloc[i,0] in frauds:
        is_fraud[i] = 1

#Use the ANN model 
#Feature scaling 
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
customers = sc_X.fit_transform(customers)

#Make the ANN

#import the keras libraries and packages
import keras
from keras.models import Sequential 
from keras.layers import Dense

#Initialize the ANN 
classifier = Sequential()

#Add the input layer and the first hidden layer 
classifier.add(Dense(units = 2, kernel_initializer = "uniform", activation = "relu", input_dim = 15))

#Add the output layer 
classifier.add(Dense(units = 1, kernel_initializer = "uniform", activation = "sigmoid"))

#compiling the ANN
classifier.compile(optimizer = "adam", loss = "binary_crossentropy", metrics = ["accuracy"])

#Fit the ANN to the training set 
classifier.fit(x = customers, y= is_fraud, batch_size = 1, epochs = 2)

#Part 3: Making the predictions and evaluate the ANN

#Predict with the test set 
y_pred = classifier.predict(customers)
y_pred = np.concatenate((dataset.iloc[:,0:1],y_pred),axis = 1).values
y_pred = y_pred[y_pred[:,1].argsort()]
        

