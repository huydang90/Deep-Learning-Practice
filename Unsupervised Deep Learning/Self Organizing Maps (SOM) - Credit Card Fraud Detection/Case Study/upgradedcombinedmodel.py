#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 23 10:17:09 2019

@author: dangngochuy
"""

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

#Finding the fraudster

# Get the customers that are likely to defraud the bank (potential cheaters)
# Add indices to SOM values & sort by value
helper = np.concatenate(
    (som.distance_map().reshape(rows*cols, 1),         # the SOM map values
     np.arange(rows*cols).reshape(rows*cols, 1)),      # concatenated with an index
    axis=1)                                            # as a 2D matrix with 2 columns of data
helper = helper[helper[:, 0].argsort()][::-1]          # sort by first column (map values) and reverse (so top values are first)
# First we choose how many cells to take as outliers...
use_threshold = True   # toggle usage for calculating indices (pick cells that exceed threshold or use hardcoded number of cells)
top_cells = 4          # 4 out of 100 seems a valid idea, but ideally it might be chosen after inspecting the current SOM plot
threshold = 0.8        # Use threshold to select top cells
# Take indices that correspond to cells we're interested in
idx = helper[helper[:, 0] > threshold, 1] if use_threshold else helper[:top_cells, 1]
# Find the data entries assigned to cells corresponding to the selected indices
result_map = []
mappings = som.win_map(X)
for i in range(rows):
    for j in range(cols):
        if (i*rows+j) in idx:
            if len(result_map) == 0:                
                result_map = mappings[(i,j)]
            else:
                # Sometimes a cell contains no observations (customers)... weird
                # This will cause numpy to raise an exception so guard against that!
                if len(mappings[(i,j)]) > 0:
                    result_map = np.concatenate((result_map, mappings[(i,j)]), axis=0)                
 
# finally we get our fraudster candidates
frauds = sc.inverse_transform(result_map)
 
# This is the list of potential cheaters (customer ids)
print(frauds[:, 0])


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
y_pred = np.concatenate((dataset.iloc[:,0:1],y_pred),axis = 1)
y_pred = y_pred[y_pred[:,1].argsort()]


