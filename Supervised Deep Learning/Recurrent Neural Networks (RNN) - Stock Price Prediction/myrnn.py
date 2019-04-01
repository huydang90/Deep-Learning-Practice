#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 19 23:47:01 2019

@author: dangngochuy
"""

#Part 1: Data preprocessing
import numpy as np  #only numpy arrays can be the input of neural networks in keras
import matplotlib.pyplot as plt #visualize the results
import pandas as pd  #to import the dataset and manage it easily 

#importing the training set 
dataset_train = pd.read_csv("Google_Stock_Price_Train.csv")
training_set = dataset_train.iloc[:, 1:2].values

#Feature scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range=(0, 1))
training_set_scaled = sc.fit_transform(training_set)

#create a data structure with 60 timesteps and 1 output 
# at time t, model look at 60 stock prices before time t (stock price in 60 days before time t )
# based on the trend captured in the 60 timesteps => try to predict the next output 
# each day look at the stock prices of 3 previous months (60 wokring days) to try and predict what happens tomorrow 
X_train = []
y_train = []
for i in range(60,1258):
    X_train.append(training_set_scaled[i-60:i,0])
    y_train.append(training_set_scaled[i,0])
X_train, y_train = np.array(X_train), np.array(y_train)

#Reshaping 
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

#Part 2: Building the RNN 

#Import the keras libraries and packages 
from keras.models import Sequential 
from keras.layers import Dense
from keras.layers import LSTM 
from keras.layers import Dropout

#Initialize RNN
regressor = Sequential() # to represent the sequence of layers

#Add first LSTM layer and dropout regularization to prevent overfitting 
regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 1)))
regressor.add(Dropout(rate = 0.2))

#Add second LSTM layer 
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(rate = 0.2))

#Add third LSTM layer 
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(rate = 0.2))

#Add fourth LSTM layer 
regressor.add(LSTM(units = 50, return_sequences = False))
regressor.add(Dropout(rate = 0.2))

#Add the output layer
regressor.add(Dense(units = 1))

#compile the RNN
regressor.compile(optimizer = "adam", loss = "mean_squared_error")

#fitting the RNN to the training set 
regressor.fit(X_train, y_train, epochs = 100, batch_size = 32) 

#Part 3: Making the prediction and visualizing the results 

#Getting the real stock price of Google 2017 
dataset_test = pd.read_csv("Google_Stock_Price_Test.csv")
real_stock_price = dataset_test.iloc[:,1:2].values

#Getting the predicted stock price 
dataset_total = pd.concat((dataset_train["Open"], dataset_test["Open"]), axis = 0)
inputs = dataset_total[len(dataset_total)-len(dataset_test)-60:].values
inputs = inputs.reshape(-1, 1)
inputs = sc.transform(inputs)
X_test = []
for i in range(60,80):
    X_test.append(inputs[i-60:i,0])
X_test = np.array(X_test)

#reshape
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
predicted_stock_price = regressor.predict(X_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)

#Visualizing the results 
plt.plot(real_stock_price, color = "red", label = "Real Google Stock Price")
plt.plot(predicted_stock_price, color = "blue", label = "Predicted Google Stock Price")
plt.title("Google Stock Price Prediction")
plt.xlabel("Time")
plt.ylabel("Stock Price")
plt.legend()
plt.show()