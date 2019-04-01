#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 17 10:46:46 2019

@author: dangngochuy
"""

# Part 1 - Building the CNN
from keras.models import Sequential # to initialize the neural networks 
from keras.layers import Conv2D #Step 1 of CNN to create the convolutional layers
from keras.layers import MaxPooling2D
from keras.layers import Flatten 
from keras.layers import Dense # use to connect the fully connected layer to the classic ann 

#Initialize the CNN
classifier = Sequential()

#Step 1- Convolution 
classifier.add(Conv2D(32,3,3, input_shape = (64, 64, 3), activation = 'relu'))

#Step 2 - Pooling
classifier.add(MaxPooling2D(pool_size = (2,2)))

#Adding a new layer of convolution
classifier.add(Conv2D(32,3,3, activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2,2)))


#Step 3 - Flatten 
classifier.add(Flatten())

#Step 4 - Full connection 
classifier.add(Dense(units = 128, activation = 'relu'))
classifier.add(Dense(units = 1, activation = 'sigmoid'))

#Compiling the CNN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

#Part 2: Fitting the CNN to the images 

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(
            'dataset/training_set',
            target_size=(64, 64),
            batch_size=32,
            class_mode='binary')

test_set = test_datagen.flow_from_directory(
        'dataset/test_set',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')

classifier.fit_generator(
        training_set,
        steps_per_epoch=8000,
        epochs=25,
        validation_data=test_set,
        validation_steps=2000)


#Part 3: Making new predictions 

import numpy as np #to preprocess our image so that the model can accept it 
from keras.preprocessing import image 
test_image = image.load_img('dataset/single_prediction/cat_or_dog_1.jpg', 
                            target_size = (64,64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = classifier.predict(test_image)
training_set.class_indices 
if result[0][0] == 1: 
    prediction = "dog"
else: 
    prediction = "cat" 

#the predict method expect to have 4 dimension => add another dimension with the expand_dims from numpy => this dimension correspond to the batch 
# the function of neural networks like the predict function cannot accept a single input by itself (like an image), it only accepts input in a batch 