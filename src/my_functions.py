# -*- coding: utf-8 -*-
"""
Created on Tue Jul  6 16:20:45 2021

@author: 2922919
"""

# import tensorflow.keras as keras
import tensorflow as tf
import keras
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from keras.layers import Conv2D
from tensorflow.keras.layers import Dropout
from keras.layers import MaxPooling2D
import numpy as np


##############################################################################
"""MNIST MODEL"""
##############################################################################
class MNIST_Model(tf.keras.Model):
    
    def __init__(self):
        super(MNIST_Model,self).__init__()

        self.input_layer = Conv2D(32, input_shape=(28, 28, 1), kernel_size=(3, 3), activation="relu")
        self.conv2 = Conv2D(32, kernel_size=(3, 3), activation="relu")
        self.pool1 = MaxPooling2D(pool_size=(2, 2))
        self.conv3 = Conv2D(64, kernel_size=(3, 3), activation="relu")
        self.conv4 = Conv2D(64, kernel_size=(3, 3), activation="relu")
        self.pool2 = MaxPooling2D(pool_size=(2, 2))
        self.flat = Flatten()
        
        self.att_in = Dense(16, activation='tanh')
        self.att_drop = Dropout(0.1)
        self.att_out = Dense(1, activation='sigmoid')
        
        self.bag_in = Dense(64, activation='relu')
        self.bag_drop = Dropout(0.1)
        self.bag_out = Dense(1, activation='sigmoid')
        
    def call(self,x):
        
        x = self.input_layer(x)
        x = self.conv2(x)
        x = self.pool1(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.pool2(x)
        emb = self.flat(x)
        
        y_att = self.att_in(emb)
        y_att = self.att_drop(y_att)
        y_att = self.att_out(y_att)
        
        A = tf.transpose(y_att)
        A = tf.nn.softmax(A)
        M = tf.matmul(A, emb)
        
        y_bag = self.bag_in(M)
        y_bag = self.bag_drop(y_bag)
        y_bag = self.bag_out(y_bag)
        
        return y_bag
 
 
##############################################################################
"""MNIST GENERATOR"""
##############################################################################
class MNIST_generator(keras.utils.Sequence):
    """Generates data for Keras."""

    def __init__(self, images, labels, shuffle):
        """Initialization.

        Args:
            images: A dictionary with a bag of image files per dictionary slot.
            labels: A dictionary of corresponding labels.
        """
        self.images = images
        self.labels = labels
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        """Denotes the number of batches per epoch."""
        return len(self.images)

    def __getitem__(self, index):
        """Generate one batch of data."""            
        
        # Generate bags and labels
        bag, bag_label = self.__data_generation(self.images[index], self.labels[index])

        return bag, bag_label

    def on_epoch_end(self):
        """Updates indexes after each epoch."""
        self.indexes = np.arange(len(self.images))
        if self.shuffle is True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, images_temp, label_temp):
        """Generates data containing samples."""
        
        # DO SOME GENERATION HERE
        
        # Random example
        bag = np.array(images_temp, dtype=np.float32)/255
        
        if label_temp == 1:
            bag_label = np.ones((1,1))
        else:
            bag_label = np.zeros((1,1))

        return bag, bag_label