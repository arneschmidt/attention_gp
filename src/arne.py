# -*- coding: utf-8 -*-
"""
Created on Tue Jul  6 16:04:53 2021

@author: 2922919
"""



# import tensorflow_addons as tfa
from keras.datasets import mnist
from keras.optimizers import Adam
from keras.callbacks import ReduceLROnPlateau
from keras.callbacks import EarlyStopping
import numpy as np
import my_functions


# VARIABLES
bag_size = 4
LRpatience = 5
ESpatience = 10
learning_rate = 0.00015

if __name__ == '__main__':

    # Load dataset
    (trainX, trainY), (testX, testY) = mnist.load_data()
    
    # Add channel dimension
    trainX = trainX[..., np.newaxis]
    testX = testX[..., np.newaxis]
        
    # Load model
    classifier_model = my_functions.MNIST_Model()
    my_optimist = Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

    # classifier_model.built = True
    # classifier_model.load_weights('MNIST_weights')

    classifier_model.compile(optimizer=my_optimist,
                                    loss=['binary_crossentropy'],
                                    metrics=['accuracy'],
                                    run_eagerly=True)
    
    # Generate bags and save them in dictionaries
    train_data = dict()
    train_labels = dict()
    # ....
        
    # Generate bags and save them in dictionaries
    val_data = dict()
    val_labels = dict()
    # ....
    
    # Generate bags and save them in dictionaries
    test_data = dict()
    test_labels = dict()
    # ....
            
    # Define generators
    train_generator = my_functions.MNIST_generator(images=train_data,
                                                   labels=train_labels,
                                                   shuffle=True)
    
    val_generator = my_functions.MNIST_generator(images=val_data,
                                                   labels=val_labels,
                                                   shuffle=False)
    
    test_generator = my_functions.MNIST_generator(images=test_data,
                                                   labels=test_labels,
                                                   shuffle=False)
        
        
        
    my_reduce_LR_callback = ReduceLROnPlateau(monitor='val_loss',
                                                      factor=0.1,
                                                      patience=LRpatience,
                                                      verbose=1,
                                                      mode='auto',
                                                      min_delta=0.0001,
                                                      cooldown=0,
                                                      min_lr=0)

    early_stop_val_loss_callback = EarlyStopping(monitor='val_loss',  # quantity to be monitored
                                                 min_delta=0.000001,  # minimum change in the monitored quantity to qualify as an improvement, i.e. an absolute change of less than min_delta, will count as no improvement.
                                                 patience=ESpatience,  # number of epochs with no improvement after which training will be stopped.
                                                 verbose=1,
                                                 mode='auto',
                                                 baseline=None,
                                                 restore_best_weights=True)
    
    callback_array = [my_reduce_LR_callback, early_stop_val_loss_callback]
            
    
    # Train model with generators (I do not use classifier_history myself, but you can check it out if you want)
    classifier_history = classifier_model.fit_generator(generator=train_generator,
                                                                        steps_per_epoch=None,
                                                                        epochs=20,
                                                                        verbose=1,  # Verbosity mode. 0 = silent, 1 = progress bar, 2 = one line per epoch.
                                                                        validation_data=val_generator,
                                                                        callbacks=callback_array,
                                                                        # use_multiprocessing=True,
                                                                        shuffle=True)
    
    # Save weights
    classifier_model.save_weights('MNIST_convMIL_weights')
    
    # Run a prediction
    predict = classifier_model.predict_generator(test_generator, verbose=1)