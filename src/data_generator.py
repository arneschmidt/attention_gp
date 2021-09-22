import tensorflow as tf
import numpy as np


class DataGenerator(tf.keras.utils.Sequence):
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
        # bag = np.array(images_temp, dtype=np.float32) / 255
        bag = np.array(images_temp, dtype=np.float32)

        # if label_temp == 1:
        #     bag_label = np.ones((1, 1))
        # else:
        #     bag_label = np.zeros((1, 1))

        # bag_label = np.ones((1, 1))*label_temp
        bag_label= label_temp #label_temp)
        # bag_label = np.expand_dims(bag_label, axis=0)

        return bag, bag_label