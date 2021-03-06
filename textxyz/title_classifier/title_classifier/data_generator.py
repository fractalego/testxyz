import random

import numpy as np
import tensorflow.keras as keras

from title_classifier.utils import bin_data_into_buckets


class DataGenerator(keras.utils.Sequence):
    def __init__(self, data, batch_size, shuffle=True):
        self.buckets = bin_data_into_buckets(data, batch_size)
        if shuffle:
            self.buckets = sorted(self.buckets, key=lambda x: random.random())
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __len__(self):
        return int(np.floor(len(self.buckets)))

    def __getitem__(self, index):
        X, y = self.__data_generation(self.buckets[index])
        return X, y

    def on_epoch_end(self):
        if self.shuffle:
            self.buckets = sorted(self.buckets, key=lambda x: random.random())

    def __data_generation(self, bucket):
        training_bucket = []
        for item in bucket:
            text_vectors = item['text_vectors']
            attributes = item['attributes']
            coordinates = item['coordinates']
            y = item['is_title']
            training_bucket.append((text_vectors, attributes, coordinates, y))

        items = list(zip(*training_bucket))
        data = [np.array(item) for item in items[:-1]]
        labels = np.array(items[-1])
        return data, labels