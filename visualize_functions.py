"""Utility functions."""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import cv2
from sklearn.preprocessing import StandardScaler

from tensorflow.keras import Model
from tensorflow.keras.layers import Conv3D, MaxPool3D, ConvLSTM2D, Flatten, Dense
from tensorflow.keras.optimizers import RMSprop


def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140])


def normalize_data(np_data):
    scaler = StandardScaler()
    scaled_images = np_data.reshape(-1, 30, 64, 64, 1)
    return scaled_images


class Conv3DModel(Model):
    def __init__(self, n_classes):
        super(Conv3DModel, self).__init__()

        # convolutions
        self.conv1 = Conv3D(
            32, (3, 3, 3),
            activation='relu',
            data_format='channels_last',
            name='conv1'
        )
        self.pool1 = MaxPool3D(
            pool_size=(2, 2, 2),
            data_format='channels_last'
        )
        self.conv2 = Conv3D(
            64, (3, 3, 3),
            activation='relu',
            data_format='channels_last',
            name='conv1'
        )
        self.pool2 = MaxPool3D(
            pool_size=(2, 2, 2),
            data_format='channels_last'
        )
        self.convLSTM = ConvLSTM2D(40, (3, 3))

        # flatten
        self.flatten = Flatten(name='flatten')

        # dense layers
        self.d1 = Dense(
            128, activation='relu',
            name='d1'
        )
        self.out = Dense(
            n_classes,
            activation='softmax',
            name='output'
        )

    def call(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.convLSTM(x)
        x = self.flatten(x)
        x = self.d1(x)
        return self.out(x)
