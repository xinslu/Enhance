import tensorflow as tf
from tensorflow.keras import layers
import pandas as pd
import numpy as np
from pandas.io.json import json_normalize
import json
from tensorflow.keras.layers import Dense, Input, GlobalMaxPooling1D
from tensorflow.keras.layers import LSTM, Embedding
from tensorflow.keras.models import Model
import  matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from matplotlib import pyplot as plt
from tensorflow.keras.preprocessing import image
from matplotlib.patches import Rectangle
from tensorflow.keras.losses import binary_crossentropy


class Discriminator(tf.keras.Model):
    def __init__(self, rate):
        super().__init()
        self.rate = rate
        self.cnn_0 = tf.keras.layers.Conv2D(kernel_size=3, filters=64, strides=1, padding='same')
        self.cnn_block = [
            self.ConvolutionBlock(32,  3 ,2),
            self.ConvolutionBlock(64, 3 ,1),
            self.ConvolutionBlock(64, 3 ,2),
            self.ConvolutionBlock(128, 3 ,1),
            self.ConvolutionBlock(128, 3 ,2),
            self.ConvolutionBlock(256, 3 ,1),
            self.ConvolutionBlock(256, 3 ,2),
        ]
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(512)
        self.leaky1 = tf.keras.layers.LeakyReLU(alpha=0.2)
        self.dense2 = tf.keras.layers.Dense(1)
        self.leaky2 = tf.keras.layers.LeakyReLU(alpha=0.2)


    def ConvolutionBlock(self, filters, kernel_size, strides):
        result = tf.keras.Sequential()
        result.add(tf.keras.layers.Conv2D(filters=filters, kernel_size = kernel_size , strides=strides, padding='same', use_bias=False))
        result.add(tf.keras.layers.BatchNormalization())
        result.add(tf.keras.layers.LeakyReLU(alpha=0.2))
        return result

    def call(self, x):
        x = self.convo0(x)
        x = self.leaky1(x)
        for convo in self.convoblock:
            x = convo(x)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.leaky2(x)
        logits = self.dense2(x)
        x = tf.keras.activations.sigmoid(logits)
        return x, logits


class Generator(tf.keras.Model):
    def __init__(self, rate, blocks):
        super().__init__()
        self.rate = rate
        self.discriminator = discriminator
        self.blocks = blocks
        self.cnn_0 = tf.keras.layers.Conv2D(filters = 64, kernel_size=9 , strides = 1, padding='same')
        self.cnn_1 = tf.keras.layers.Conv2D(filters = 64, kernel_size=3 , strides = 1, padding='same')
        self.cnn_1 = tf.keras.layers.Conv2D(filters = 64, kernel_size=3 , strides = 1, padding='same')
        self.residualblocks = []
        for i in range(16):
            result = tf.keras.Sequential()
            result.add(tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel_size, strides = strides, padding = 'same', use_bias = False))
            result.add(tf.keras.layers.BatchNormalization())
            result.add(tf.keras.layers.PReLU(shared_axis=[1,2]))
            result.add(tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel_size, strides = strides, padding = 'same', use_bias = False))
            self.residualblocks.append(result)
        self.upsample1 = tf.keras.layers.Conv2DTranspose(256, 4, strides=2, padding='same', activation='linear')
        self.upsample2 = tf.keras.layers.Conv2DTranspose(256, 4, strides=2, padding='same', activation='linear')


    def call(self, x):
        x = self.convo0(x)
        x = tf.keras.layers.PReLU(shared_axes=[1,2])(x)
        skip = x
        for residuallayer in self.residualblocks:
            skipx = x
            x = residuallayer(x)
            x += skipx
        x = self.cnn_1(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x += skip
        x = self.upsample1(x)
        x = self.upsample2(x)
        x = self.convo2(x)
        return x

