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
from model import *
from img_processing import *

generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)
generator = Generator(1e-4)
discriminator = Discriminator(1e-4)

def content_loss(y, pred):
    vgg = VGG((384,384,3))
    return vgg.loss(y, pred)


def adversarial_loss(pred):
  loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)
  y, logits = Discriminator(pred)
  return tf.reduce_mean(loss_object(logits, tf.ones_like(logits)))


def gen_loss_function(y, pred):
  return content_loss(y, pred) + 1e-3*(adversarial_loss(pred))


def discriminator_loss_function(y_real_pred, y_fake_pred, y_real_pred_logits, y_fake_pred_logits):
  loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)
  loss_real = tf.reduce_mean(loss_object(tf.ones_like(y_real_pred_logits), y_real_pred_logits))
  loss_fake = tf.reduce_mean(loss_object(tf.zeros_like(y_fake_pred_logits), y_fake_pred_logits))
  return loss_real + loss_fake

@tf.function
def train_step(x, y , epoch):
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        pred = generator(x)
        print(type(x))
        gen_loss = gen_loss_function(y, pred)
        y, y_logits = discriminator(y)
        pred, pred_logits = discriminator(pred)
        d_loss = discriminator_loss_function(y, pred, y_logits, pred_logits)
        gen_gradients = gen_tape.gradient(gen_loss, generator.trainable_variables)
        disc_gradients = disc_tape.gradient(d_loss, discriminator.trainable_variables)
        generator_optimizer.apply_gradients(zip(gen_gradients, generator.trainable_variables))
        discriminator_optimizer.apply_gradients(zip(disc_gradients, discriminator.trainable_variables))
    return  gen_loss, d_loss

image = crop_img(getimages("0002x4.png", "images/"))
train_step(image,image,20)


