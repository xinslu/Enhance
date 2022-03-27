import tensorflow as tf
import tensorlayer as tl
from tensorlayer.prepro import *
import scipy
import numpy as np

def getimages(file_name, path):
    return scipy.misc.imread(path + file_name, mode='RGB')

def crop_img(x, is_random=True):
    x = crop(x, wrg=384, hrg=384, is_random=is_random)
    x = x / (255. / 2.)
    x = x - 1.
    return x

def downsample(x):
    x = imresize(x, size=[96, 96], interp='bicubic', mode=None)
    x = x / (255. / 2.)
    x = x - 1.
    return x


