import tensorflow as tf
import imageio
import numpy as np
import matplotlib.pyplot as plt


def getimages(file_name, path):
    return imageio.imread(path + file_name)

def crop_img(x, is_random=True):
    w = x.shape[0]-384
    assert (w >= 0)
    h = x.shape[1]-384
    assert (h >= 0)
    w = np.random.randint(0,w)
    h = np.random.randint(0,h)
    x = x[w:w+384, h:h+384]
    x = x / (255. / 2.)
    x = x - 1.
    return x

def downsample(x):
    x = imresize(x, size=[96, 96], interp='bicubic', mode=None)
    x = x / (255. / 2.)
    x = x - 1.
    return x




