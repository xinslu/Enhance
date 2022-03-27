# Enhance

This is my implementation of the SRGAN Paper (https://arxiv.org/pdf/1609.04802.pdf) using Tensorflow. This document will also detail my understanding to concepts useful for this project.

# ResNet

A resnet is characterized by two important things: Residual Block and Skip Connection, which sort of also go hand in hand. A residual block is a stack of layers set in such a way that the output of a layer is taken and added to another layer deeper in the block. The non-linearity is then applied after adding it together with the output of the corresponding layer in the main path. Each of these Skip Connection Layers along with the input make up a residual block and form the core of making deeper Neural Networks. 

# Why ResNets Work

The reason why resnets work are because with skip connections and residual blocks you have that it is much easier to learn indenity mapping. The advantage of this is with ReLu activation, if the deep neural network nudges it towards a wrong and negative or zero direction it will be mapped to the indentity of the place from where the skip connection is established. The disadvantage of plain deep neural networks is that for them it is hard to maintain even indentity mapping and thus it resnets help maintain stability with larger amount of layers.


