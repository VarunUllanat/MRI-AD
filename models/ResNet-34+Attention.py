#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from keras.layers import BatchNormalization
from keras.layers import Dense, Flatten, Conv3D, MaxPooling3D, AveragePooling3D
from keras.layers import UpSampling3D
from keras.layers import Activation, Reshape
from keras.layers import Add
from keras.layers import Multiply
from keras.layers import Lambda
from keras.layers import Input
from keras.models import Model
from keras.layers import Dropout
from keras.regularizers import l2
from keras.layers.convolutional import MaxPooling3D, AveragePooling3D
from keras.layers import Flatten, add,  GlobalAveragePooling3D
from keras.models import Model
import matplotlib.cm as cm
import nibabel as nb
import numpy as np
import cv2
import matplotlib.pyplot as plt
import random
import tensorflow as tf
import scipy
from sklearn.model_selection import train_test_split
import keras
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv3D, MaxPooling3D
from keras.utils import to_categorical
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.normalization import BatchNormalization
from keras.layers import Dropout
from keras.optimizers import SGD
from keras.regularizers import l2
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import KFold
from keras.engine import Input
from keras.layers import (
    Activation,
    Conv3D,
    MaxPooling3D,
    UpSampling3D,
)
from keras.layers.merge import concatenate
from keras.layers.convolutional import Conv3D, ZeroPadding3D
from scipy.ndimage import zoom
import pickle
import sklearn
import random


# In[ ]:


def res_identity(x, filter, r = 0): 
    
    x_skip = x # this will be used for addition with the residual block 
    
    x = Conv3D(filter, kernel_size=(3, 3,3), strides=(1, 1,1), padding='same', kernel_regularizer=l2(0.001))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv3D(filter, kernel_size=(3, 3,3), strides=(1, 1,1), padding='same', kernel_regularizer=l2(0.001))(x)
    x = BatchNormalization()(x)
    if r == 1:
      x_skip = Conv3D(filter, kernel_size=(3, 3,3), strides=(1, 1,1), padding='same', kernel_regularizer=l2(0.001))(x_skip)
      x_skip = BatchNormalization()(x_skip)
    x = add([x, x_skip])
    x = Activation('relu')(x)

    return x


def attention_block(input, input_channels=None, output_channels=None, encoder_depth=1):
    p = 0
    t = 0
    r = 0

    if input_channels is None:
        input_channels = input.get_shape()[-1]
    if output_channels is None:
        output_channels = input_channels

    # First Residual Block
    for i in range(p):
        input = residual_block(input)

    # Trunc Branch
    output_trunk = input
    for i in range(t):
        output_trunk = residual_block(output_trunk)

    # Soft Mask Branch

    ## encoder
    ### first down sampling

    output_soft_mask = MaxPooling3D(padding='same')(input)  # 32x32
    for i in range(r):
        output_soft_mask = residual_block(output_soft_mask)

    skip_connections = []
    for i in range(encoder_depth - 1):

        ## skip connections
        output_skip_connection = residual_block(output_soft_mask)
        skip_connections.append(output_skip_connection)
        # print ('skip shape:', output_skip_connection.get_shape())

        ## down sampling
        output_soft_mask = MaxPooling3D(padding='same')(output_soft_mask)
        for _ in range(r):
            output_soft_mask = residual_block(output_soft_mask)

            ## decoder
    skip_connections = list(reversed(skip_connections))
    for i in range(encoder_depth - 1):
        ## upsampling
        for _ in range(r):
            output_soft_mask = residual_block(output_soft_mask)
        output_soft_mask = UpSampling3D()(output_soft_mask)
        ## skip connections
        output_soft_mask = Add()([output_soft_mask, skip_connections[i]])

    ### last upsampling
    for i in range(r):
        output_soft_mask = residual_block(output_soft_mask)
    output_soft_mask = UpSampling3D()(output_soft_mask)

    ## Output
    output_soft_mask = Conv3D(input_channels, (1, 1,1))(output_soft_mask)
    output_soft_mask = Conv3D(input_channels, (1, 1,1))(output_soft_mask)
    output_soft_mask = Activation('sigmoid')(output_soft_mask)

    # Attention: (1 + output_soft_mask) * output_trunk
    output = Lambda(lambda x: x + 1)(output_soft_mask)
    output = Multiply()([output, output_trunk])  #

    # Last Residual Block
    for i in range(p):
        output = residual_block(output)

    return output

def resnet_custom_2(input_shape,targets):
    input_im = Input(shape=input_shape) 
    x = Conv3D(32, kernel_size=(3, 3,3), strides=(1, 1,1), padding='same')(input_im)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = AveragePooling3D((3, 3,3), strides=(2,2 ,2), padding='same')(x)

    x = res_identity(x, filter=32)
    x = res_identity(x, filter=32)
    x = res_identity(x, filter=32)
    x = AveragePooling3D((3, 3,3), strides=(2,2 ,2), padding='same')(x)
    x = res_identity(x, filter=64, r = 1)
    x = res_identity(x, filter=64)
    x = res_identity(x, filter=64)
    x = res_identity(x, filter=64)
    x = attention_block(x)
    x = AveragePooling3D((3, 3,3), strides=(2,2 ,2), padding='same')(x)
    x = res_identity(x, filter=128, r = 1)
    x = res_identity(x, filter=128)
    x = res_identity(x, filter=128)
    x = res_identity(x, filter=128)
    x = res_identity(x, filter=128)
    x = res_identity(x, filter=128)
    x = AveragePooling3D((3, 3,3), strides=(2,2 ,2), padding='same')(x)
    x = res_identity(x, filter=512, r = 1)
    x = res_identity(x, filter=512)
    x = res_identity(x, filter=512)

    x = AveragePooling3D((2, 2,2), padding='same')(x)
    x = Flatten()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.2)(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.2)(x)
    x = Dense(targets, activation='softmax',kernel_initializer='he_normal')(x)

    model = Model(inputs=input_im, outputs=x, name='Resnet50')
    return model

