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
import os
from IPython.display import Image
import matplotlib.pyplot as plt
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

def resnet_18(input_shape,targets):
    input_im = Input(shape=input_shape) 
    x = Conv3D(32, kernel_size=(3, 3,3), strides=(1, 1,1), padding='same')(input_im)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = AveragePooling3D((3, 3,3), strides=(2,2 ,2), padding='same')(x)
    x = res_identity(x, filter=32, r = 1)
    x = res_identity(x, filter=32)
    x = AveragePooling3D((3, 3,3), strides=(2,2 ,2), padding='same')(x)
    x = res_identity(x, filter=64, r = 1)
    x = res_identity(x, filter=64)
    x = AveragePooling3D((3, 3,3), strides=(2,2 ,2), padding='same')(x)
    x = res_identity(x, filter=128, r = 1)
    x = res_identity(x, filter=128)
    x = AveragePooling3D((3, 3,3), strides=(2,2 ,2), padding='same')(x)
    x = res_identity(x, filter=256, r = 1)
    x = res_identity(x, filter=256)
    x = AveragePooling3D((2, 2,2), padding='same')(x)
    x = Flatten()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.2)(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.2)(x)
    x = Dense(targets, activation='softmax',kernel_initializer='he_normal')(x)

    model = Model(inputs=input_im, outputs=x, name='Resnet50')
    return model

