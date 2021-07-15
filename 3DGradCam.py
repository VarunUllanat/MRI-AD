from keras.layers.convolutional import MaxPooling3D, AveragePooling3D
from keras.layers import Flatten, add
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
import imageio
import nibabel

def compute_gradcam(mri, model, last_conv, activation, file_name, conv_size, dim_num = 1):
  last_conv_layer = model.get_layer(last_conv)
  last_conv_layer_model = keras.Model(model.inputs, last_conv_layer.output)
  classifier_input = keras.Input(shape=last_conv_layer.output.shape[1:])
  x = classifier_input
  x = model.get_layer(activation)(x)
  x = AveragePooling3D((2, 2,2), padding='same')(x)
  x = Flatten()(x)
  x = Dense(512, activation='relu')(x)
  x = Dropout(0.2)(x)
  x = Dense(256, activation='relu')(x)
  x = Dropout(0.2)(x)
  x = Dense(2, activation='softmax',kernel_initializer='he_normal')(x)
  classifier_model = keras.Model(classifier_input, x)
  with tf.GradientTape() as tape:
    last_conv_layer_output = last_conv_layer_model(mri)
    tape.watch(last_conv_layer_output)
    preds = classifier_model(last_conv_layer_output)
    top_pred_index = tf.argmax(preds[0])
    top_class_channel = preds[:, top_pred_index]
  grads = tape.gradient(top_class_channel, last_conv_layer_output)
  pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2,3))
  last_conv_layer_output = last_conv_layer_output.numpy()[0]
  pooled_grads = pooled_grads.numpy()
  for i in range(pooled_grads.shape[-1]):
    last_conv_layer_output[:, :, :, i] *= pooled_grads[i]
  heatmap = np.mean(last_conv_layer_output, axis=-1)
  print("Computed heatmap")
  resize_dim = 94/conv_size
  heatmap_3d_re = zoom(heatmap, (resize_dim, resize_dim, resize_dim))
  heatmap_3d_re = np.maximum(heatmap_3d_re, 0) / np.max(heatmap_3d_re)
  heatmap_3d_re = np.uint8(255 * heatmap_3d_re)
  mri_im = mri[0,:,:,:,0]
  x_train_3d = mri_im[:,:,:]
  heatmap_3d_re[x_train_3d < 100] = 0
  x_train_3d_sc = np.maximum(x_train_3d, 0) / np.max(x_train_3d)
  x_train_3d_sc = np.uint8(255 * x_train_3d_sc )
  jet3d = cm.get_cmap("jet")
  jet_colors3d = jet3d(np.arange(256))[:, :3]
  jet_heatmap3d = jet_colors3d[heatmap_3d_re]
  jet_img_3d = cm.get_cmap("jet")
  jet_colors_img_3d = jet_img_3d(np.arange(256))[:, :3]
  jet_img_3d = jet_colors_img_3d[x_train_3d_sc]
  superimposed_3d_img = jet_heatmap3d * 0.5 + jet_img_3d*0.5
  print("Superimposed images")
  if dim_num == 0:
    slices = [np.rot90(superimposed_3d_img[i,:,:,:]) for i in range(superimposed_3d_img.shape[0])]
    imageio.mimsave(file_name, slices)
  elif dim_num == 1:
    slices = [np.rot90(superimposed_3d_img[:,i,:,:]) for i in range(superimposed_3d_img.shape[0])]
    imageio.mimsave(file_name, slices)
  elif dim_num == 2:
    slices = [np.rot90(superimposed_3d_img[:,:,i,:]) for i in range(superimposed_3d_img.shape[0])]
    imageio.mimsave(file_name, slices)
  print("Done")
  return superimposed_3d_img