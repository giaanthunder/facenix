from __future__ import absolute_import, division, print_function

import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers, Model
import math, sys, os
from tensorflow.keras.applications.resnet import ResNet50, preprocess_input
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import logging
logger = tf.get_logger()
logger.setLevel(logging.ERROR)

# ==============================================================================
# =                               Classifier                                   =
# ==============================================================================
class Classifier(keras.Model):
   def __init__(self):
      super(Classifier, self).__init__()
      self.extractor = VGG16(include_top=False, weights='imagenet')
      self.extractor.trainable = False
      
      self.glb_avg = layers.GlobalAveragePooling2D()
      self.fc1 = dense_blk(1024, act='leaky_relu')
      self.dropout1 = layers.Dropout(0.25)
      self.fc2 = dense_blk(512, act='leaky_relu')
      self.dropout2 = layers.Dropout(0.25)
      self.fc3 = dense_blk(2, act='leaky_relu')


   def call(self, x, is_training=True):
      y = self.extractor(x)
      y = self.glb_avg(y)
      y = self.fc1(y)
      if is_training:
         y = self.dropout1(y)
      y = self.fc2(y)
      if is_training:
         y = self.dropout2(y)
      y = self.fc3(y)
      
      return y

# ==============================================================================
# =                               Custom layers                                =
# ==============================================================================
class dense_blk(layers.Layer):
   def __init__(self, units, act='linear'):
      super(dense_blk, self).__init__()
      self.fc = layers.Dense(units)
      self.bn = layers.BatchNormalization()
      if act == 'leaky_relu':
         self.act = layers.LeakyReLU()
      else:
         self.act = layers.Activation(act)
      
   def call(self, x):
      y = self.fc(x)
      y = self.bn(y)
      y = self.act(y)
      
      return y
      
# ==============================================================================
# =                                 Test model                                 =
# ==============================================================================
if __name__ == '__main__':
   net = Classifier()
   x = tf.random.uniform([16, 299, 299, 3])
   out = net(x)
   print(out.shape)
#
   
   