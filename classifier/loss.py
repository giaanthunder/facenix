from __future__ import absolute_import, division, print_function

import tensorflow as tf
import numpy as np
from tensorflow.python.keras import layers
from tensorflow.python import keras
import math, sys, os


class losses():
   def __init__(self, g_batch_size = 1, dist_train=False):
      self.dist_train   = dist_train
      self.g_batch_size = g_batch_size
      
   def loss(self, model_lst, inputs):
      Cls = model_lst
      img, label = inputs
      
      pred = Cls(img)
      #loss = self.mean(tf.losses.binary_crossentropy(y_true=label, y_pred=pred))
      loss = self.mean(tf.losses.mse(y_true=label, y_pred=pred))
      #loss = self.mean(tf.losses.categorical_crossentropy(y_true=label, y_pred=pred))
      #print(pred)
      #print(label)
      #print(loss)
      #sys.exit()

      return loss

   # ==============================================================================
   # =                                  Utilities                                 =
   # ==============================================================================
   
   def mean(self, loss):
      return tf.reduce_sum(loss) * (1. / self.g_batch_size)

#








