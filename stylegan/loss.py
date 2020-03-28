import tensorflow as tf
import numpy as np
from tensorflow.python.keras import layers
from tensorflow.python import keras
import math, sys, os


class losses():
   def __init__(self, g_batch_size = 1, dist_train=False):
      self.dist_train   = dist_train
      self.g_batch_size = g_batch_size
      vgg16             = keras.applications.vgg16.VGG16(weights='imagenet', include_top=False)
      self.extractor    = keras.Model(inputs=vgg16.input, outputs=vgg16.get_layer('block3_pool').output)
      self.flat         = layers.Flatten()
      self.ar           = 0.3

   def g_loss(self, model_lst, inputs):
      Gen, Dis  = model_lst
      z, x_real = inputs

      x_fake = Gen(z)
      d_fake = Dis(x_fake)
      loss   = -self.mean(d_fake)
      return loss

   def d_loss(self, model_lst, inputs):
      Gen, Dis  = model_lst
      z, x_real = inputs

      x_fake = Gen(z)
      d_real = Dis(x_real)
      d_fake = Dis(x_fake)

      real_loss  = -self.mean(d_real)
      fake_loss  =  self.mean(d_fake)
      gp         =  self.mean(grad_pen(Dis, x_real, x_fake))
      drift_loss =  self.mean(tf.square(d_real)) * 1e-8

      loss = real_loss + fake_loss + drift_loss + gp
      return loss

   def e_loss(self, model_lst, inputs):
      Gen, Enc  = model_lst
      _, x_real = inputs

      w     = Enc(x_real)
      w     = lerf(Gen.w_avg, w, self.ar)
      x_rec = Gen.synthesis(w)

      # loss = tf.losses.mean_absolute_error(y_pred=x_rec, y_true=x_real)

      x_rec = (x_rec + 1.)/2.
      x_rec = tf.image.resize(x_rec, [224, 224], tf.image.ResizeMethod.NEAREST_NEIGHBOR)
      f_rec = self.extractor(x_rec)
      f_rec = self.flat(f_rec)

      x_real = (x_real + 1)/2
      x_real = tf.image.resize(x_real, [224, 224], tf.image.ResizeMethod.NEAREST_NEIGHBOR)
      f_real = self.extractor(x_real)
      f_real = self.flat(f_real)

      loss = tf.losses.mean_squared_error(y_pred=f_rec, y_true=f_real)

      loss = self.mean(loss)

      return loss

   
   def mean(self, loss):
      if self.dist_train:
         return  tf.reduce_sum(loss) * (1. / self.g_batch_size)
      else:
         return tf.reduce_mean(loss)

def grad_pen(f, real, fake):
   def _interpolate(a, b):
      shape = [tf.shape(a)[0]] + [1] * (a.shape.ndims - 1)
      alpha = tf.random.uniform(shape=shape, minval=0., maxval=1.)
      inter = a + alpha * (b - a)
      inter.set_shape(a.shape)
      return inter
      
   x = _interpolate(real, fake)
   with tf.GradientTape() as t:
      t.watch(x)
      pred = f(x)
   grad = t.gradient(pred, x)
   norm = tf.norm(tf.reshape(grad, [tf.shape(grad)[0], -1]), axis=1)
   gp = 10. * ((norm - 1.)**2)   
   
   return gp

def lerf(a, b, t):
   y = a + (b - a) * t
   return y









