import tensorflow as tf
import numpy as np
from tensorflow.python.keras import layers
from tensorflow.python import keras
import math, sys, os


class losses():
   def __init__(self, g_batch_size = 1, dist_train=False):
      self.dist_train   = dist_train
      self.g_batch_size = g_batch_size

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











