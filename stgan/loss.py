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
      Gen, Dis, Enc, Stu = model_lst
      x_real, a, b = inputs

      z_real     = Enc(x_real)
      z_fake_stu = Stu(z_real, b_diff)
      x_fake     = Gen(z_fake_stu, b_diff)
      z_real_stu = Stu(z_real, a_diff)
      x_rec      = Gen(z_real_stu, a_diff)

      d_real, a_rec = Dis(x_real)
      d_fake, b_rec = Dis(x_fake)

      fake_loss     = -self.mean(d_fake)
      fake_att_loss =  self.mean(tf.losses.binary_crossentropy(b, b_rec))
      rec_loss      =  self.mean(self.abs_error(x_real, x_rec))

      g_loss = fake_loss + fake_att_loss * 10.0 + rec_loss * 100.0
      return g_loss

   def d_loss(self, model_lst, inputs):
      Gen, Dis, Enc, Stu = model_lst
      x_real, a, b = inputs

      z_real     = Enc(x_real)
      z_fake_stu = Stu(z_real, b_diff)
      x_fake     = Gen(z_fake_stu, b_diff)

      d_real, a_rec = Dis(x_real)
      d_fake, b_rec = Dis(x_fake)

      real_loss     = -self.mean(d_real)
      fake_loss     =  self.mean(d_fake)
      gp            =  self.mean(self.grad_pen(Dis, x_real, x_fake))
      real_att_loss =  self.mean(tf.losses.binary_crossentropy(a, a_rec))
      d_loss = real_loss + fake_loss + gp + real_att_loss
      return d_loss
      
      
   # ==============================================================================
   # =                                  Utilities                                 =
   # ==============================================================================
   def abs_error(self, a, b):
      loss = tf.keras.losses.mean_absolute_error(a,b)
      loss = tf.reduce_mean(loss, axis=[1,2])
      return loss
      
   def grad_pen(self, f, real, fake):
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
   
   def mean(self, loss):
      if self.dist_train:
         return  tf.reduce_sum(loss) * (1. / self.g_batch_size)
      else:
         return tf.reduce_mean(loss)
#








