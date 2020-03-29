import tensorflow as tf
from tensorflow.python import keras
from tensorflow.python.keras import layers

import numpy as np

import math, sys, os, time

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import logging
logger = tf.get_logger()
logger.setLevel(logging.ERROR)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ==============================================================================
# =                                   Encoder                                  =
# ==============================================================================
class EncoderNN(keras.Model):
   def __init__(self):
      super(EncoderNN, self).__init__()
      max_res      = 1024
      max_lod      = int(math.log2(max_res)) - 2
      self.lod     = max_lod
      self.alpha   = tf.Variable(0.0, name='E/alpha', trainable=False)
      nf           = [512, 512, 512, 512, 512, 256, 128, 64, 32, 16]
      gain         = tf.sqrt(2.)
      hi_res_lod   = 5

      self.fromrgb_layers = []
      for i in range(max_lod+1):
         lay = from_rgb(nf, i)
         self.fromrgb_layers.append(lay)

      self.conv_layers = [e_primary_block(nf, 0, gain=gain)]
      for i in range(1, hi_res_lod):
         lay = d_low_res_block(nf, i, gain=gain)
         self.conv_layers.append(lay)

      for i in range(hi_res_lod, max_lod+1):
         lay = d_hi_res_block(nf, i, gain=gain)
         self.conv_layers.append(lay)


   def call(self, img_in):
      img = to_nchw(img_in)
      x = self.fromrgb_layers[self.lod](img)

      for i in range(self.lod, 0, -1):
         x   = self.conv_layers[i](x)
         img = downscale2d(img)
         y = self.fromrgb_layers[i-1](img)
         x   = lerf(x, y, self.alpha)
      x = self.conv_layers[0](x)
      return x

class e_primary_block(layers.Layer):
   def __init__(self, nf, i, gain):
      super(e_primary_block, self).__init__()
      
      # channel_1    = nf[i+1]+1
      channel_1    = nf[i+1]
      channel_2    = nf[i]
      res          = 2**(i+2)
      self.res     = res
      name         = 'E/%dx%d/'%(res, res)

      self.conv    = conv2d(kernel=3, in_channel=channel_1, out_channel=channel_2, name=name+'Conv/', gain=gain)
      init_conv_b  = tf.zeros(shape=[channel_2])
      self.conv_b  = tf.Variable(init_conv_b, name=name+'Conv/bias', trainable=True)
      self.flat    = layers.Flatten()

      self.dense_0 = dense_block(res*res*channel_2, channel_2, act='none', name=name+'Dense_0/', gain=gain)
      self.dense_1 = dense_block(channel_2, 512, act='none', name=name+'Dense_1/', gain=1.)

      self.latent = []

      for i in range(18):
         w = dense_block(512, 512, act='none', name=name+'Latent_%02d/'%i, gain=1.)
         self.latent.append(w)
   
   def call(self, x):
      # y = minibatch_stddev(x)
      # conv
      y = self.conv(x)
      y = y + tf.reshape(self.conv_b, shape=[1, -1, 1, 1])
      y = tf.nn.leaky_relu(y)
      y = self.flat(y)

      # dense 0
      y = self.dense_0(y)
      y = tf.nn.leaky_relu(y)
      # dense 1
      y = self.dense_1(y)

      ws = []
      for i in range(18):
         w_i = self.latent[i](y)
         w_i = tf.expand_dims(w_i, axis=1)
         ws.append(w_i)
      w = tf.concat(ws, axis=1)


      return w





class Encoder(keras.Model):
   def __init__(self, Gen):
      super(Encoder, self).__init__()
      vgg16          = keras.applications.vgg16.VGG16(weights='imagenet', include_top=False)
      self.extractor = keras.Model(inputs=vgg16.input, outputs=vgg16.get_layer('block3_pool').output)
      self.flat      = layers.Flatten()
      self.Gen       = Gen
      self.ar        = 0.3
      
   def call(self, x_real, opt_ite=500):
      x_real = tf.cast(x_real, dtype=tf.float32)
      x_real = x_real/255.
      x_real = tf.image.resize(x_real, [224, 224], tf.image.ResizeMethod.NEAREST_NEIGHBOR)
      f_real = self.extractor(x_real)
      f_real = self.flat(f_real)

      lr = tf.Variable(1.)

      n_try  = 100
      z_fake = tf.constant(tf.random.normal(shape=[n_try, 512]))
      loss = []
      for i in range(n_try//10):
         w = self.Gen.mapping(z_fake[i*10:(i+1)*10], brc=True)
         l = self.loss_calc(w, f_real)
         loss.append(l)
      loss = tf.concat(loss, axis=0)
      loss_best = tf.argsort(loss)[:18]
      z_best = tf.gather(z_fake, loss_best)
      z_best = tf.reverse(z_best, axis=[0])
      w_best = self.Gen.mapping(z_best, brc=False)
      w_best = tf.Variable(w_best)
      
      count = 0
      min_loss = 1000000.
      loop = tf.function(self.loop)
      tik = time.time()
      for i in range(opt_ite):
         loss, w = loop(w_best, f_real, lr)
         if i%500==0 and i!=0:
            lr.assign(lr/3.)

         if loss < min_loss:
            min_loss = loss
            min_w = w
            min_i = i
         print('loop: %d, loss: %f, min loss: %f from loop %d'%(i, loss.numpy(), min_loss.numpy(), min_i))
      
      tok = time.time()
      print('EXECUTING TIME: %2f'%(tok-tik))
      return min_w, min_loss
   
   def loss_calc(self, w, f_real):
      w      = lerf(self.Gen.w_avg, w, self.ar)
      x_fake = (self.Gen.synthesis(w) + 1.)/2.
      x_fake = tf.image.resize(x_fake, [224, 224], tf.image.ResizeMethod.NEAREST_NEIGHBOR)
      f_fake = self.extractor(x_fake)
      f_fake = self.flat(f_fake)

      loss = tf.losses.mean_squared_error(y_pred=f_fake, y_true=f_real)
      return loss


   def loop(self, w_fake, f_real, lr):
      n = 18 // w_fake.shape[0]
      with tf.GradientTape() as tape:
         w    = tf.tile(w_fake, multiples=[n,1])
         w    = tf.expand_dims(w, axis=0)
         loss = self.loss_calc(w, f_real)

      vars = [w_fake]
      grad = tape.gradient(loss, vars)
      w_fake.assign_sub(grad[0]*lr)

      w = lerf(self.Gen.w_avg, w, self.ar)

      return loss, w

# ==============================================================================
# =                                  Generator                                 =
# ==============================================================================
class G_style(keras.Model):
   def __init__(self, max_res=1024):
      super(G_style, self).__init__()
      z_size                 = 512
      w_size                 = 512
      self.num_lod           = int(math.log2(max_res)) - 1
      self.truncation_psi    = 0.7   
      self.truncation_cutoff = 8
      self.w_avg_beta        = 0.995 
      self.style_mixing_prob = 0.9   

      self.w_avg             = tf.Variable(tf.zeros(shape=[w_size]), name='Gs/w_avg', trainable=False)
      
      self.mapping           = G_mapping(z_size, w_size, 2*self.num_lod)
      self.synthesis         = G_synthesis(w_size, self.num_lod-1)
      
      self.is_training       = False
      
   def call(self, z1):
      # Evaluate mapping network
      w1 = self.mapping(z1)
      
      if self.is_training:
         # Update moving average of W
         batch_avg = tf.reduce_mean(w1[:,0], axis=0)
         self.w_avg.assign(lerf(batch_avg, self.w_avg, self.w_avg_beta))
         
         # Style mixing regularization
         batch_size = w1.shape[0]
         z2 = tf.random.normal([batch_size, self.z_size])
         w2 = self.mapping(z2)
         rand = tf.random.uniform([], 0., 1.)
         if rand < self.style_mixing_prob:
            rand = tf.random.uniform([], 1, self.mapping.mapping_layers)
            w = tf.stack( [ w1[:rand] , w2[rand:] ] )
         else:
            w = w1
      else:
         # Apply truncation trick, use in testing phase
         layer_id = tf.range(self.num_lod*2)
         psi = tf.where(layer_id<self.truncation_cutoff, self.truncation_psi, 1.)
         psi = tf.reshape(psi, shape=[1, -1, 1])
         w = lerf(self.w_avg, w1, psi)
      
      # Evaluate synthesis network
      img = self.synthesis(w)
      
      return img

class G_mapping(layers.Layer):
   def __init__(self, z_size, w_size, w_broadcast):
      super(G_mapping, self).__init__()
      self.z_size         = z_size
      self.w_size         = w_size
      self.mapping_layers = 8
      self.mapping_fmap   = 512
      self.mapping_lrmul  = 0.01
      self.gain           = tf.math.sqrt(2.)
      self.w_broadcast    = w_broadcast
      

      self.layer = []
      for i in range(self.mapping_layers):
         init_std = 1.0/self.mapping_lrmul
         
         if i < self.mapping_layers - 1:
            out_channel = self.mapping_fmap
         else:
            out_channel = self.w_size
         
         lay = dense_block(self.mapping_fmap, out_channel, act='lrelu', 
               name='G_mapping/Dense%d/'%i, lrmul=self.mapping_lrmul, gain=self.gain)
         self.layer.append(lay)


   def call(self, z, brc=True):
      # Normalize latents
      w = pixel_norm(z)
      
      # Mapping layers
      for i in range(self.mapping_layers):
         w = self.layer[i](w)
         
      # broadcast
      if brc:
         w = tf.expand_dims(w, axis=1)
         w = tf.tile(w, [1, self.w_broadcast, 1])
      
      return w

class G_synthesis(layers.Layer):
   def __init__(self, w_size, max_lod):
      super(G_synthesis, self).__init__()
      
      self.w_size  = w_size
      self.lod     = max_lod
      self.alpha   = tf.Variable(0., name='G_synthesis/alpha', trainable=False)
      gain         = tf.sqrt(2.)
      hi_res_lod   = 5
      nf           = [512, 512, 512, 512, 512, 256, 128, 64, 32, 16]
      
      # add block 4x4
      self.conv_layers = [g_primary_block(nf, 0, gain=gain)]
      
      # add block 8x8 -> 64x64
      for i in range(1, hi_res_lod):
         lay = g_low_res_block(nf, i, gain=gain)
         self.conv_layers.append(lay)
      
      # add block 128x128 -> 1024x1024
      for i in range(hi_res_lod, max_lod+1):
         lay = g_hi_res_block(nf, i, gain=gain)
         self.conv_layers.append(lay)
      
      # create list of to_rgb layers
      self.torgb_layers = []
      for i in range(0, max_lod+1):
         lay = to_rgb(nf, i)
         self.torgb_layers.append(lay)
      
   def call(self, w, rand_noise=False):
      x = self.conv_layers[0](w, rand_noise)
      img_out = self.torgb_layers[0](x)
      
      for i in range(1, self.lod+1):
         x = self.conv_layers[i](x, w, i, rand_noise)
         img = self.torgb_layers[i](x)
         img_out = upscale2d(img_out)
         img_out = lerf(img, img_out,self.alpha)
      
      img_out = to_nhwc(img_out)
      
      return img_out

# ==============================================================================
# =                                 Discrimator                                =
# ==============================================================================
class D_basic(keras.Model):
   def __init__(self, max_res=1024):
      super(D_basic, self).__init__()
      max_lod      = int(math.log2(max_res)) - 2
      self.lod     = max_lod
      self.alpha   = tf.Variable(0.0, name='D/alpha', trainable=False)
      nf           = [512, 512, 512, 512, 512, 256, 128, 64, 32, 16]
      gain         = tf.sqrt(2.)
      hi_res_lod   = 5

      self.fromrgb_layers = []
      for i in range(max_lod+1):
         lay = from_rgb(nf, i)
         self.fromrgb_layers.append(lay)

      self.conv_layers = [d_primary_block(nf, 0, gain=gain)]
      for i in range(1, hi_res_lod):
         lay = d_low_res_block(nf, i, gain=gain)
         self.conv_layers.append(lay)

      for i in range(hi_res_lod, max_lod+1):
         lay = d_hi_res_block(nf, i, gain=gain)
         self.conv_layers.append(lay)


   def call(self, img_in):
      img = to_nchw(img_in)
      x = self.fromrgb_layers[self.lod](img)

      for i in range(self.lod, 0, -1):
         x   = self.conv_layers[i](x)
         img = downscale2d(img)
         y = self.fromrgb_layers[i-1](img)
         x   = lerf(x, y, self.alpha)
      x = self.conv_layers[0](x)
      return x
      
# ==============================================================================
# =                               Custom layers                                =
# ==============================================================================
# resolution = 4x4
class g_primary_block(layers.Layer):
   def __init__(self, nf, i, gain):
      super(g_primary_block, self).__init__()
      
      channel_1        = nf[i]
      channel_2        = nf[i+1]
      res              = 2**(i+2)
      self.res         = res
      name             = 'G_synthesis/%dx%d/'%(res, res)   

      init_const       = tf.ones(shape=[1, channel_1, 4, 4])
      self.const       = tf.Variable(init_const, name=name+'Const/const', trainable=True)
      self.const_noise = noise_block(res=4, channel=channel_2, name=name+'Const/')
      init_const_b     = tf.zeros(shape=[channel_2])
      self.const_b     = tf.Variable(init_const_b, name=name+'Const/bias', trainable=True)
      self.const_style = style_mod(in_channel=512, out_channel=channel_2*2, name=name+'Const/')
      
      self.conv        = conv2d(kernel=3, in_channel=channel_2, out_channel=channel_2, name=name+'Conv/', gain=gain)
      self.conv_noise  = noise_block(res=4, channel=channel_2, name=name+'Conv/')
      init_conv_b      = tf.zeros(shape=[channel_2])
      self.conv_b      = tf.Variable(init_conv_b, name=name+'Conv/bias', trainable=True)
      self.conv_style  = style_mod(in_channel=512, out_channel=channel_2*2, name=name+'Conv/')
   
   def call(self, w, rand_noise=False):
      # const
      batch_size = w.shape[0]
      x = tf.tile(self.const, multiples=[batch_size, 1, 1, 1])
      x = self.const_noise(x, rand_noise)
      x = x + tf.reshape(self.const_b, shape=[1, -1, 1, 1])
      x = tf.nn.leaky_relu(x)
      x = instance_norm(x)
      x = self.const_style(x, w[:,0])
      
      # conv
      x = self.conv(x)
      x = self.conv_noise(x, rand_noise)
      x = x + tf.reshape(self.conv_b, shape=[1, -1, 1, 1])
      x = tf.nn.leaky_relu(x)
      x = instance_norm(x)
      x = self.conv_style(x, w[:,1])
      
      return x


# resolution 8x8 -> 64x64
class g_low_res_block(layers.Layer):
   def __init__(self, nf, i, gain):
      super(g_low_res_block, self).__init__()
      
      channel_1 = nf[i]
      channel_2 = nf[i+1]
      res=2**(i+2)
      self.res = res
      
      name             = 'G_synthesis/%dx%d/'%(res, res)
      
      self.conv0_up       = conv2d(kernel=3, in_channel=channel_1, out_channel=channel_2, name=name+'Conv0_up/', gain=gain)
      self.conv0_up_noise = noise_block(res=res, channel=channel_2, name=name+'Conv0_up/')
      init_conv0_up_b     = tf.zeros(shape=[channel_2])
      self.conv0_up_b     = tf.Variable(init_conv0_up_b, name=name+'Conv0_up/bias', trainable=True)
      self.conv0_up_style = style_mod(in_channel=512, out_channel=channel_2*2, name=name+'Conv0_up/')
      
      
      self.conv1          = conv2d(kernel=3, in_channel=channel_2, out_channel=channel_2, name=name+'Conv1/', gain=gain)
      self.conv1_noise    = noise_block(res=res, channel=channel_2, name=name+'Conv1/')
      init_conv1_b        = tf.zeros(shape=[channel_2])
      self.conv1_b        = tf.Variable(init_conv1_b, name=name+'Conv1/bias', trainable=True)
      self.conv1_style    = style_mod(in_channel=512, out_channel=channel_2*2, name=name+'Conv1/')
   
   def call(self, x, w, i, rand_noise=False):
      # conv0_up
      x = upscale2d(x)
      x = self.conv0_up(x)
      x = blur2d(x)
      x = self.conv0_up_noise(x, rand_noise)
      x = x + tf.reshape(self.conv0_up_b, shape=[1, -1, 1, 1])
      x = tf.nn.leaky_relu(x)
      x = instance_norm(x)
      x = self.conv0_up_style(x, w[:,i*2])
      
      # conv1
      x = self.conv1(x)
      x = self.conv1_noise(x, rand_noise)
      x = x + tf.reshape(self.conv1_b, shape=[1, -1, 1, 1])
      x = tf.nn.leaky_relu(x)
      x = instance_norm(x)
      x = self.conv1_style(x, w[:,i*2+1])
      
      return x
      
# resolution 128x128 -> 1024x1024
class g_hi_res_block(layers.Layer):
   def __init__(self, nf, i, gain):
      super(g_hi_res_block, self).__init__()
      
      channel_1 = nf[i]
      channel_2 = nf[i+1]
      res=2**(i+2)
      self.res = res
      
      name             = 'G_synthesis/%dx%d/'%(res, res)
      
      self.conv0_up       = conv2d_trans(kernel=3, in_channel=channel_1, out_channel=channel_2, name=name+'Conv0_up/', gain=gain, fused_scale=True)
      self.conv0_up_noise = noise_block(res=res, channel=channel_2, name=name+'Conv0_up/')
      init_conv0_up_b     = tf.zeros(shape=[channel_2])
      self.conv0_up_b     = tf.Variable(init_conv0_up_b, name=name+'Conv0_up/bias', trainable=True)
      self.conv0_up_style = style_mod(in_channel=512, out_channel=channel_2*2, name=name+'Conv0_up/')
      
      
      self.conv1          = conv2d(kernel=3, in_channel=channel_2, out_channel=channel_2, name=name+'Conv1/', gain=gain)
      self.conv1_noise    = noise_block(res=res, channel=channel_2, name=name+'Conv1/')
      init_conv1_b        = tf.zeros(shape=[channel_2])
      self.conv1_b        = tf.Variable(init_conv1_b, name=name+'Conv1/bias', trainable=True)
      self.conv1_style    = style_mod(in_channel=512, out_channel=channel_2*2, name=name+'Conv1/')
   
   def call(self, x, w, i, rand_noise=False):
      # conv0_up
      x = self.conv0_up(x)
      x = blur2d(x)
      x = self.conv0_up_noise(x, rand_noise)
      x = x + tf.reshape(self.conv0_up_b, shape=[1, -1, 1, 1])
      x = tf.nn.leaky_relu(x)
      x = instance_norm(x)
      x = self.conv0_up_style(x, w[:,i*2])
      
      # conv1
      x = self.conv1(x)
      x = self.conv1_noise(x, rand_noise)
      x = x + tf.reshape(self.conv1_b, shape=[1, -1, 1, 1])
      x = tf.nn.leaky_relu(x)
      x = instance_norm(x)
      x = self.conv1_style(x, w[:,i*2+1])
      
      return x
      
      
class to_rgb(layers.Layer):
   def __init__(self, nf, i):
      super(to_rgb, self).__init__()
      res       = 2**(i+2) # i = 8
      in_channel= nf[i+1]
      self.conv = conv2d(kernel=1, in_channel=in_channel, out_channel=3, 
            name='G_synthesis/%dx%d/ToRGB/'%(res,res), gain=1.)
      init_b    = tf.zeros(shape=[3])
      self.bias = tf.Variable(init_b, name='G_synthesis/%dx%d/ToRGB/bias'%(res,res), trainable=True)
      
   def call(self, x):
      b = tf.reshape(self.bias, shape=[1, -1, 1, 1])
      y = self.conv(x) + b
      return y
      
# resolution = 4x4
class d_primary_block(layers.Layer):
   def __init__(self, nf, i, gain):
      super(d_primary_block, self).__init__()
      
      channel_1    = nf[i+1]+1
      channel_2    = nf[i]
      res          = 2**(i+2)
      self.res     = res
      name         = 'D/%dx%d/'%(res, res)

      self.conv    = conv2d(kernel=3, in_channel=channel_1, out_channel=channel_2, name=name+'Conv/', gain=gain)
      init_conv_b  = tf.zeros(shape=[channel_2])
      self.conv_b  = tf.Variable(init_conv_b, name=name+'Conv/bias', trainable=True)
      self.flat    = layers.Flatten()

      self.dense_0 = dense_block(res*res*channel_2, channel_2, act='none', name=name+'Dense_0/', gain=gain)
      self.dense_1 = dense_block(channel_2, 1, act='none', name=name+'Dense_1/', gain=1.)
   
   def call(self, x):
      y = minibatch_stddev(x)
      # conv
      y = self.conv(y)
      y = y + tf.reshape(self.conv_b, shape=[1, -1, 1, 1])
      y = tf.nn.leaky_relu(y)
      y = self.flat(y)

      # dense 0
      y = self.dense_0(y)
      y = tf.nn.leaky_relu(y)
      # dense 1
      y = self.dense_1(y)
      return y


# resolution 8x8 -> 64x64
class d_low_res_block(layers.Layer):
   def __init__(self, nf, i, gain):
      super(d_low_res_block, self).__init__()
      
      channel_1 = nf[i+1]
      channel_2 = nf[i]
      res=2**(i+2)
      self.res = res
      
      name              = 'D/%dx%d/'%(res, res)
       
      self.conv0        = conv2d(kernel=3, in_channel=channel_1, out_channel=channel_1, name=name+'Conv0/')
      init_conv0_b      = tf.zeros(shape=[channel_1])
      self.conv0_b      = tf.Variable(init_conv0_b, name=name+'Conv0/bias', trainable=True)
      
      self.conv1_down   = conv2d(kernel=3, in_channel=channel_1, out_channel=channel_2, name=name+'Conv1_down/', gain=gain)
      init_conv1_down_b = tf.zeros(shape=[channel_2])
      self.conv1_down_b = tf.Variable(init_conv1_down_b, name=name+'Conv1_down/bias', trainable=True)
   
   def call(self, x):
      # conv0
      y = self.conv0(x)
      y = y + tf.reshape(self.conv0_b, shape=[1, -1, 1, 1])
      y = tf.nn.leaky_relu(y)
      
      # conv1_down
      y = blur2d(y)
      y = self.conv1_down(y)
      y = downscale2d(y)
      y = y + tf.reshape(self.conv1_down_b, shape=[1, -1, 1, 1])
      y = tf.nn.leaky_relu(y)
      
      return y


# resolution 128x128 -> 1024x1024
class d_hi_res_block(layers.Layer):
   def __init__(self, nf, i, gain):
      super(d_hi_res_block, self).__init__()
      
      channel_1 = nf[i+1]
      channel_2 = nf[i]
      res=2**(i+2)
      self.res = res
      
      name              = 'D/%dx%d/'%(res, res)
       
      self.conv0        = conv2d(kernel=3, in_channel=channel_1, out_channel=channel_1, name=name+'Conv0/')
      init_conv0_b      = tf.zeros(shape=[channel_1])
      self.conv0_b      = tf.Variable(init_conv0_b, name=name+'Conv0/bias', trainable=True)
      
      self.conv1_down   = conv2d(kernel=3, in_channel=channel_1, out_channel=channel_2, name=name+'Conv1_down/', fused_scale=True, gain=gain)
      init_conv1_down_b = tf.zeros(shape=[channel_2])
      self.conv1_down_b = tf.Variable(init_conv1_down_b, name=name+'Conv1_down/bias', trainable=True)
   
   def call(self, x):
      # conv0
      y = self.conv0(x)
      y = y + tf.reshape(self.conv0_b, shape=[1, -1, 1, 1])
      y = tf.nn.leaky_relu(y)
      
      # conv1_down
      y = blur2d(y)
      y = self.conv1_down(y)

      y = y + tf.reshape(self.conv1_down_b, shape=[1, -1, 1, 1])
      y = tf.nn.leaky_relu(y)
      
      return y


class from_rgb(layers.Layer):
   def __init__(self, nf, i):
      super(from_rgb, self).__init__()
      res       = 2**(i+2)
      out_channel= nf[i+1]
      self.conv = conv2d(kernel=1, in_channel=3, out_channel=out_channel, 
            name='D/%dx%d/FromRGB/'%(res,res), gain=tf.sqrt(2.))
      init_b    = tf.zeros(shape=[out_channel])
      self.bias = tf.Variable(init_b, name='D/%dx%d/FromRGB/bias'%(res,res), trainable=True)
      
   def call(self, x):
      b = tf.reshape(self.bias, shape=[1, -1, 1, 1])
      y = self.conv(x) + b
      y = tf.nn.leaky_relu(y)
      return y



class noise_block(layers.Layer):
   def __init__(self, res, channel, name):
      super(noise_block, self).__init__()
      self.noise_shape = [1, 1, res, res]
      init_noise  = tf.random.normal(shape=self.noise_shape)
      self.noise  = tf.Variable(init_noise, name=name+'noise', trainable=False)
      
      init_weight = tf.zeros(shape=[channel])
      self.weight = tf.Variable(init_weight, name=name+'noise_weight', trainable=True)
   
   def call(self, x, rand_noise=False):
      weight = tf.reshape(self.weight, shape=[1, -1, 1, 1])
      if rand_noise:
         noise = tf.random.normal(shape=self.noise_shape)
      else:
         noise = self.noise
      y = x + noise * weight
      return y
      
      
class style_mod(layers.Layer):
   def __init__(self, in_channel, out_channel, name):
      super(style_mod, self).__init__()
      self.dense = dense_block(in_channel, out_channel, act='none', name=name+'StyleMod/', gain=1.)
      
   def call(self, x, w):
      style  = self.dense(w)
      style  = tf.reshape(style, shape=[-1, 2, x.shape[1], 1, 1])
      style_scale = style[:, 0]
      style_bias  = style[:, 1]
      y = x * style_scale + style_bias + x
      return y
      
     
class conv2d_trans(layers.Layer):
   def __init__(self, kernel, in_channel, out_channel, name, fused_scale=False, lrmul=1., gain=tf.sqrt(2.)):
      super(conv2d_trans, self).__init__()
      self.fused_scale = fused_scale
      fan_in = tf.cast(kernel*kernel*in_channel, dtype=tf.float32)
      gain   = gain
      he_std = gain / tf.sqrt(fan_in)
      self.runtime_coef = he_std * lrmul
      
      init_std = 1.0/lrmul
      w_shape = [kernel, kernel, in_channel, out_channel]
      init_w  = tf.random.normal(shape=w_shape, mean=0., stddev=init_std)
      self.w  = tf.Variable(initial_value=init_w, name=name+'filter', trainable=True)
      
   def call(self, x):
      # equalized learning rate
      w = self.w * self.runtime_coef
      
      if self.fused_scale:
         bs, c, h, width = x.shape
         kernel, kernel, in_channel, out_channel = self.w.shape
         w = tf.transpose(w, [0,1,3,2]) # [kernel, kernel, out, in]
         w = tf.pad(w, [[1,1],[1,1],[0,0],[0,0]], mode='CONSTANT')
         w = tf.add_n([w[1:, 1:], w[:-1, 1:], w[1:, :-1], w[:-1, :-1]])
         os = [bs, out_channel, h*2, width*2]
      y = tf.nn.conv2d_transpose(x, w, os, strides=[1,1,2,2], padding='SAME', data_format='NCHW')
      return y

     
class conv2d(layers.Layer):
   def __init__(self, kernel, in_channel, out_channel, name, fused_scale=False, lrmul=1., gain=tf.sqrt(2.)):
      super(conv2d, self).__init__()
      self.fused_scale = fused_scale
      fan_in = tf.cast(kernel*kernel*in_channel, dtype=tf.float32)
      # gain   = gain
      he_std = gain / tf.sqrt(fan_in)
      self.runtime_coef = he_std * lrmul
      
      init_std = 1.0/lrmul
      w_shape = [kernel, kernel, in_channel, out_channel]
      init_w  = tf.random.normal(shape=w_shape, mean=0., stddev=init_std)
      # init_w  = tf.zeros(shape=w_shape)
      self.w  = tf.Variable(initial_value=init_w, name=name+'filter', trainable=True)
      
   def call(self, x):
      # equalized learning rate
      w = self.w * self.runtime_coef

      if self.fused_scale:
         w = tf.pad(w, [[1,1], [1,1], [0,0], [0,0]], mode='CONSTANT')
         w = tf.add_n([w[1:, 1:], w[:-1, 1:], w[1:, :-1], w[:-1, :-1]]) * 0.25
         strides=[1,1,2,2]
      else:
         strides=[1,1,1,1]

      y = tf.nn.conv2d(x, w, strides=strides, padding='SAME', data_format='NCHW')

      return y
      
      
class dense_block(layers.Layer):
   def __init__(self, in_channel, out_channel, act, name, lrmul=1., gain=tf.sqrt(2.)):
      super(dense_block, self).__init__()
      fan_in = tf.cast(in_channel, dtype=tf.float32)
      gain   = gain
      he_std = gain / tf.sqrt(fan_in)
      self.runtime_coef = he_std * lrmul
      self.lrmul  = lrmul
      
      init_std = 1.0/lrmul
      w_shape  = [in_channel, out_channel]
      init_w   = tf.random.normal(shape=w_shape, mean=0., stddev=init_std)
      self.w   = tf.Variable(initial_value=init_w, name=name+'weight', trainable=True)
               
      b_shape  = [out_channel]
      init_b   = tf.random.normal(shape=b_shape, mean=0., stddev=init_std)
      self.b   = tf.Variable(initial_value=init_b, name=name+'bias', trainable=True)
      
      self.act_name = act
      if act == 'none':
         self.act = layers.Activation('linear')
      if act == 'lrelu':
         self.act = layers.LeakyReLU(alpha=0.2)
      
   def call(self, x):
      # equalized learning rate
      w_scale = self.w * self.runtime_coef
      b_scale = self.b * self.lrmul
      y = tf.matmul(x, w_scale)
      y = y  + b_scale
      y = self.act(y)
      return y

# ==============================================================================
# =                                 Utilities                                  =
# ==============================================================================
def lerf(a, b, t):
   y = a + (b - a) * t
   return y
   
def minibatch_stddev(x):
   bs, c, h, w = x.shape
   gs = tf.minimum(4, bs)
   y = tf.reshape(x, [gs, -1, 1, c, h, w])
   y -= tf.reduce_mean(y, axis=0, keepdims=True)
   y = tf.reduce_mean(tf.square(y), axis=0)
   y = tf.sqrt(y + 1e-8)
   y = tf.reduce_mean(y, axis=[2,3,4], keepdims=True)
   y = tf.reduce_mean(y, axis=[2])
   y = tf.tile(y, multiples=[gs, 1, h, w])
   y = tf.concat([x, y], axis=1)
   return y

def pixel_norm(x, epsilon=1e-8):
   y = tf.square(x)
   y = tf.reduce_mean(y, axis=1, keepdims=True)
   y = tf.math.rsqrt(y + epsilon)
   y = x * y
   return y
   
def instance_norm(x, epsilon=1e-8):
   x_mean = tf.reduce_mean(x, axis=[2,3], keepdims=True)
   x = x - x_mean
   y = tf.square(x)
   # variance
   y = tf.reduce_mean(y, axis=[2,3], keepdims=True) + epsilon
   # y = (x-x_mean) / stddev
   y = x * tf.math.rsqrt(y)
   return y
   
def upscale2d(x, factor=2):
   _upscale_2d = layers.UpSampling2D(size=(factor, factor), data_format="channels_first", interpolation='nearest')
   y = _upscale_2d(x)
   return y

def downscale2d(x, factor=2, gain=1):
   f = [np.sqrt(gain) / factor] * factor
   y = blur2d(x, f, normalize=False, stride=2)
   return y

def blur2d(x, f=[1,2,1], normalize=True, stride=1):
   f = tf.convert_to_tensor(f, dtype=tf.float32)
   f1 = tf.expand_dims(f, axis=-1)
   f2 = tf.expand_dims(f, axis=0)
   f = f1 * f2
   
   # normalize
   if normalize:
      f = f/tf.reduce_sum(f)
   
   # reshape f to match x shape
   f = tf.reshape(f, shape=f.shape+[1, 1])
   channel = x.shape[1]
   f = tf.tile(f, multiples=[1,1,channel,1])

   # use depthwise_conv2d to blur each channel
   y = tf.nn.depthwise_conv2d(x, f, strides=[1,1,stride,stride], padding='SAME', data_format='NCHW')
   return y
   
def to_img(x):
   img = (x + 1.) * 127.5
   img = tf.clip_by_value(img, 0., 255.)
   #img = tf.cast(img, dtype=tf.uint8)
   return img

# NCHW to NHWC
def to_nhwc(x):
   x = tf.transpose(x, [0, 2, 3, 1])
   return x

# NHWC to NCHW
def to_nchw(x):
   x = tf.transpose(x, [0, 3, 1, 2])
   return x

def pretrained_models(experiment_name='origin'):
   Gen = G_style()
   checkpoint_dir = BASE_DIR+'/stylegan/output/%s/trained_model' % experiment_name
   checkpoint = tf.train.Checkpoint(
      Gen=Gen
   )
   manager = tf.train.CheckpointManager(checkpoint, checkpoint_dir, max_to_keep=3)

   if manager.latest_checkpoint:
      checkpoint.restore(manager.latest_checkpoint)
      return Gen
   else:
      print("RESTORE FAILED")
      return None
   
def create_pretrained_point():
   Gen = pretrained_models()
   Dis = D_basic()
   vars = Dis.trainable_variables
   for i in range(56):
      src = d_vars[i][1]
      dst = d_vars[i][0]
      for var in vars:
         if var.name == dst:
            tmp = np.load('./stylegan-encoder/d_vars/'+src)
            tmp = tf.convert_to_tensor(tmp)
            var.assign(tmp)
            break
   experiment_name = 'pretrained_stylegan'
   checkpoint_dir = './output/%s/trained_model' % experiment_name
   checkpoint = tf.train.Checkpoint(
      Gen=Gen, Dis=Dis
   )
   manager = tf.train.CheckpointManager(checkpoint, checkpoint_dir, max_to_keep=3)
   manager.save()
   
def pretrained_stylegan():
   Gen = pretrained_models()
   Dis = D_basic()
   vars = Dis.trainable_variables
   for i in range(56):
      src = d_vars[i][1]
      dst = d_vars[i][0]
      for var in vars:
         if var.name == dst:
            tmp = np.load('./d_vars/'+src)
            tmp = tf.convert_to_tensor(tmp)
            var.assign(tmp)
            break
   return Gen, Dis

   experiment_name = 'pretrained_stylegan'
   checkpoint_dir = './output/%s/trained_model' % experiment_name
   checkpoint = tf.train.Checkpoint(
      Gen=Gen, Dis=Dis
   )
   manager = tf.train.CheckpointManager(checkpoint, checkpoint_dir, max_to_keep=3)
   if manager.latest_checkpoint:
      checkpoint.restore(manager.latest_checkpoint)
      return Gen, Dis
   else:
      print("RESTORE FAILED")
      return None, None

d_vars = [
   ['D/1024x1024/FromRGB/filter:0'     , 'D_FromRGB_lod0_weight.npy'        ],
   ['D/1024x1024/FromRGB/bias:0'       , 'D_FromRGB_lod0_bias.npy'          ],
   ['D/1024x1024/Conv0/filter:0'       , 'D_1024x1024_Conv0_weight.npy'     ],
   ['D/1024x1024/Conv0/bias:0'         , 'D_1024x1024_Conv0_bias.npy'       ],
   ['D/1024x1024/Conv1_down/filter:0'  , 'D_1024x1024_Conv1_down_weight.npy'],
   ['D/1024x1024/Conv1_down/bias:0'    , 'D_1024x1024_Conv1_down_bias.npy'  ],

   ['D/512x512/FromRGB/filter:0'       , 'D_FromRGB_lod1_weight.npy'        ],
   ['D/512x512/FromRGB/bias:0'         , 'D_FromRGB_lod1_bias.npy'          ],
   ['D/512x512/Conv0/filter:0'         , 'D_512x512_Conv0_weight.npy'       ],
   ['D/512x512/Conv0/bias:0'           , 'D_512x512_Conv0_bias.npy'         ],
   ['D/512x512/Conv1_down/filter:0'    , 'D_512x512_Conv1_down_weight.npy'  ],
   ['D/512x512/Conv1_down/bias:0'      , 'D_512x512_Conv1_down_bias.npy'    ],

   ['D/256x256/FromRGB/filter:0'       , 'D_FromRGB_lod2_weight.npy'        ],
   ['D/256x256/FromRGB/bias:0'         , 'D_FromRGB_lod2_bias.npy'          ],
   ['D/256x256/Conv0/filter:0'         , 'D_256x256_Conv0_weight.npy'       ],
   ['D/256x256/Conv0/bias:0'           , 'D_256x256_Conv0_bias.npy'         ],
   ['D/256x256/Conv1_down/filter:0'    , 'D_256x256_Conv1_down_weight.npy'  ],
   ['D/256x256/Conv1_down/bias:0'      , 'D_256x256_Conv1_down_bias.npy'    ],

   ['D/128x128/FromRGB/filter:0'       , 'D_FromRGB_lod3_weight.npy'        ],
   ['D/128x128/FromRGB/bias:0'         , 'D_FromRGB_lod3_bias.npy'          ],
   ['D/128x128/Conv0/filter:0'         , 'D_128x128_Conv0_weight.npy'       ],
   ['D/128x128/Conv0/bias:0'           , 'D_128x128_Conv0_bias.npy'         ],
   ['D/128x128/Conv1_down/filter:0'    , 'D_128x128_Conv1_down_weight.npy'  ],
   ['D/128x128/Conv1_down/bias:0'      , 'D_128x128_Conv1_down_bias.npy'    ],

   ['D/64x64/FromRGB/filter:0'         , 'D_FromRGB_lod4_weight.npy'        ],
   ['D/64x64/FromRGB/bias:0'           , 'D_FromRGB_lod4_bias.npy'          ],
   ['D/64x64/Conv0/filter:0'           , 'D_64x64_Conv0_weight.npy'         ],
   ['D/64x64/Conv0/bias:0'             , 'D_64x64_Conv0_bias.npy'           ],
   ['D/64x64/Conv1_down/filter:0'      , 'D_64x64_Conv1_down_weight.npy'    ],
   ['D/64x64/Conv1_down/bias:0'        , 'D_64x64_Conv1_down_bias.npy'      ],

   ['D/32x32/FromRGB/filter:0'         , 'D_FromRGB_lod5_weight.npy'        ],
   ['D/32x32/FromRGB/bias:0'           , 'D_FromRGB_lod5_bias.npy'          ],
   ['D/32x32/Conv0/filter:0'           , 'D_32x32_Conv0_weight.npy'         ],
   ['D/32x32/Conv0/bias:0'             , 'D_32x32_Conv0_bias.npy'           ],
   ['D/32x32/Conv1_down/filter:0'      , 'D_32x32_Conv1_down_weight.npy'    ],
   ['D/32x32/Conv1_down/bias:0'        , 'D_32x32_Conv1_down_bias.npy'      ],

   ['D/16x16/FromRGB/filter:0'         , 'D_FromRGB_lod6_weight.npy'        ],
   ['D/16x16/FromRGB/bias:0'           , 'D_FromRGB_lod6_bias.npy'          ],
   ['D/16x16/Conv0/filter:0'           , 'D_16x16_Conv0_weight.npy'         ],
   ['D/16x16/Conv0/bias:0'             , 'D_16x16_Conv0_bias.npy'           ],
   ['D/16x16/Conv1_down/filter:0'      , 'D_16x16_Conv1_down_weight.npy'    ],
   ['D/16x16/Conv1_down/bias:0'        , 'D_16x16_Conv1_down_bias.npy'      ],

   ['D/8x8/FromRGB/filter:0'           , 'D_FromRGB_lod7_weight.npy'        ],
   ['D/8x8/FromRGB/bias:0'             , 'D_FromRGB_lod7_bias.npy'          ],
   ['D/8x8/Conv0/filter:0'             , 'D_8x8_Conv0_weight.npy'           ],
   ['D/8x8/Conv0/bias:0'               , 'D_8x8_Conv0_bias.npy'             ],
   ['D/8x8/Conv1_down/filter:0'        , 'D_8x8_Conv1_down_weight.npy'      ],
   ['D/8x8/Conv1_down/bias:0'          , 'D_8x8_Conv1_down_bias.npy'        ],

   ['D/4x4/FromRGB/filter:0'           , 'D_FromRGB_lod8_weight.npy'        ],
   ['D/4x4/FromRGB/bias:0'             , 'D_FromRGB_lod8_bias.npy'          ],
   ['D/4x4/Conv/filter:0'              , 'D_4x4_Conv_weight.npy'            ],
   ['D/4x4/Conv/bias:0'                , 'D_4x4_Conv_bias.npy'              ],
   ['D/4x4/Dense_0/weight:0'           , 'D_4x4_Dense0_weight.npy'          ],
   ['D/4x4/Dense_0/bias:0'             , 'D_4x4_Dense0_bias.npy'            ],
   ['D/4x4/Dense_1/weight:0'           , 'D_4x4_Dense1_weight.npy'          ],
   ['D/4x4/Dense_1/bias:0'             , 'D_4x4_Dense1_bias.npy'            ]
]

# ==============================================================================
# =                                 Test model                                 =
# ==============================================================================
if __name__ == '__main__':

   # Gen = G_style()
   
   # z = tf.linspace(-1., 1., num=1024)
   # z = tf.reshape(z, shape=[2,512])
   # y = Gen.mapping(z)
   # y = Gen.synthesis(y, rand_noise=True)
   # print(y.shape)

   # Dis = D_basic()
   # vars = Dis.trainable_variables
   # for i in range(56):
   #    print(i)
   #    src = d_vars[i][1]
   #    dst = d_vars[i][0]
   #    for var in vars:
   #       if var.name == dst:
   #          print(var.name)
   #          tmp = np.load('./stylegan-encoder/d_vars/'+src)
   #          tmp = tf.convert_to_tensor(tmp)
   #          var.assign(tmp)
   #          # print(tf.reduce_sum(var))
   #          break
   # x = np.load('/data2/01_luan_van/10x.npy')[:8].astype(np.float32)
   # x = tf.convert_to_tensor(x)
   # y = Dis(x)
   # # print(np.sum(y.numpy()))
   # print(y)
   
   create_pretrained_point()
   
   