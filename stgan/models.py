import tensorflow as tf
tf.get_logger().setLevel('INFO')
import tensorflow_addons as tfa
from tensorflow.python.keras import layers
from tensorflow.python import keras

import math, sys, os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ==============================================================================
# =                                   Encoder                                  =
# ==============================================================================
class Encoder(keras.Model):
   def __init__(self):
      super(Encoder, self).__init__()
      self.layer = [
         enc_conv_block(64),
         enc_conv_block(128),
         enc_conv_block(256),
         enc_conv_block(512),
         enc_conv_block(1024)
      ]
      
   def call(self, z):
      zs = []
      n = len(self.layer)
      for i in range(n):
         z = self.layer[i](z)
         zs.append(z)
      return zs
      

# ==============================================================================
# =                                  Generator                                 =
# ==============================================================================
class Generator(keras.Model):
   def __init__(self):
      super(Generator, self).__init__()
      self.layer = [
         gen_conv_block(1024),
         gen_conv_block(512),
         gen_conv_block(256),
         gen_conv_block(128),
         to_rgb()
      ]

   def call(self, zs, a):
      n = len(self.layer)
      x = None
      for i in range(n):
         x = self.layer[i](x, zs[-(i+1)], a)
      return x

      
# ==============================================================================
# =                               Discriminator                                =
# ==============================================================================
class Discriminator(keras.Model):
   def __init__(self, n_att):
      super(Discriminator, self).__init__()
      self.layer = [
         dis_conv_block(64),
         dis_conv_block(128),
         dis_conv_block(256),
         dis_conv_block(512),
         dis_conv_block(1024)
      ]
      
      self.dis = dis_block()
      self.cls = cls_block(n_att)
      
   def call(self, x, mode="all"):
      n = len(self.layer)
      for i in range(n):
         x = self.layer[i](x)
      if mode == "dis":
         dis = self.dis(x)
         return dis
      if mode == "cls":
         cls = self.cls(x)
         return cls
      dis = self.dis(x)
      cls = self.cls(x)
      return dis, cls

      
# ==============================================================================
# =                                    STU                                     =
# ==============================================================================
class STU(keras.Model):
   def __init__(self):
      super(STU, self).__init__()
      self.layer = [
         stu_block(512),
         stu_block(256),
         stu_block(128),
         stu_block(64)
      ]
      
   def call(self, zs, a):
      n = len(self.layer)
      new_zs = [zs[-1]]
      new_state = zs[-1]
      for i in range(n):
         output, new_state = self.layer[i](new_state, zs[-(i+2)], a)
         new_zs.insert(0, output)
      return new_zs


# ==============================================================================
# =                               Custom layers                                =
# ==============================================================================
class enc_conv_block(layers.Layer):
   def __init__(self, filters):
      super(enc_conv_block, self).__init__()

      self.conv = layers.Conv2D(filters=filters, kernel_size=4, strides=2, padding='same', use_bias=False)
      self.norm = layers.BatchNormalization()
      self.acti = layers.LeakyReLU(alpha=0.2)
   
   def call(self, x):
      x = self.conv(x)
      x = self.norm(x)
      x = self.acti(x)
      return x

class gen_conv_block(layers.Layer):
   def __init__(self, filters):
      super(gen_conv_block, self).__init__()

      self.conv = layers.Conv2DTranspose(filters=filters, kernel_size=4, strides=2, padding='same', use_bias=False)
      self.norm = layers.BatchNormalization()
      self.acti = layers.ReLU()

   def call(self, z, zs, a):
      x = concat(z, zs, a)
      x = self.conv(x)
      x = self.norm(x)
      x = self.acti(x)
      return x
    
class to_rgb(layers.Layer):
   def __init__(self):
      super(to_rgb, self).__init__()

      self.conv = layers.Conv2DTranspose(filters=3, kernel_size=4, strides=2, padding='same')
      self.acti = layers.Activation("tanh")

   def call(self, z, zs, a):
      x = concat(z, zs, a)
      x = self.conv(x)
      x = self.acti(x)
      return x
    
class dis_conv_block(layers.Layer):
   def __init__(self, filters):
      super(dis_conv_block, self).__init__()

      self.conv = layers.Conv2D(filters=filters, kernel_size=4, strides=2, padding='same', use_bias=False)
      self.norm = tfa.layers.InstanceNormalization()
      self.acti = layers.LeakyReLU(alpha=0.2)

   def call(self, x):
      x = self.conv(x)
      x = self.norm(x)
      x = self.acti(x)
      return x


class dis_block(layers.Layer):
   def __init__(self):
      super(dis_block, self).__init__()

      self.flatten = layers.Flatten()
      self.dense_1 = layers.Dense(1024)
      self.dense_2 = layers.Dense(1)

   def call(self, x):
      x = self.flatten(x)
      x = tf.nn.leaky_relu(self.dense_1(x))
      x = self.dense_2(x)
      return x

class cls_block(layers.Layer):
   def __init__(self, n_att):
      super(cls_block, self).__init__()

      self.flatten = layers.Flatten()
      self.dense_1  = layers.Dense(1024)
      self.dense_2  = layers.Dense(n_att)

   def call(self, x):
      x = self.flatten(x)
      x = tf.nn.leaky_relu(self.dense_1(x))
      x = self.dense_2(x)
      return x


class stu_block(layers.Layer):
   def __init__(self, filters):
      super(stu_block, self).__init__()

      self.upsample   = layers.Conv2DTranspose(filters=filters, kernel_size=4, strides=2, padding='same', use_bias=True)
      self.reset_gate = layers.Conv2D(filters=filters, kernel_size=3, padding='same', use_bias=True, activation='sigmoid')
      self.update_gate= layers.Conv2D(filters=filters, kernel_size=3, padding='same', use_bias=True, activation='sigmoid')
      self.info_gate  = layers.Conv2D(filters=filters, kernel_size=3, padding='same', use_bias=True, activation='tanh')

   def call(self, state, zs, a):
      state       = concat(state, None, a)
      state       = self.upsample(state)
      data        = tf.concat([zs, state], axis=3)
      reset_gate  = self.reset_gate(data)
      update_gate = self.update_gate(data)
      new_state   = reset_gate * state
      data        = tf.concat([zs, new_state], axis=3)
      new_info    = self.info_gate(data)
      output      = (1-update_gate)*state + update_gate*new_info

      return output, new_state


# functions for custom layers
def concat(z, zs, a):
   feats = []
   if z is not None:
      _, h, w, _ = z.shape
      feats.append(z)
   if zs is not None:
      _, h, w, _ = zs.shape
      feats.append(zs)
   if a is not None:
      a = tf.reshape(a, shape=[-1, 1, 1, a.shape[-1]])
      a = tf.tile(a, [1, h, w, 1])
      feats.append(a)
   return tf.concat(feats, axis=3)

def pretrained_models(experiment_name = 'origin'):
   Enc = Encoder()
   Gen = Generator()
   Stu = STU()

   x = tf.ones(shape=[1,128,128,3], dtype=tf.float32)
   a = tf.ones(shape=[1,13], dtype=tf.float32)

   z      = Enc(x)
   z_stu  = Stu(z, a)
   x_fake = Gen(z_stu, a)

   checkpoint_dir = BASE_DIR+'/stgan/output/%s/trained_model' % experiment_name
   checkpoint = tf.train.Checkpoint(
      Enc=Enc, Gen=Gen, Stu=Stu
   )
   manager = tf.train.CheckpointManager(checkpoint, checkpoint_dir, max_to_keep=3)

   if manager.latest_checkpoint:
      checkpoint.restore(manager.latest_checkpoint).expect_partial()
      return Enc, Gen, Stu
   else:
      return None
   
#

# ==============================================================================
# =                                 Test model                                 =
# ==============================================================================
if __name__ == '__main__':
   gen = Generator(512)
   dis = Discriminator(512)
   enc = Encoder(512)
   
