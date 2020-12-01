import tensorflow as tf
import numpy as np
import tensorflow_addons as tfa
from tensorflow import keras
from tensorflow.keras import layers
import math, sys, os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import logging
logger = tf.get_logger()
logger.setLevel(logging.ERROR)

# ==============================================================================
# =                                 BiSeNet                                    =
# ==============================================================================
class BiSeNet(keras.Model):
   def __init__(self, n_class):
      super(BiSeNet, self).__init__()
      self.SP   = SpatialPath()
      self.CP   = ContextPath()
      self.FFM  = FFM(256)
      self.out1 = OutBlock(filters=256, n_class=n_class)
      self.out2 = OutBlock(filters= 64, n_class=n_class)
      self.out3 = OutBlock(filters= 64, n_class=n_class)

   def call(self, x):
      feat_sp8 , feat_sp16, feat_sp32 = self.SP(x)
      feat_cp8, feat_cp16 = self.CP(feat_sp16, feat_sp32)
      feat_fuse = self.FFM(feat_sp8, feat_cp8)
      
      feat_out8  = self.out1(feat_fuse)
      feat_out8  = layers.UpSampling2D(size=[ 8,  8])(feat_out8)
      feat_out16 = self.out2(feat_cp8)
      feat_out16 = layers.UpSampling2D(size=[ 8,  8])(feat_out16)
      feat_out32 = self.out3(feat_cp16)
      feat_out32 = layers.UpSampling2D(size=[16, 16])(feat_out32)
      
      return feat_out8, feat_out16, feat_out32

# ==============================================================================
# =                             Spatial Path (ResNet)                          =
# ==============================================================================
class SpatialPath(keras.layers.Layer):
   def __init__(self):
      super(SpatialPath, self).__init__()
      self.conv_7x7  = conv_block(filters=64, kernel_size=7, strides=2, activation='relu')
      self.max_pool  = layers.MaxPool2D(pool_size=(3, 3), strides=2, padding='valid')
      self.res_blk_1 = res_block(filters=64, strides=1)
      self.res_blk_2 = res_block(filters=128, strides=2, sync_shape=True) # filters change, stride != 1
      self.res_blk_3 = res_block(filters=256, strides=2, sync_shape=True)
      self.res_blk_4 = res_block(filters=512, strides=2, sync_shape=True)
      
   def call(self, x):
      x = self.conv_7x7(x)
      x = tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]]) # padding = 1
      x = self.max_pool(x)
      x = self.res_blk_1(x)
      feat8  = self.res_blk_2(x)
      feat16 = self.res_blk_3(feat8)
      feat32 = self.res_blk_4(feat16)
      
      return feat8, feat16, feat32
      

# ==============================================================================
# =                             Context Path (Xception)                        =
# ==============================================================================
class ContextPath(keras.layers.Layer):
   def __init__(self):
      super(ContextPath, self).__init__()
      self.arm16       = ARM(128)
      self.arm32       = ARM(128)
      self.conv_3x3_32 = conv_block(filters=128, activation='relu')
      self.conv_3x3_16 = conv_block(filters=128, activation='relu')
      self.conv_1x1    = conv_block(filters=128, kernel_size=1, activation='relu')
      
   def call(self, feat16, feat32):
      _, h32, w32, _ = feat32.shape
   
      feat32_avg = global_avg(feat32)
      feat32_avg = self.conv_1x1(feat32_avg)
      feat32_up1 = layers.UpSampling2D(size=[h32, w32])(feat32_avg) # (1)
      
      feat32_arm = self.arm32(feat32) # (2)
      feat32_sum = feat32_up1 + feat32_arm # (1) + (2)
      feat32_up2 = layers.UpSampling2D(size=[2, 2])(feat32_sum)
      feat32_up2 = self.conv_3x3_32(feat32_up2) # (3)
      
      feat16_arm = self.arm16(feat16) # (4)
      feat16_sum = feat32_up2 + feat16_arm # (3) + (4)
      feat16_up  = layers.UpSampling2D(size=[2, 2])(feat16_sum)
      feat16_up  = self.conv_3x3_16(feat16_up)
      
      return feat16_up, feat32_up2

      
# ==============================================================================
# =                        Attention Refinement Module                         =
# ==============================================================================
class ARM(keras.layers.Layer):
   def __init__(self, filters):
      super(ARM, self).__init__()
      self.conv_3x3 = conv_block(filters=filters, activation='relu')
      self.conv_1x1 = conv_block(filters=filters, kernel_size=1, activation='sigmoid')
      
   def call(self, x):
      feat  = self.conv_3x3(x)
      atten = global_avg(feat)
      atten = self.conv_1x1(atten)
      out   = feat * atten
      return out

      
# ==============================================================================
# =                           Feature Fusion Module                            =
# ==============================================================================
class FFM(keras.layers.Layer):
   def __init__(self, filters):
      super(FFM, self).__init__()
      self.conv_1 = conv_block(filters=filters, kernel_size=1, activation='relu')
      self.conv_2 = conv_block(filters=filters//4, kernel_size=1, norm='none', activation='relu')
      self.conv_3 = conv_block(filters=filters, kernel_size=1, norm='none', activation='sigmoid')
      
   def call(self, feat_sp, feat_cp):
      fcat  = tf.concat([feat_sp, feat_cp], axis=3)
      feat  = self.conv_1(fcat)
      atten = global_avg(feat)
      atten = self.conv_2(atten)
      atten = self.conv_3(atten)
      out   = atten*feat + feat
      return out

# ==============================================================================
# =                                 Out Block                                  =
# ==============================================================================
class OutBlock(keras.layers.Layer):
   def __init__(self, filters, n_class):
      super(OutBlock, self).__init__()
      self.conv_1 = conv_block(filters=filters, activation='relu')
      self.conv_2 = conv_block(filters=n_class, kernel_size=1, norm='none')
      
   def call(self, x):
      x = self.conv_1(x)
      x = self.conv_2(x)
      return x
      

# ==============================================================================
# =                               Custom layers                                =
# ==============================================================================
class res_block(layers.Layer):
   def __init__(self, filters, strides, sync_shape=False):
      super(res_block, self).__init__()

      # block 1
      self.conv_block_1 = conv_block(filters=filters, strides=strides, activation='relu')
      self.conv_block_2 = conv_block(filters=filters)
      if sync_shape:
         self.sync_block = conv_block(filters=filters, kernel_size=1, strides=strides)
      else:
         self.sync_block = layers.Activation('linear')
         
      # block 2
      self.conv_block_3 = conv_block(filters=filters, activation='relu')
      self.conv_block_4 = conv_block(filters=filters)
      

   
   def call(self, x):
      x1 = self.conv_block_1(x)
      x1 = self.conv_block_2(x1)
      x  = self.sync_block(x)
      x1 = x1 + x
      x1 = layers.ReLU()(x1)
      
      x2 = self.conv_block_3(x1)
      x2 = self.conv_block_4(x2)
      x2 = x2 + x1
      x2 = layers.ReLU()(x2)
      return x2

class conv_block(layers.Layer):
   def __init__(self, filters, kernel_size=3, strides=1, norm='batchnorm', activation='linear'):
      super(conv_block, self).__init__()
      p = kernel_size//2
      self.padding = [[0, 0], [p, p], [p, p], [0, 0]]
      self.conv = layers.Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding='valid', use_bias=False)
      if norm == 'batchnorm':
         self.norm = layers.BatchNormalization(epsilon=1e-5)
      else:
         self.norm = layers.Activation('linear')
      self.acti = layers.Activation(activation)

   def call(self, x):
      x = tf.pad(x, self.padding)
      x = self.conv(x)
      x = self.norm(x)
      x = self.acti(x)
      return x

def global_avg(x):
   bs, h, w, c = x.shape
   x = layers.GlobalAveragePooling2D()(x)
   x = tf.reshape(x, [bs, 1, 1, c])
   return x
      

def pretrained_models(experiment_name = 'origin'):
   bnet = BiSeNet(19)
   x = tf.ones([1, 512, 512, 3])
   out, out16, out32 = bnet(x)
   checkpoint_path = BASE_DIR+'/bisenet/output/%s/bisenet.ckpt' % experiment_name
   bnet.load_weights(checkpoint_path)
   return bnet

#

# ==============================================================================
# =                                 Test model                                 =
# ==============================================================================
if __name__ == '__main__':
   net = BiSeNet(19)
   in_ten = tf.random.uniform([16, 512, 512, 3])
   out, out16, out32 = net(in_ten)
   checkpoint_path = './ouput/origin/bisenet.ckpt'
   net.load_weights(checkpoint_path)
   
   x = tf.io.read_file("./0.jpg")
   x = tf.io.decode_jpeg(x, channels=3)
   x = tf.image.resize(x, [512,512])
   x = tf.cast(x, dtype=tf.float32)
   x = x/255.
   
   h, w, c = x.shape
   mean = tf.convert_to_tensor((0.485, 0.456, 0.406))
   mean = tf.reshape(mean, [1,1,-1])
   mean = tf.tile(mean, [h,w,1])
   std  = tf.convert_to_tensor((0.229, 0.224, 0.225))
   std  = tf.reshape(std, [1,1,-1])
   std  = tf.tile(std, [h,w,1])
   
   x = (x - mean) / std
   
   x = tf.expand_dims(x, axis=0)


   
   
   out, out16, out32 = net(x)
   
   img = out[0].numpy()
   
   img = np.transpose(img, [2,0,1])
   img = np.argmax(img, axis=0)
   #min = np.min(img)
   #max = np.max(img)
   #img = ((img - min) * (1/(max - min)) * 255)
   #img = np.expand_dims(img, axis=-1)

   
   from PIL import Image
   for i in range(19):
      im = np.where(img==i, 255, 0).astype('uint8')
      im = Image.fromarray(im)
      im.save('./sample_%02d.jpg'%i)

   
   #a = out.numpy()
   #print(a.shape)
   #print(a)
   #sys.exit()
   
   
   
