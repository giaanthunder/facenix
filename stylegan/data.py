import os, sys, random, shutil, threading, math
import numpy as np
import tensorflow as tf
from PIL import Image


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import logging
logger = tf.get_logger()
logger.setLevel(logging.ERROR)



class img_ds():
   def __init__(self, data_dir, img_resize=256, batch_size=1, n_batch=0):
      data_dir = data_dir + '/CelebAMask-HQ/CelebA-HQ-img/'
      if os.path.isfile(data_dir):
         img_paths = [data_dir]
      else:
         ls_dir = os.listdir(data_dir)
         img_paths = []
         for file in ls_dir:
            f_path = os.path.join(data_dir, file)
            if os.path.isfile(f_path):
               img_paths.append(f_path)
      
      if n_batch == 0:
         self.n_batch = len(img_paths)//batch_size
      else:
         self.n_batch = n_batch

      self.count = self.n_batch*batch_size
      img_paths  = img_paths[:self.count]

      z_rnd = []
      for i in range(self.count):
         z = tf.random.normal([512])
         z_rnd.append(z)
      
      def load_and_preproc(z_rnd, img_path):
         img = load_img(img_path)
         img = preprocess(img, img_resize)
         return z_rnd, img
      
      
      ds = tf.data.Dataset.from_tensor_slices((z_rnd, img_paths))
      ds = ds.shuffle(buffer_size=self.count)
      ds = ds.map(load_and_preproc, num_parallel_calls=tf.data.experimental.AUTOTUNE)
      ds = ds.repeat(-1)
      ds = ds.batch(batch_size, drop_remainder=True)
      ds = ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
      self.ds = ds
   

# ==============================================================================
# =                                Utilities                                   =
# ==============================================================================
@tf.function
def preprocess(img, size):
   img = tf.image.resize(img, [size, size], tf.image.ResizeMethod.BICUBIC)
   img = tf.cast(img, tf.float32)
   img = img/127.5 - 1.
   return img

def load_img(image_path):
   img = tf.io.read_file(image_path)
   img = tf.io.decode_jpeg(img, channels=3)
   return img
   
def to_img(x):
   img = (x + 1.) * 127.5
   img = tf.clip_by_value(img, 0., 255.)
   return img

def to_file(img, img_path):
   img = tf.cast(img, dtype=tf.uint8)
   img = tf.image.encode_jpeg(img)
   tf.io.write_file(img_path, img)
   
def to_img_file(x, img_path):
   img = to_img(x)
   to_file(img, img_path)

# NCHW to NHWC
def to_nhwc(x):
   x = tf.transpose(x, [0, 2, 3, 1])
   return x

# NHWC to NCHW
def to_nchw(x):
   x = tf.transpose(x, [0, 3, 1, 2])
   return x

# ==============================================================================
# =                                  Test                                      =
# ==============================================================================
if __name__ == '__main__':
   data_dir = '/data2/01_luan_van/data/CelebAMask-HQ/CelebA-HQ-img/'

   print("Test dataset")
   dataset = img_ds(data_dir=data_dir, img_resize=1024, batch_size=8)
   data_ite = iter(dataset.ds)
   
   print("Number of batch :", dataset.n_batch)
   print("Number of images:", dataset.count)
   
   
   z, img = next(data_ite)
   print("z_rnd shape: ", z.shape)
   print("image shape: ", img.shape)
#











