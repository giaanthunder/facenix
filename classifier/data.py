import os, sys, random, shutil, threading, math
import numpy as np
import tensorflow as tf
#from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2, preprocess_input
#from tensorflow.keras.applications import NASNetLarge
#from tensorflow.keras.applications.nasnet import preprocess_input
#from tensorflow.keras.applications.resnet_v2 import ResNet101V2, preprocess_input
#from tensorflow.keras.applications.xception import Xception, preprocess_input
#from tensorflow.keras.applications.resnet_v2 import ResNet50V2, preprocess_input
#from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
#from tensorflow.keras.applications.resnet import ResNet50, preprocess_input
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from PIL import Image


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import logging
logger = tf.get_logger()
logger.setLevel(logging.ERROR)

att_dict = {
   '5_o_Clock_Shadow'   : 0 , 'Arched_Eyebrows'  : 1 , 'Attractive'     : 2 ,
   'Bags_Under_Eyes'    : 3 , 'Bald'             : 4 , 'Bangs'          : 5 ,
   'Big_Lips'           : 6 , 'Big_Nose'         : 7 , 'Black_Hair'     : 8 ,
   'Blond_Hair'         : 9 , 'Blurry'           : 10, 'Brown_Hair'     : 11,
   'Bushy_Eyebrows'     : 12, 'Chubby'           : 13, 'Double_Chin'    : 14,
   'Eyeglasses'         : 15, 'Goatee'           : 16, 'Gray_Hair'      : 17,
   'Heavy_Makeup'       : 18, 'High_Cheekbones'  : 19, 'Male'           : 20,
   'Mouth_Slightly_Open': 21, 'Mustache'         : 22, 'Narrow_Eyes'    : 23,
   'No_Beard'           : 24, 'Oval_Face'        : 25, 'Pale_Skin'      : 26,
   'Pointy_Nose'        : 27, 'Receding_Hairline': 28, 'Rosy_Cheeks'    : 29,
   'Sideburns'          : 30, 'Smiling'          : 31, 'Straight_Hair'  : 32,
   'Wavy_Hair'          : 33, 'Wearing_Earrings' : 34, 'Wearing_Hat'    : 35,
   'Wearing_Lipstick'   : 36, 'Wearing_Necklace' : 37, 'Wearing_Necktie': 38,
   'Young'              : 39
}

class img_ds():
   def __init__(self, data_dir, att, img_resize=299, batch_size=1, part='train'):
      # create list of image_paths
      img_dir   = os.path.join(data_dir, 'CelebA-HQ-img')
      list_file = os.path.join(data_dir, 'CelebAMask-HQ-attribute-anno.txt')

      #img_dir   = os.path.join(data_dir, 'img_align_celeba_crop_128')
      #list_file = os.path.join(data_dir, 'list_attr_celeba.txt')

      names     = np.loadtxt(list_file, skiprows=2, usecols=[0], dtype=np.str)
      img_paths = [os.path.join(img_dir, name) for name in names]

      # create list of labels
      att_id = att_dict[att]+1
      labels = np.loadtxt(list_file, skiprows=2, usecols=att_id, dtype=np.int64)
      
      if part == 'train':
         img_paths = img_paths[:26000]
         labels    =    labels[:26000]
      elif part == 'val':
         img_paths = img_paths[26000:28000]
         labels    =    labels[26000:28000]
      elif part == 'test':
         img_paths = img_paths[28000:]
         labels    =    labels[28000:]
      
      
      
      self.n_batch = len(labels)//batch_size
      self.count   = self.n_batch * batch_size

      
      def load_and_preproc(img_path, label):
         img = load_img(img_path)
         img = preprocess(img, img_resize)
         #img = preprocess_input(img)
         lbl = (label + 1)//2
         #lbl = tf.cast(lbl, dtype=tf.float32)
         lbl = tf.one_hot(indices=lbl, depth=2, on_value=1., off_value=0.)
         return img, lbl
      
      
      ds = tf.data.Dataset.from_tensor_slices((img_paths, labels))
      ds = ds.shuffle(buffer_size=self.count)
      ds = ds.map(load_and_preproc, num_parallel_calls=tf.data.experimental.AUTOTUNE)
      ds = ds.repeat(-1)
      ds = ds.batch(batch_size, drop_remainder=True)
      ds = ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
      self.ds = ds
   

# ==============================================================================
# =                                Utilities                                   =
# ==============================================================================
def load_img(image_path):
   img = tf.io.read_file(image_path)
   img = tf.io.decode_jpeg(img, channels=3)
   return img

@tf.function
def preprocess(img, size):
   img = Standardize(img, size)
   return img

def Standardize(img, size):
   img = tf.image.resize(img, [size, size], tf.image.ResizeMethod.NEAREST_NEIGHBOR)
   img = tf.cast(img, tf.float32)
   img = img/255.
   #img = img/127.5 - 1.
   return img

def to_img(x):
   img = (x + 1.) * 127.5
   img = tf.clip_by_value(img, 0., 255.)
   #img = tf.cast(img, dtype=tf.uint8)
   return img
   
def to_jpg_file(x, img_path):
   img = to_img(x)
   img = tf.cast(img, dtype=tf.uint8)
   img = tf.image.encode_jpeg(img)
   tf.io.write_file(img_path, img)

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
   data_dir = '/data2/01_luan_van/data/CelebAMask-HQ/'

   print("Test dataset")
   dataset = img_ds(data_dir=data_dir, att='Young', img_resize=299, batch_size=8)
   data_ite = iter(dataset.ds)
   
   print("Number of batch :", dataset.n_batch)
   print("Number of images:", dataset.count)
   
   
   img, label = next(data_ite)
   print("image shape: ", img.shape)
   print("label shape: ", label.shape)
   
   img = to_jpg_file(img[0], './0.jpg')
   print(label[0].numpy())
   
   
#











