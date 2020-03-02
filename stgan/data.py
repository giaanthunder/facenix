from __future__ import absolute_import, division, print_function
import os, sys
import numpy as np
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

AUTOTUNE = tf.data.experimental.AUTOTUNE


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

#@tf.function
def img_ds(data_dir, atts, img_resize, batch_size,
             prefetch_batch=2, drop_remainder=True, shuffle=True,
             repeat=-1, part='train'):
   # create list of img_paths
   list_file = os.path.join(data_dir, 'list_attr_celeba.txt')
   img_dir   = os.path.join(data_dir, 'img_align_celeba_crop_128')

   names     = np.loadtxt(list_file, skiprows=2, usecols=[0], dtype=np.str)
   img_paths = [os.path.join(img_dir, name) for name in names]

   # create list of labels
   att_id = [att_dict[att] + 1 for att in atts]
   labels = np.loadtxt(list_file, skiprows=2, usecols=att_id, dtype=np.int64)
   
   
   if part == 'test':
      img_paths = img_paths[182637:182638]
      labels    =    labels[182637:182638]
   elif part == 'val':
      img_paths = img_paths[182000:182637]
      labels    =    labels[182000:182637]
   else: # 'train'
      img_paths = img_paths[:120000]
      labels    =    labels[:120000]
      
      
   def load_and_preproc(img_path, label):
      img = load_img(img_path)
      
      #img = tf.image.crop_to_bounding_box(img, 26, 3, 170, 170)
      
      img = preprocess(img, img_resize)
      label = (label+1) / 2
      label = tf.cast(label, tf.float32)
      return img, label
      
   ds = tf.data.Dataset.from_tensor_slices((img_paths, labels))
   ds = ds.map(load_and_preproc, num_parallel_calls=AUTOTUNE)
   #ds = ds.cache(filename='./ds_cache')

   img_count = len(img_paths)
   if shuffle:
      ds = ds.shuffle(buffer_size=4096)

   ds = ds.repeat(repeat)
   ds = ds.batch(batch_size, drop_remainder=drop_remainder)
   ds = ds.prefetch(buffer_size=prefetch_batch)

   return img_count, ds
   
def preprocess(img, size):
   img = tf.image.resize(img, [size, size], tf.image.ResizeMethod.BICUBIC)
   img = tf.cast(img, tf.float32)
   img = tf.clip_by_value(img, 0, 255) / 127.5 - 1
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


import cv2, dlib
def align_face(img, lm_path):
   detector  = dlib.get_frontal_face_detector()
   predictor = dlib.shape_predictor(lm_path)
   
   
# set confilcting att from 1 to 0 
# att_batch: batch of att array
# att      : att will be checked
# att_lst  : list of chosen att
@staticmethod
def check_attribute_conflict(att_batch, att, att_lst):
   def set_att(att_array, value, att):
      if att in att_lst:
         id = att_lst.index(att)
         att_array[id] = value
   
   id = att_lst.index(att)
   for att_array in att_batch:
      if att_array[id] == 1:
         if att in ['Bald', 'Receding_Hairline']:
            set_att(att_array, 0, 'Bangs')
         elif att == 'Bangs':
            set_att(att_array, 0, 'Bald')
            set_att(att_array, 0, 'Receding_Hairline')
         elif att in ['Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Gray_Hair']:
            for conflict_att in ['Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Gray_Hair']:
               if att != conflict_att:
                  set_att(att_array, 0, conflict_att)
         elif att in ['Straight_Hair', 'Wavy_Hair']:
            for conflict_att in ['Straight_Hair', 'Wavy_Hair']:
               if att != conflict_att:
                  set_att(att_array, 0, conflict_att)

   return att_batch


if __name__ == '__main__':
   atts = ['Bald', 'Bangs', 'Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Bushy_Eyebrows', 'Eyeglasses', 'Male', 'Mouth_Slightly_Open', 'Mustache', 'No_Beard', 'Pale_Skin', 'Young']
   
   print("Tensorflow ver: ",tf.__version__)
   print("Eager enable: ",tf.executing_eagerly())
   
   img_count, data = img_ds(data_dir='/data2/01_luan_van/data/', atts=atts, img_resize=128, 
      batch_size=32, part='val')
   
   print(data)
   
   for _ in range(1):
      image_batch, label_batch = next(iter(data))
      print("img: ", image_batch)
      print("label: ", label_batch)
#











