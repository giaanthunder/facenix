from django.db import models

# Create your models here.

# ==== tensorflow ====
import tensorflow as tf
import numpy as np
import os, sys, shutil
sys.path.append(os.path.abspath('..'))
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import stgan
from face_alignment.face_alignment import image_align
from face_alignment.landmarks_detector import LandmarksDetector

# ==============================================================================
# =                                   STGAN                                    =
# ==============================================================================
class StganGUI():
   def __init__(self):
      self.Enc, self.Gen, self.Stu = stgan.models.pretrained_models()
      self.atts = [
         'Bald', 'Bangs', 'Black_Hair', 
         'Blond_Hair', 'Brown_Hair', 'Bushy_Eyebrows', 
         'Eyeglasses', 'Male', 'Mouth_Slightly_Open', 
         'Mustache', 'No_Beard', 'Pale_Skin', 
         'Young'
      ]

   def set_res(self, cur_zs_path, res_path, att=tf.zeros(shape=[1, 13], dtype=tf.float32)):
      zs = []
      for i in range(5):
         z = np.load(cur_zs_path+'/z%d.npy'%(i))

         zs.append(tf.convert_to_tensor(z))
      img = self.Gen(self.Stu(zs, att), att)[0]
      stgan.data.to_img_file(img, res_path)
      
   def align_img(self, ori_path, ali_path):
      detector = LandmarksDetector('shape_predictor_68_face_landmarks.dat')
      for i, face_landmarks in enumerate(detector.get_landmarks(ori_path), start=1):
         image_align(ori_path, ali_path, face_landmarks)

   def find_zs(self, ori_path, rs_zs_path, cur_zs_path):
      img  = stgan.data.load_img(ori_path)
      img  = stgan.data.preprocess(img, size=128)
      img  = tf.expand_dims(img, axis=0)
      zs = self.Enc(img)
      for i in range(5):
         np.save(cur_zs_path+'/z%d.npy'%(i), zs[i].numpy())
         np.save( rs_zs_path+'/z%d.npy'%(i), zs[i].numpy())

   def att_click(self, value, cur_zs_path, res_path):
      values = value.split("_")
      val_lst = {"Unchanged":0, "Add":1, "Remove":-1}
      att = []
      for val in values:
         att.append(val_lst[val])
      print(att)
      att = tf.convert_to_tensor([att], dtype=tf.float32)
      self.set_res(cur_zs_path, res_path, att)


stgangui = StganGUI()
# ==============