from django.db import models

# Create your models here.

# ==== tensorflow ====
import tensorflow as tf
import numpy as np
import os, sys, shutil
sys.path.append(os.path.abspath('..'))
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import boundary_finder, stylegan
from face_alignment.face_alignment import image_align
from face_alignment.landmarks_detector import LandmarksDetector


# ==============================================================================
# =                                StyleGAN                                    =
# ==============================================================================
class StyleGUI():
   def __init__(self):
      atts = [
         '5_o_Clock_Shadow'   , 'Arched_Eyebrows'    , 'Attractive'         , 
         'Bags_Under_Eyes'    , 'Bald'               , 'Bangs'              , 
         'Big_Lips'           , 'Big_Nose'           , 'Black_Hair'         , 
         'Blond_Hair'         , 'Blurry'             , 'Brown_Hair'         , 
         'Bushy_Eyebrows'     , 'Chubby'             , 'Double_Chin'        , 
         'Eyeglasses'         , 'Goatee'             , 'Gray_Hair'          , 
         'Heavy_Makeup'       , 'High_Cheekbones'    , 'Male'               , 
         'Mouth_Slightly_Open', 'Mustache'           , 'Narrow_Eyes'        , 
         'No_Beard'           , 'Oval_Face'          , 'Pale_Skin'          , 
         'Pointy_Nose'        , 'Receding_Hairline'  , 'Rosy_Cheeks'        , 
         'Sideburns'          , 'Smiling'            , 'Straight_Hair'      , 
         'Wavy_Hair'          , 'Wearing_Earrings'   , 'Wearing_Hat'        , 
         'Wearing_Lipstick'   , 'Wearing_Necklace'   , 'Wearing_Necktie'    , 
         'Young'              , 'Pose'               , 'FIX'
      ]
      
      self.Gen = stylegan.models.pretrained_models()
      self.Enc = stylegan.models.Encoder(self.Gen)

      self.att_vectors = boundary_finder.test.get_att_vectors(name='stylegan')
      
   def align_img(self, ori_path, ali_path):
      detector = LandmarksDetector('shape_predictor_68_face_landmarks.dat')
      for i, face_landmarks in enumerate(detector.get_landmarks(ori_path), start=1):
         image_align(ori_path, ali_path, face_landmarks)
      
   def find_z(self, ori_path, cur_z_path, rs_z_path, opt_ite=1000):
      img  = stylegan.data.load_img(ori_path)
      img  = tf.expand_dims(img, axis=0)
      cur_w, _ = self.Enc(img, opt_ite)
      np.save(cur_z_path+"/w.npy", cur_w.numpy())
      np.save(rs_z_path+"/w.npy", cur_w.numpy())


   def set_res(self, cur_z_path, res_path):
      cur_w = np.load(cur_z_path+"/w.npy")
      img = self.Gen.synthesis(cur_w)[0]
      stylegan.data.to_img_file(img, res_path)
      
   def att_click(self, att_name, value, cur_z_path):
      cur_w = np.load(cur_z_path+"/w.npy")
      att = self.att_vectors[att_name] * 0.5
      if value=='add':
         cur_w = cur_w + att
      if value=='minus':
         cur_w = cur_w - att
      np.save(cur_z_path+"/w.npy", cur_w)


   def rand_face(self, cur_z_path, rs_z_path):
      z = tf.random.normal([1,512], -1., 1.)
      cur_w = self.Gen.mapping(z)
      np.save(cur_z_path+"/w.npy", cur_w.numpy())
      np.save(rs_z_path+"/w.npy", cur_w.numpy())




stylegui = StyleGUI()
# ==============