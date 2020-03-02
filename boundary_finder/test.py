import os, sys, random, shutil, threading, math
import numpy as np
import tensorflow as tf
from sklearn import svm

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

def get_att_vectors(name):
   CUR_DIR = os.path.dirname(os.path.realpath(__file__))
   att_vectors = {}
   for att in atts:
      try:
         b = np.load(CUR_DIR+'/output/'+name+'/' + att + '.npy')
      except:
         # print('Cannot load:', dir_path+'/output/'+name+'/' + att + '.npy')
         continue
      att_vectors[att] = b
   return att_vectors