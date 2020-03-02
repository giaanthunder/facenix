import os, sys, random, shutil, threading, math
import numpy as np
import tensorflow as tf
from sklearn import svm
import pickle

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import logging
logger = tf.get_logger()
logger.setLevel(logging.ERROR)
sys.path.append(os.path.abspath('..'))

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
   'Young'              #, 'Pose'               , 'FIX'
]

# ==============================================================================
# =                                  main                                      =
# ==============================================================================
if __name__ == '__main__':
   model_name = 'stylegan'
   CUR_DIR   = os.path.dirname(os.path.abspath(__file__))
   MODEL_DIR = CUR_DIR + '/output/' + model_name + '/boundary/'

   out_dir = CUR_DIR + '/output/'+model_name+'/data/'
   w_data  = np.load(out_dir+'w_data.npy')
   y_data  = np.load(out_dir+'y_data.npy')


   for i, att in enumerate(atts):
      try: 
         labels  = y_data[:,i]
         pos_idx = tf.squeeze(tf.where(labels==1))
         neg_idx = tf.squeeze(tf.where(labels==0))
         
         pos_w = tf.gather(w_data, pos_idx)
         neg_w = tf.gather(w_data, neg_idx)
         
         pos_y = tf.gather(labels, pos_idx)
         neg_y = tf.gather(labels, neg_idx)
         
         pos_num = pos_y.shape[0]
         neg_num = neg_y.shape[0]
         
         
         if pos_num < neg_num:
            neg_w = neg_w[:pos_num]
            neg_y = neg_y[:pos_num]
         else:
            pos_w = pos_w[:neg_num]
            pos_y = pos_y[:neg_num]
         
         val_idx = tf.cast(pos_num*0.2, dtype=tf.int64).numpy()
         
         train_w = tf.concat([pos_w[val_idx:], neg_w[val_idx:]], axis=0).numpy()
         train_y = tf.concat([pos_y[val_idx:], neg_y[val_idx:]], axis=0).numpy()
         
         val_w = tf.concat([pos_w[:val_idx], neg_w[:val_idx]], axis=0).numpy()
         val_y = tf.concat([pos_y[:val_idx], neg_y[:val_idx]], axis=0).numpy()

         clf = svm.LinearSVC()
         classifier = clf.fit(train_w, train_y)

         val_pred = classifier.predict(val_w)
         correct = np.where(val_pred==val_y,1,0)
         acc = np.sum(correct)/ (2*val_idx)
         print(att, ' '*(25-len(att)), acc)

         vector_b = classifier.coef_
         vector_b = vector_b/np.linalg.norm(vector_b)
         np.save(CUR_DIR+'/output/'+model_name+'/' + att + '.npy', vector_b)
         pickle.dump(classifier, open(MODEL_DIR+att+'_model.sav', 'wb'))
      except Exception as e:
         print(att, ' '*(25-len(att)), str(e))
         continue
   




