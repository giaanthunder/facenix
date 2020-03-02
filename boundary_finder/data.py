import os, sys, random, shutil, threading, math
import numpy as np
import tensorflow as tf


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import logging
logger = tf.get_logger()
logger.setLevel(logging.ERROR)

exp_names = [
   '5_o_Clock_Shadow',
   'Arched_Eyebrows',
   'Attractive',
   'Bags_Under_Eyes',
   'Bald',
   'Bangs',
   'Big_Lips',
   'Big_Nose',
   'Black_Hair',
   'Blond_Hair',
   'Blurry',
   'Brown_Hair',
   'Bushy_Eyebrows',
   'Chubby',
   'Double_Chin',
   'Eyeglasses',
   'Goatee',
   'Gray_Hair',
   'Heavy_Makeup',
   'High_Cheekbones',
   'Male',
   'Mouth_Slightly_Open',
   'Mustache',
   'Narrow_Eyes',
   'No_Beard',
   'Oval_Face',
   'Pale_Skin',
   'Pointy_Nose',
   'Receding_Hairline',
   'Rosy_Cheeks',
   'Sideburns',
   'Smiling',
   'Straight_Hair',
   'Wavy_Hair',
   'Wearing_Earrings',
   'Wearing_Hat',
   'Wearing_Lipstick',
   'Wearing_Necklace',
   'Wearing_Necktie',
   'Young'
]


# ==============================================================================
# =                                  main                                      =
# ==============================================================================
if __name__ == '__main__':
   num_data = 200
   model_name = 'stylegan'
   
   BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
   CUR_DIR   = os.path.dirname(os.path.abspath(__file__))
   sys.path.append(BASE_DIR)
   import classifier
   
   if model_name == 'stylegan':
      import stylegan
      Gen = stylegan.models.pretrained_models()

   print("======= Load pretrained classifiers =======")
   cls_lst = []
   for exp in exp_names:
      Cls = classifier.models.Classifier()
      checkpoint = tf.train.Checkpoint(Cls=Cls)
      checkpoint_dir = os.path.join(BASE_DIR+'/classifier/output/', exp)
      manager = tf.train.CheckpointManager(checkpoint, checkpoint_dir, max_to_keep=1)
      checkpoint.restore(manager.latest_checkpoint)
      if manager.latest_checkpoint is None:
         print(exp + ' is None')
      cls_lst.append(Cls)
   
   
   print("======= Generate %d0 samples (w, y) ======="%num_data)
   w_data = []
   y_data = []
   
   for i in range(num_data):
      print('%d0'%(i+1))
      z = tf.random.uniform([10,512], -1., 1.)
      
      if model_name == 'stylegan':
         w = Gen.mapping(z)
         x = Gen.synthesis(w)
      x = tf.image.resize(x, [224, 224], tf.image.ResizeMethod.NEAREST_NEIGHBOR)
      x = tf.clip_by_value((x+1)/2, clip_value_min=0., clip_value_max=1.)
      ys = []

      for Cls in cls_lst:
         y = Cls(x, is_training=False)
         y = tf.math.argmax(y, axis=1)
         y = tf.expand_dims(y, axis=1)
         ys.append(y)
         
      if model_name == 'stylegan':
         w_data.append(tf.squeeze(w[:,0,:]))
      
      y = tf.concat(ys, axis=1)
      y_data.append(y)
   
   w_data = tf.concat( w_data, axis=0 )
   y_data = tf.concat( y_data, axis=0 )
   
   
   print(tf.reduce_sum(y_data, axis=0)/(num_data*10))
   
   
   out_dir = CUR_DIR+'/output/'+model_name+'/data/'
   np.save(out_dir+'w_data.npy', w_data.numpy())
   np.save(out_dir+'y_data.npy', y_data.numpy())
   
   print(w_data.shape)
   print(y_data.shape)
#











