import argparse
import datetime
import json
import traceback

import numpy as np
import tensorflow as tf

import data, models, loss, test

import os, shutil, sys, math, time
from PIL import Image

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import logging
logger = tf.get_logger()
logger.setLevel(logging.ERROR)


class tester():
   def __init__(self, sample_dir, batch_size, z_dim=512):
      self.sample_dir = sample_dir
      self.batch_size = batch_size
      self.z_dim      = z_dim

   def make_sample(self, model_lst, inputs, cur_res, epoch, batch, samp_per_batch=3, att_name="fake"):
      x_real, x_fake, x_rec = self.create_sample_data(model_lst, inputs)
      
      for i in range(samp_per_batch):
         img_name = self.sample_dir+"/sample_%dx%d_%03d_%03d_%d_"%(
               cur_res, cur_res,epoch, batch, i)
         # real
         im = self.convert_to_img(x_real[i])
         im.save(img_name+"real.jpg")
         
         # fake
         im = self.convert_to_img(x_fake[i])
         im.save(img_name+att_name+".jpg")
         
         # real recovery
         im = self.convert_to_img(x_rec[i])
         im.save(img_name+"rec.jpg")

   def create_sample_data(self, model_lst, inputs):
      Gen, Dis, Enc, Stu = model_lst
      x_real, a, b = inputs
      b_diff = b - a
      a_diff = a - a

      z_real     = Enc(x_real)
      z_fake_stu = Stu(z_real, b_diff)
      x_fake     = Gen(z_fake_stu, b_diff)
      z_real_stu = Stu(z_real, a_diff)
      x_rec      = Gen(z_real_stu, a_diff)
      return (x_real, x_fake, x_rec)

   def convert_to_img(self, x):
      img = ( (x.numpy()+1)*127.5 ).astype('uint8')
      return Image.fromarray(img)
   
if __name__ == '__main__':
   # ==============================================================================
   # =                                    param                                   =
   # ==============================================================================

   parser = argparse.ArgumentParser()
   # model
   att_default = ['Bald', 'Bangs', 'Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Bushy_Eyebrows', 'Eyeglasses', 'Male', 'Mouth_Slightly_Open', 'Mustache', 'No_Beard', 'Pale_Skin', 'Young']
   parser.add_argument('--atts', dest='atts', default=att_default, choices=data.att_dict.keys(), nargs='+', help='attributes to learn')
   parser.add_argument('--img_size', dest='img_size', type=int, default=128)
   parser.add_argument('--batch_size', dest='batch_size', type=int, default=1)
   parser.add_argument('--experiment_name', dest='experiment_name', default='stgan_128')

   args = parser.parse_args()
   # model
   atts = args.atts
   n_att = len(atts)
   img_size = args.img_size
   batch_size = args.batch_size
   experiment_name = args.experiment_name
 
   Gen = models.Generator()
   Dis = models.Discriminator(n_att)
   Enc = models.Encoder()
   Stu = models.Stu()
   
   x = tf.ones(shape=[2,128,128,3], dtype=tf.float32)
   a = tf.ones(shape=[2,13], dtype=tf.float32)

   z      = Enc(x)
   z_stu  = Stu(z, a)
   x_fake = Gen(z_stu, a-a)
   d, att = Dis(x)
   
   
   lr    = tf.Variable(initial_value=0., trainable=False)
   g_opt = tf.optimizers.Adam(lr, beta_1 =0., beta_2=0.99)
   d_opt = tf.optimizers.Adam(lr, beta_1 =0., beta_2=0.99)
   params= tf.Variable(initial_value=[5, 0], trainable=False, dtype=tf.int64)
 
   sample_dir = './output/%s/%s' % (experiment_name,experiment_name+"_test")
   if os.path.exists(sample_dir):
      shutil.rmtree(sample_dir)
   os.makedirs(sample_dir, exist_ok=True)
   tester = test.tester(sample_dir, batch_size)
 
   checkpoint_dir = './output/%s/trained_model' % experiment_name
   checkpoint = tf.train.Checkpoint(
      params=params,
      d_opt=d_opt, g_opt=g_opt,
      Gen=Gen, Dis=Dis, Enc=Enc, Stu=Stu 
   )

   manager = tf.train.CheckpointManager(checkpoint, checkpoint_dir, max_to_keep=3)
   checkpoint.restore(manager.latest_checkpoint)
   if manager.latest_checkpoint:
      print("Restore success from:")
      print(manager.latest_checkpoint)
   else:
      print("Restore fail")
   
   test_count, test_data = data.img_ds(data_dir='/data2/01_luan_van/data/', 
      atts=atts, img_resize=img_size, batch_size=batch_size, part='test')
   test_ite = iter(test_data)

   it_per_epoch = test_count//batch_size
   for it in range(it_per_epoch):
      model_lst = (Gen, Dis, Enc, Stu)
      x_real, a = next(test_ite)
      #np.save("/data2/01_luan_van/img_test.npy", x_real.numpy())
      x_real = tf.convert_to_tensor(np.load("/data2/01_luan_van/img_test.npy"))
      a = (a * 2 - 1) * 0.5
      
      img = tf.io.read_file("/data2/01_luan_van/10_stgan/normal.jpg")
      img = tf.io.decode_jpeg(img, channels=3)
      img = tf.cast(img, tf.float32)
      img = tf.clip_by_value(img, 0, 255) / 127.5 - 1
      
      normal = tf.reshape(img, shape=[1,128,128,3])
      
      img = tf.io.read_file("/data2/01_luan_van/10_stgan/pale_skin.jpg")
      img = tf.io.decode_jpeg(img, channels=3)
      img = tf.cast(img, tf.float32)
      img = tf.clip_by_value(img, 0, 255) / 127.5 - 1
      
      pale = tf.reshape(img, shape=[1,128,128,3])
      
      normal_zs = Enc(normal)
      pale_zs   = Enc(pale)
      
      fake_zs = normal_zs[:-1]
      fake_zs.append(pale_zs[-1])      
      
      x_fake    = Gen(Stu(fake_zs, a-a), a-a)
      x_rec     = Gen(Stu(normal_zs, a-a), a-a)
      
      img = ( (x_fake.numpy()[0]+1)*127.5 ).astype('uint8')
      img = Image.fromarray(img)
      img.save("./fake.jpg")
      
      img = ( (x_rec.numpy()[0]+1)*127.5 ).astype('uint8')
      img = Image.fromarray(img)
      img.save("./rec.jpg")
      
      
      #for i in range(n_att):
      #   mask_tmp = [1. for i in range(n_att)]
      #   mask_tmp[i] = -1.
      #   b = tf.convert_to_tensor(mask_tmp) * a
      #   inputs = [x_real, a, b]
      #   tester.make_sample(model_lst, inputs, 128, 0, it, batch_size, atts[i])
      #print("EPOCH %d/%d" % (it, it_per_epoch))

 #
 
 
 
 
 
 