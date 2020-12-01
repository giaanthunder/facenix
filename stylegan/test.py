import os, shutil, sys, math, time
import argparse

import numpy as np
import tensorflow as tf

import data, models


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class tester():
   def __init__(self, sample_dir, batch_size):
      self.sample_dir = sample_dir
      self.batch_size = batch_size

   def make_sample(self, model_lst, inputs, cur_res, epoch, batch, phs_i, samp_per_batch=3):
      Gen, Dis  = model_lst
      z, x_real = inputs
      w = Gen.mapping(z)
      w = models.lerf(Gen.w_avg, w, 0.7)
      x_fake = Gen.synthesis(w)

      bs = x_real.shape[0]
      if bs < samp_per_batch:
         samp_per_batch = bs

      for i in range(samp_per_batch):
         save_dir = self.sample_dir+"/sample_%dx%d_%d/"%(cur_res, cur_res, phs_i)
         if not os.path.exists(save_dir):
            os.makedirs(save_dir)
         img_path = save_dir + "%03d_%03d_%d.jpg"%(epoch, batch, i)
         with tf.device('/cpu:0'):
            data.to_img_file(x_fake[i], img_path)
   



if __name__ == '__main__':
   parser = argparse.ArgumentParser()
   parser.add_argument('--experiment_name', dest='experiment_name', default='origin')
   parser.add_argument('--res', dest='res', type=int, default=1024)

   args = parser.parse_args()
   experiment_name = args.experiment_name

   Gen = models.G_style()
   checkpoint_dir = 'output/%s/trained_model' % experiment_name
   checkpoint = tf.train.Checkpoint(Gen=Gen)
   manager = tf.train.CheckpointManager(checkpoint, checkpoint_dir, max_to_keep=3)

   tf.random.set_seed(np.random.randint(1 << 31))
   # ckpt = manager.checkpoints[j]

   # print(j, ckpt)

   ckpt = manager.latest_checkpoint
   if ckpt:
      checkpoint.restore(ckpt)
   else:
      print(ckpt)
      print("RESTORE FAILED")
      exit()

   out_dir = '%s/samples_%d/'%(experiment_name, j)
   if not os.path.exists(out_dir):
      os.makedirs(out_dir)

   Gen.mapping.lod   = int(math.log2(args.res) - 2)
   Gen.synthesis.lod = int(math.log2(args.res) - 2)

   Gen.mapping.alpha   = 0.0
   Gen.synthesis.alpha = 0.0

   Gen.is_training = False

   tik = time.time()
   for i in range(1000):
      print(i)
      # z = tf.random.uniform([1,512],-1., 1.)
      z = tf.random.normal([1,512])
      w = Gen.mapping(z)
      w = models.lerf(Gen.w_avg, w, 0.7)
      img = Gen.synthesis(w)[0]
      # img  = Gen(z)[0]
      path = out_dir + str(i) + '.jpg'
      data.to_img_file(img, path)
   tok = time.time()
   print(tok-tik)



 