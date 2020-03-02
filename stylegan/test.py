import argparse
import datetime
import json
import traceback

import numpy as np
import tensorflow as tf

import data

import os, shutil, sys, math, time

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class tester():
   def __init__(self, sample_dir, batch_size):
      self.sample_dir = sample_dir
      self.batch_size = batch_size

   def make_sample(self, model_lst, inputs, cur_res, epoch, batch, samp_per_batch=3):
      Gen, Dis  = model_lst
      z, x_real = inputs
      x_fake = Gen(z)


      for i in range(samp_per_batch):
         img_path = self.sample_dir+"/sample_%dx%d_%03d_%03d_%d.jpg"%(
               cur_res, cur_res,epoch, batch, i)
         with tf.device('/cpu:0'):
            data.to_img_file(x_fake[i], img_path)
   
if __name__ == '__main__':
   print('Not implement')

 
 
 
 
 