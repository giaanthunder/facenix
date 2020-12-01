import os, shutil, sys, math, time, argparse

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import logging
logger = tf.get_logger()
logger.disabled = True
logger.setLevel(logging.FATAL)

import numpy as np

import data, models



if __name__ == '__main__':
    Gen = models.pretrained_models()

    tf.random.set_seed(np.random.randint(1 << 31))
    out_dir = 'samples_1/'
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

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


 
