import os, sys, math, time
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import logging
logger = tf.get_logger()
logger.setLevel(logging.ERROR)

import util

data_dir  = '/media/anhuynh/DATA/03_task/12_dataset/CelebAMask-HQ/'
img_dir   = data_dir + 'CelebA-HQ-img/'
anno_path = data_dir + 'CelebAMask-HQ-attribute-anno.txt'
model_dir = 'models/'

bb_name = 'nas'
x = util.make_data(img_dir, bb_name)
# np.save('celebahq_nas.npy', x)
# x = np.load('celebahq_nas.npy')

for att, i in util.att_dict.items():
    acc, n = util.train_att(att,anno_path,x,bb_name,model_dir)
    text = att + ' '*(30-len(att)) + '%.2f'%acc + ' (%d)'%n
    print(text)

