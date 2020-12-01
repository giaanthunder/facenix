import os, sys, math, time, shutil
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import logging
logger = tf.get_logger()
logger.setLevel(logging.ERROR)

sys.path.append(os.path.abspath('..'))

from classifier.util import Extractor, att_dict



import stylegan as gan
name = 'stylegan'

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/'
cur_dir = os.path.dirname(os.path.abspath(__file__)) + '/'
out_dir = '%s/data/'%name
cls_dir = BASE_DIR + 'classifier/models/'

if not os.path.exists(out_dir):
    os.makedirs(out_dir)


num_data = 10000 # x10


Gen = gan.models.pretrained_models()
f_model = Extractor('nas')


w_data = []
f_data = []
for i in range(num_data):
    print('%d/%d'%(i+1,num_data))
    # z = tf.random.uniform([10,512], -1., 1.)
    z = tf.random.normal([10,512])
    w = Gen.mapping(z)
    w = gan.models.lerf(Gen.w_avg, w, 0.7)
    img = Gen.synthesis(w)
    img = gan.data.to_img(img)
    f1 = f_model.get_feature2(img)

    w = tf.squeeze(w[:,0,:])
    w_data.append(w)
    f_data.append(f1)

w_data = tf.concat(w_data, axis=0)
f_data = tf.concat(f_data, axis=0).numpy()



np.save(out_dir+'f_data.npy',f_data)
np.save(out_dir+'w_data.npy',w_data)
# f_data = np.load(out_dir+'f_data.npy')
# w_data = np.load(out_dir+'w_data.npy')


n = f_data.shape[0]
bs = 1000
y_data = []
for att, i in att_dict.items():
    print(i, att)
    path = cls_dir + att
    c_model = tf.keras.models.load_model(path)
    ys = []
    for i in range(n//bs):
        p1 = i*bs
        p2 = (i+1)*bs
        f1 = f_data[p1:p2]
        y = c_model.predict(f1)
        y = tf.math.argmax(y, axis=1)
        # (bs)
        ys.append(y)
    del c_model
                    
    y = tf.concat(ys, axis=0)
    # (n)
    y_data.append(y)

y_data = tf.stack( y_data, axis=1 )
# (n, num_att)
print(y_data.shape)
print(tf.reduce_sum(y_data, axis=0)/n)

np.save(out_dir+'y_data.npy', y_data.numpy())












