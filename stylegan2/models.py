import os, sys, time, math

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import logging
logger = tf.get_logger()
logger.disabled = True
logger.setLevel(logging.FATAL)

from tensorflow.python import keras
from tensorflow.python.keras import layers

import numpy as np
from PIL import Image

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/'

# ==============================================================================
# =                                   Encoder                                  =
# ==============================================================================
class Encoder(keras.Model):
    def __init__(self, Gen):
        super(Encoder, self).__init__()
        vgg16          = keras.applications.vgg16.VGG16(weights='imagenet', include_top=False)
        self.extractor = keras.Model(inputs=vgg16.input, outputs=vgg16.get_layer('block3_pool').output)
        self.flat      = layers.Flatten()
        self.Gen       = Gen
        self.ar        = 0.3
        
    def call(self, x_real, opt_ite=500, parser=None):
        masks = parser.parse(x_real[0],smooth=True, percent=5)
        mask  = masks['background'] + masks['neck'] + masks['neck_l'] + masks['cloth']

        masked = True
        if not masked:
            mask = np.zeros(mask.shape, dtype=np.float32)

        mask1 = np.expand_dims(mask, axis=0)
        mask2 = np.ones(mask1.shape, dtype=np.float32) - mask1
        mask2 = tf.image.resize(mask2, [224, 224], tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        mask2 = tf.cast(mask2, dtype=tf.float32)

        x_real = tf.cast(x_real, dtype=tf.float32)
        x_real = x_real/255.
        x_real = tf.image.resize(x_real, [224, 224], tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        x_real = x_real * mask2 #
        f_real = self.extractor(x_real)
        f_real = self.flat(f_real)

        lr = tf.Variable(1.)

        n_try  = 100
        z_fake = tf.constant(tf.random.normal(shape=[n_try, 512]))
        loss = []
        for i in range(n_try//10):
            w = self.Gen.mapping(z_fake[i*10:(i+1)*10], brc=True)
            l = self.loss_calc(w, f_real, mask2)
            loss.append(l)
        loss = tf.concat(loss, axis=0)
        loss_best = tf.argsort(loss)[:18]
        z_best = tf.gather(z_fake, loss_best)
        z_best = tf.reverse(z_best, axis=[0])
        w_best = self.Gen.mapping(z_best, brc=False)

        expand_space = False
        if expand_space:
            w_best = tf.Variable(w_best)
        else:
            w_best = w_best[-1]
            w_best = tf.reshape(w_best,[1,512])
            w_best = tf.Variable(w_best)

        count = 0
        min_loss = 1000000.
        loop = tf.function(self.loop)
        tik = time.time()
        for i in range(opt_ite):
            loss, w = loop(w_best, f_real, lr, mask2)
            if i%500==0 and i!=0:
                lr.assign(lr/3.)

            if loss < min_loss:
                min_loss = loss
                min_w = w
                min_i = i
            print('loop: %d, loss: %f, min loss: %f from loop %d'%(i, loss.numpy(), min_loss.numpy(), min_i))
        
        tok = time.time()
        print('EXECUTING TIME: %2f'%(tok-tik))
        return min_w, min_loss
    
    def loss_calc(self, w, f_real, mask2):
        w      = lerf(self.Gen.w_avg, w, self.ar)
        x_fake = (self.Gen.synthesis(w) + 1.)/2.
        x_fake = tf.image.resize(x_fake, [224, 224], tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        x_fake = x_fake * mask2 #
        f_fake = self.extractor(x_fake)
        f_fake = self.flat(f_fake)

        loss = tf.losses.mean_squared_error(y_pred=f_fake, y_true=f_real)
        return loss


    def loop(self, w_fake, f_real, lr, mask2):
        n = 18 // w_fake.shape[0]
        with tf.GradientTape() as tape:
            w     = tf.tile(w_fake, multiples=[n,1])
            w     = tf.expand_dims(w, axis=0)
            loss = self.loss_calc(w, f_real, mask2)

        vars = [w_fake]
        grad = tape.gradient(loss, vars)
        w_fake.assign_sub(grad[0]*lr)

        w = lerf(self.Gen.w_avg, w, self.ar)

        return loss, w



# ==============================================================================
# =                                 Generator                                  =
# ==============================================================================
class G_style(keras.Model):
    def __init__(self, g_mapping_path, g_synthesis_path, w_avg_path):
        super(G_style, self).__init__()
        w_avg = np.load(w_avg_path)
        self.w_avg     = tf.convert_to_tensor(w_avg)
        self.mapping   = G_mapping(g_mapping_path)
        self.synthesis = G_synthesis(g_synthesis_path)
        
    def call(self, z):
        w = self.mapping(z)
        w = lerf(self.w_avg, w, 0.7)
        img = self.synthesis(w)
        return img

class G_mapping(layers.Layer):
    def __init__(self, g_mapping_path):
        super(G_mapping, self).__init__()
        with tf.device('/device:CPU:0'):
            g_mapping  = tf.saved_model.load(g_mapping_path)
            self.model = g_mapping.signatures['serving_default']

    def call(self, z, brc=True):
        with tf.device('/device:CPU:0'):
            w = self.model(z)['out']
            if brc:
                w = tf.expand_dims(w, axis=1)
                w = tf.tile(w, [1, 18, 1])
            return w

class G_synthesis(layers.Layer):
    def __init__(self, g_synthesis_path):
        super(G_synthesis, self).__init__()
        g_synthesis = tf.saved_model.load(g_synthesis_path)
        self.model  = g_synthesis.signatures['serving_default']

    def call(self, w):
        img = self.model(w)['out']
        img = to_nhwc(img)
        return img

# ==============================================================================
# =                                 Utilities                                  =
# ==============================================================================
def lerf(a, b, t):
    y = a + (b - a) * t
    return y
    
def to_img(x):
    img = (x + 1.) * 127.5
    img = tf.clip_by_value(img, 0., 255.)
    #img = tf.cast(img, dtype=tf.uint8)
    return img

# NCHW to NHWC
def to_nhwc(x):
    x = tf.transpose(x, [0, 2, 3, 1])
    return x

# NHWC to NCHW
def to_nchw(x):
    x = tf.transpose(x, [0, 3, 1, 2])
    return x

def pretrained_models():
    g_mapping_path   = BASE_DIR+'stylegan2/stylegan2_models/g_mapping'
    g_synthesis_path = BASE_DIR+'stylegan2/stylegan2_models/g_synthesis'
    w_avg_path = BASE_DIR+'stylegan2/stylegan2_models/w_avg.npy'
    Gen = G_style(g_mapping_path, g_synthesis_path, w_avg_path)
    return Gen
    