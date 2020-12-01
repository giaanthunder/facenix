import os, sys, math, time, shutil
import numpy as np


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import logging
logger = tf.get_logger()
logger.setLevel(logging.ERROR)

from tensorflow import keras
from tensorflow.keras import layers




att_dict = {
    '5_o_Clock_Shadow'   : 0 , 'Arched_Eyebrows'  : 1 , 'Attractive'     : 2 ,
    'Bags_Under_Eyes'    : 3 , 'Bald'             : 4 , 'Bangs'          : 5 ,
    'Big_Lips'           : 6 , 'Big_Nose'         : 7 , 'Black_Hair'     : 8 ,
    'Blond_Hair'         : 9 , 'Blurry'           : 10, 'Brown_Hair'     : 11,
    'Bushy_Eyebrows'     : 12, 'Chubby'           : 13, 'Double_Chin'    : 14,
    'Eyeglasses'         : 15, 'Goatee'           : 16, 'Gray_Hair'      : 17,
    'Heavy_Makeup'       : 18, 'High_Cheekbones'  : 19, 'Male'           : 20,
    'Mouth_Slightly_Open': 21, 'Mustache'         : 22, 'Narrow_Eyes'    : 23,
    'No_Beard'           : 24, 'Oval_Face'        : 25, 'Pale_Skin'      : 26,
    'Pointy_Nose'        : 27, 'Receding_Hairline': 28, 'Rosy_Cheeks'    : 29,
    'Sideburns'          : 30, 'Smiling'          : 31, 'Straight_Hair'  : 32,
    'Wavy_Hair'          : 33, 'Wearing_Earrings' : 34, 'Wearing_Hat'    : 35,
    'Wearing_Lipstick'   : 36, 'Wearing_Necklace' : 37, 'Wearing_Necktie': 38,
    'Young'              : 39
}


def create_model(name):
    if name == 'nas':
        inputs = tf.keras.Input(shape=(4032,))
    if name == 'res':
        inputs = tf.keras.Input(shape=(2048,))
    x = layers.Dense(4096, activation=tf.nn.relu)(inputs)
    x = layers.Dropout(0.1)(x)
    x = layers.Dense(4096, activation=tf.nn.relu)(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(1024, activation=tf.nn.relu)(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(512, activation=tf.nn.relu)(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(256, activation=tf.nn.relu)(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(128, activation=tf.nn.relu)(x)
    outputs = layers.Dense(2, activation=tf.nn.softmax)(x)
    c_model = tf.keras.Model(inputs=inputs, outputs=outputs)
    c_model.compile(
        optimizer=keras.optimizers.Adam(),
        loss=keras.losses.SparseCategoricalCrossentropy(),
        metrics=[keras.metrics.SparseCategoricalAccuracy()],
    )
    return c_model


class Extractor():
    def __init__(self, name):
        if name == 'nas':
            from tensorflow.keras.applications.nasnet import NASNetLarge, preprocess_input
            self.model = NASNetLarge(include_top=False, weights='imagenet', pooling='avg')
            self.size = 331
        if name == 'res':
            from tensorflow.keras.applications.resnet import ResNet50, preprocess_input
            self.model = ResNet50(include_top=False, weights='imagenet', pooling='avg')
            self.size = 224
        self.proc = preprocess_input
        self.model.trainable = False

    def load_img(self, image_path):
        img = tf.io.read_file(image_path)
        img = tf.io.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, [self.size, self.size], tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        img = tf.cast(img, tf.float32)
        img = tf.expand_dims(img,axis=0)
        img = self.proc(img)
        return img

    def get_feature2(self, img):
        img = tf.image.resize(img, [self.size, self.size], tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        img = tf.cast(img, tf.float32)
        img = self.proc(img)
        f1 = self.model(img)
        f1 = tf.math.l2_normalize(f1)
        return f1

    def get_feature(self, path):
        img = self.load_img(path)
        f1 = self.model(img)
        f1 = tf.math.l2_normalize(f1)
        return f1

def make_data(img_dir, bb_name):
    xtr = Extractor(bb_name)
    n = 30000
    x = []
    for i in range(n):
        print(i, '/', n)
        path = os.path.join(img_dir, "%d.jpg"%i)
        f1 = xtr.get_feature(path)
        x.append(f1)
    x = tf.concat(x, axis=0)
    return x

def train_att(att,anno_path,x,bb_name,model_dir):
    att_id = att_dict[att]+1
    y = np.loadtxt(anno_path, skiprows=2, usecols=att_id, dtype=np.int64)
    y = (y + 1) / 2

    pos_i = np.where(y==1)[0]
    neg_i = np.where(y==0)[0]

    if len(pos_i) > len(neg_i):
        np.random.shuffle(pos_i)
        pos_i = pos_i[:len(neg_i)]
    else:
        np.random.shuffle(neg_i)
        neg_i = neg_i[:len(pos_i)]


    idx = np.concatenate([pos_i,neg_i], axis=0)
    np.random.shuffle(idx)

    x_train = x[idx]
    y_train = y[idx]

    c_model = create_model(bb_name)
    bs = 128 if len(idx) > 256 else len(idx)
    history = c_model.fit(
        x_train, y_train,
        batch_size=bs, epochs=30,
        # validation_split=0.2,
        verbose=0
    )
    acc = history.history['sparse_categorical_accuracy'] # val_sparse_categorical_accuracy

    path = os.path.join(model_dir,att)
    c_model.save(path)
    return (acc[-1], len(idx))


