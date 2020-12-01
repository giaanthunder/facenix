import os, sys, random, shutil, threading, math
import numpy as np

from sklearn import svm
import pickle

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
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

def get_att_vectors(name):
    cur_dir = os.path.dirname(os.path.realpath(__file__)) + '/'
    att_vectors = {}
    for att in atts:
        path = cur_dir+'%s/att_vectors/%s.npy'%(name,att)
        try:
            b = np.load(path)
        except:
            print('Cannot load:', path)
            continue
        att_vectors[att] = b
    return att_vectors


if __name__ == '__main__':
    name = 'stylegan'
    cur_dir   = os.path.dirname(os.path.abspath(__file__)) + '/'
    model_dir = cur_dir + '%s_models/'%name
    data_dir  = cur_dir + '%s/data/'%name
    att_dir   = cur_dir + '%s/att_vectors/'%name

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    if not os.path.exists(att_dir):
        os.makedirs(att_dir)

    for i, att in enumerate(atts):
        w = np.load(data_dir+'w_data.npy')
        y = np.load(data_dir+'y_data.npy')[:,i]

        pos_i = np.where(y==1)[0]
        neg_i = np.where(y==0)[0]
        np.random.shuffle(pos_i)
        np.random.shuffle(neg_i)

        if len(pos_i) > len(neg_i):
            pos_i = pos_i[:len(neg_i)]
        else:
            neg_i = neg_i[:len(pos_i)]

        n = len(pos_i)
        p = int(n*0.8)
        trn_idx = np.concatenate([pos_i[:p], neg_i[:p]], axis=0)
        val_idx = np.concatenate([pos_i[p:], neg_i[p:]], axis=0)
        np.random.shuffle(trn_idx)
        np.random.shuffle(val_idx)

        trn_w = w[trn_idx]
        trn_y = y[trn_idx]
        val_w = w[val_idx]
        val_y = y[val_idx]

        clf = svm.LinearSVC()
        
        try:
            svm_model = clf.fit(trn_w, trn_y)
        except Exception as e:
            print(att + ' '*(25-len(att)) + '0.00 (%d)'%n, str(e))
            continue

        val_pred = svm_model.predict(val_w)
        correct = np.where(val_pred==val_y,1,0)
        acc = np.sum(correct)/ val_w.shape[0]
        print(att + ' '*(25-len(att)) + '%.2f'%acc + ' (%d)'%n)

        vector_b = svm_model.coef_
        vector_b = vector_b/np.linalg.norm(vector_b)
        np.save(att_dir + att + '.npy', vector_b)
        pickle.dump(svm_model, open(model_dir+att+'_model.sav', 'wb'))






