from django.db import models

# Create your models here.

# ==== tensorflow ====
import os, sys, shutil
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import logging
logger = tf.get_logger()
logger.disabled = True
logger.setLevel(logging.FATAL)

import cv2
import numpy as np
from PIL import Image

root_dir = os.path.abspath('..')
if root_dir not in sys.path:
    sys.path.append(root_dir)
from facenix_utils import crop_align_face, rotate_img, FaceDetector, FaceParser, rand_name

from att_vector_finder.vector_finder import get_att_vectors 

name = 'stylegan'
if name == 'stylegan':
    import stylegan
if name == 'stylegan2':
    import stylegan2 as stylegan




# ==============================================================================
# =                                StyleGAN                                    =
# ==============================================================================
class StyleGUI():
    def __init__(self):
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
            'Young' 
        ]

        self.att_mask_dict = {

        }
        
        self.Gen = stylegan.models.pretrained_models()
        self.Enc = stylegan.models.Encoder(self.Gen)

        self.att_vectors = get_att_vectors(name=name)

        self.detector = FaceDetector()
        self.parser   = FaceParser()
        
        
    def find_z(self, out_dir, img_path, opt_ite=500):
        frame = Image.open(img_path).convert('RGB')
        frame = np.array(frame)
        h1,w1,_ = frame.shape
        # frame = cv2.resize(frame, (w1//3,h1//3))
        faces = self.detector.detect(frame)

        for i in range(len(faces)):
            bbox , landmarks = faces[i]

            img_rot, ali_face, crp_box, img_box, center, angle = crop_align_face(frame, landmarks)

            raw_h,raw_w,_ = ali_face.shape

            face = tf.expand_dims(ali_face, axis=0)
            w, _ = self.Enc(face, opt_ite, self.parser)

            np.save(out_dir +"/rs_w%d.npy"%i , w.numpy())
            np.save(out_dir +"/w%d.npy"%i, w.numpy())

            mod_face = self.Gen.synthesis(w)[0]
            mod_face = stylegan.data.to_img(mod_face).numpy().astype(np.uint8)
            mod_face = cv2.resize(mod_face, (raw_w,raw_h))

            masks = self.parser.parse(ali_face,smooth=True, percent=5)
            mask1 = masks['background'] + masks['neck'] + masks['neck_l'] + masks['cloth']
            mask1 = self.parser.blur_edge(mask1)
            mask2 = np.ones(mask1.shape, dtype=np.float32) - mask1
            
            merge_face = ali_face * mask1 + mod_face * mask2

            # rap face, restore
            crp_x1, crp_y1, crp_x2, crp_y2 = crp_box
            img_rot[crp_y1:crp_y2,crp_x1:crp_x2] = merge_face

            restore_img = rotate_img(img_rot, center, -angle)
            img_x1, img_y1, img_x2, img_y2 = img_box
            restore_img = restore_img[img_y1:img_y2,img_x1:img_x2]

            path, name = rand_name(out_dir)
            path = path + '.jpg'
            name = name + '.jpg'
            Image.fromarray(restore_img).save(path)
            return name

    def glass_p(self):
        pass
        
    def att_mod(self, out_dir, att_name, value):
        frame = Image.open(out_dir+"image.jpg").convert('RGB')
        frame = np.array(frame)
        faces = self.detector.detect(frame)

        for i in range(len(faces)):
            bbox , landmarks = faces[i]

            img_rot, ali_face, crp_box, img_box, center, angle = crop_align_face(frame, landmarks)

            raw_h,raw_w,_ = ali_face.shape

            w = np.load(out_dir +"/w%d.npy"%i)
            att = self.att_vectors[att_name] * 0.5
            if value=='add':
                w = w + att
            if value=='minus':
                w = w - att
            np.save(out_dir +"/w%d.npy"%i, w)

            mod_face = self.Gen.synthesis(w)[0]
            mod_face = stylegan.data.to_img(mod_face).numpy().astype(np.uint8)
            mod_face = cv2.resize(mod_face, (raw_w,raw_h))

            masks = self.parser.parse(ali_face,smooth=True, percent=5)
            mask1 = masks['background'] + masks['neck'] + masks['neck_l'] + masks['cloth']
            mask1 = self.parser.blur_edge(mask1)
            mask2 = np.ones(mask1.shape, dtype=np.float32) - mask1

            merge_face = ali_face * mask1 + mod_face * mask2

            # rap face, restore
            crp_x1, crp_y1, crp_x2, crp_y2 = crp_box
            img_rot[crp_y1:crp_y2,crp_x1:crp_x2] = merge_face

            restore_img = rotate_img(img_rot, center, -angle)
            img_x1, img_y1, img_x2, img_y2 = img_box
            restore_img = restore_img[img_y1:img_y2,img_x1:img_x2]

            path, name = rand_name(out_dir)
            path = path + '.jpg'
            name = name + '.jpg'
            Image.fromarray(restore_img).save(path)
            return name

    def reset(self, out_dir):
        frame = Image.open(out_dir+"image.jpg").convert('RGB')
        frame = np.array(frame)
        faces = self.detector.detect(frame)

        for i in range(len(faces)):
            bbox , landmarks = faces[i]

            img_rot, ali_face, crp_box, img_box, center, angle = crop_align_face(frame, landmarks)

            raw_h,raw_w,_ = ali_face.shape

            w = np.load(out_dir +"/rs_w%d.npy"%i)
            np.save(out_dir +"/w%d.npy"%i, w)

            mod_face = self.Gen.synthesis(w)[0]
            mod_face = stylegan.data.to_img(mod_face).numpy()
            mod_face = cv2.resize(mod_face, (raw_w,raw_h))

            masks = self.parser.parse(ali_face,smooth=True, percent=5)
            mask1 = masks['background'] + masks['neck'] + masks['neck_l'] + masks['cloth']
            mask1 = self.parser.blur_edge(mask1)
            mask2 = np.ones(mask1.shape, dtype=np.float32) - mask1

            merge_face = ali_face * mask1 + mod_face * mask2

            # rap face, restore
            crp_x1, crp_y1, crp_x2, crp_y2 = crp_box
            img_rot[crp_y1:crp_y2,crp_x1:crp_x2] = merge_face

            restore_img = rotate_img(img_rot, center, -angle)
            img_x1, img_y1, img_x2, img_y2 = img_box
            restore_img = restore_img[img_y1:img_y2,img_x1:img_x2]

            path, name = rand_name(out_dir)
            path = path + '.jpg'
            name = name + '.jpg'
            Image.fromarray(restore_img).save(path)
            return name



stylegui = StyleGUI()
# ==============