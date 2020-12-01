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

import numpy as np
import cv2
from PIL import Image

root_dir = os.path.abspath('..')
if root_dir not in sys.path:
    sys.path.append(root_dir)
from facenix_utils import crop_align_face, rotate_img, FaceDetector, FaceParser, rand_name, draw_text, make_vid

import stgan


class StganGUI():
    def __init__(self):
        self.Enc, self.Gen, self.Stu = stgan.models.pretrained_models()
        self.atts = [
            'Bald', 'Bangs', 'Black_Hair', 
            'Blond_Hair', 'Brown_Hair', 'Bushy_Eyebrows', 
            'Eyeglasses', 'Male', 'Mouth_Slightly_Open', 
            'Mustache', 'No_Beard', 'Pale_Skin', 
            'Young'
        ]
        self.detector = FaceDetector()
        self.parser   = FaceParser()

    def convert_plain(self, out_dir, vid_path):
        class PlainConverter():
            def __init__(self):
                self.cnt = 0
            def process_frame(self, frame):
                print(self.cnt)
                self.cnt += 1
                return frame

        path, name = rand_name(out_dir)
        path = path + '.webm'
        name = name + '.webm'
        model = PlainConverter()
        make_vid(vid_path, path, model)
        return name


    def att_mod(self, out_dir, value):
        class AttModConverter():
            def __init__(self, inputs):
                self.Enc, self.Gen, self.Stu, self.atts, self.detector, self.parser = inputs

                values = value.split("_")
                val_lst = {"Unchanged":0, "Add":1, "Remove":-1}
                att = []
                for val in values:
                    att.append(val_lst[val])
                self.att = tf.convert_to_tensor([att], dtype=tf.float32)
                self.cnt = 0

            def process_frame(self, frame):
                print(self.cnt)
                self.cnt += 1
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # h1,w1,_ = frame.shape
                # frame = cv2.resize(frame, (w1//3,h1//3))
                faces = self.detector.detect(frame)

                for i in range(len(faces)):
                    bbox , landmarks = faces[i]

                    img_rot, ali_face, crp_box, img_box, center, angle = crop_align_face(frame, landmarks)
                    raw_h,raw_w,_ = ali_face.shape

                    mod_face = stgan.data.preprocess(ali_face, size=128)
                    mod_face = tf.expand_dims(mod_face, axis=0)
                    zs = self.Enc(mod_face)
                    mod_face = self.Gen(self.Stu(zs, self.att), self.att)[0]
                    mod_face = stgan.data.to_img(mod_face).numpy().astype(np.uint8)
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

                    restore_img = cv2.cvtColor(restore_img, cv2.COLOR_RGB2BGR)
                    return restore_img

        vid_path = out_dir + "video.mp4"
        path, name = rand_name(out_dir)
        path = path + '.webm'
        name = name + '.webm'

        inputs = (self.Enc, self.Gen, self.Stu, self.atts, self.detector, self.parser)
        model = AttModConverter(inputs)
        make_vid(vid_path, path, model)
        return name



    def reset(self, out_dir):
        class PlainConverter():
            def process_frame(self, frame):
                return frame

        vid_path = out_dir + "video.mp4"
        path, name = rand_name(out_dir)
        path = path + '.webm'
        name = name + '.webm'
        model = PlainConverter()
        make_vid(vid_path, path, model)
        return name





stgangui = StganGUI()
# ==============