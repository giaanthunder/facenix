import os, sys, time, math, random

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import logging
logger = tf.get_logger()
logger.disabled = True
logger.setLevel(logging.FATAL)


import numpy as np
import cv2
from PIL import Image, ImageOps, ImageSequence
import matplotlib.pyplot as plt


cur_dir = os.path.abspath(os.path.dirname(__file__)) + '/'
# sys.path.append(os.path.abspath(cur_dir))
sys.path.append(os.path.abspath(cur_dir+'retinaface_tf2'))
from retinaface_tf2.modules.models import RetinaFaceModel
from retinaface_tf2.modules.utils import load_yaml, draw_bbox_landm, pad_input_image, recover_pad_output
import bisenet


def ls(in_dir):
    img_paths = []
    path1s = os.listdir(in_dir)
    for path1 in path1s:
        path = os.path.join(in_dir,path1)
        if os.path.isfile(path):
            img_paths.append(path)
            continue

        path2s = os.listdir(path)
        for path2 in path2s:
            path = os.path.join(in_dir,path1,path2)
            if os.path.isfile(path):
                img_paths.append(path)
                continue

            path3s = os.listdir(path)
            for path3 in path3s:
                path = os.path.join(in_dir,path1,path2,path3)
                if os.path.isfile(path):
                    img_paths.append(path)
                else:
                    print("It's a dir:", path)

    img_paths.sort()
    return img_paths

def rand_str(length=10):
    letters = 'abcdefghijkmnopqrstuvxyz0123456789'
    rand_str = ''
    for i in range(length):
        rand_str+=random.choice(letters)
    return rand_str

def rand_name(dir_path):
    name = rand_str()
    path = os.path.join(dir_path, name)
    while os.path.exists(path):
        name = rand_str()
        path = os.path.join(dir_path, name)
    return (path, name)

def angle_2_vec(x1,y1,x2,y2):
    angle = np.arctan2( x1*y2-y1*x2, x1*x2+y1*y2 )
    angle = np.degrees(angle)
    return angle

def distance(x1,y1,x2,y2):
    w = x2 - x1
    h = y2 - y1
    d = math.sqrt(w*w+h*h)
    return d

def rotate_img(img, center, angle):
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    img_rot = cv2.warpAffine(img, M, (h, w))
    return img_rot

def crop_align_face(img, lm):
    # Calculate auxiliary vectors.
    eye_left     = lm[0]
    eye_right    = lm[1]
    eye_avg      = (eye_left + eye_right) * 0.5
    eye_to_eye   = eye_right - eye_left
    mouth_left   = lm[3]
    mouth_right  = lm[4]
    mouth_avg    = (mouth_left + mouth_right) * 0.5
    eye_to_mouth = mouth_avg - eye_avg

    # Choose oriented crop rectangle. flipud: flip arr up to down. hypot: canh huyen
    x = eye_to_eye - np.flipud(eye_to_mouth) * [-1, 1]
    x /= np.hypot(*x)
    x *= max(np.hypot(*eye_to_eye) * 2.0, np.hypot(*eye_to_mouth) * 1.8)
    y = np.flipud(x) * [-1, 1]
    c = eye_avg + eye_to_mouth * 0.1
    # quad = [top_left, bot_left, bot_right, top_right]
    quad = np.stack([c - x - y, c - x + y, c + x + y, c + x - y])
    quad = (quad + 0.5).astype(np.int)
    qsize = int(np.hypot(*x) * 2)

    x1, y1, x2, y2, x3, y3, x4, y4 = quad.flatten()


    # Center of rect
    cx = int((x1+x2+x3+x4)/4 + 0.5)
    cy = int((y1+y2+y3+y4)/4 + 0.5)

    # Rotating angle
    a1, b1 = (x1-x2, y2-y1)
    a2, b2 = (0., 1.)
    angle = angle_2_vec(a1, b1, a2, b2)

    
    # max distance from center to 4 corners of img
    img_h, img_w = img.shape[:2]
    d1 = distance(cx,cy,0,0)
    d2 = distance(cx,cy,0,img_h)
    d3 = distance(cx,cy,img_w,img_h)
    d4 = distance(cx,cy,img_w,0)
    max_d = int(max(d1,d2,d3,d4))

    # pad width of up, down, left, right. Center is crop rect
    u_pad = max_d - cy
    d_pad = max_d - (img_h - cy)
    l_pad = max_d - cx
    r_pad = max_d - (img_w - cx)

    new_cx = max_d
    new_cy = max_d
    new_center = (new_cx,new_cy)

    # padding
    img_pad = np.pad(img,((u_pad,d_pad),(l_pad,r_pad),(0,0)))

    # rotate img
    img_rot = rotate_img(img_pad, new_center, angle)

    # crop face rect
    crp_x1 = new_cx - qsize//2
    crp_y1 = new_cy - qsize//2
    crp_x2 = crp_x1 + qsize
    crp_y2 = crp_y1 + qsize
    ali_face = img_rot[crp_y1:crp_y2,crp_x1:crp_x2]

    crp_box = np.array([crp_x1,crp_y1,crp_x2,crp_y2])
    img_x1  = l_pad
    img_y1  = u_pad
    img_x2  = l_pad+img_w
    img_y2  = u_pad+img_h
    img_box = np.array([img_x1,img_y1,img_x2,img_y2])
    # new_center = np.array(new_center)

    return img_rot, ali_face, crp_box, img_box, new_center, angle



class FaceDetector():
    def __init__(self):
        # set_memory_growth()

        cfg = load_yaml(cur_dir+'retinaface_tf2/configs/retinaface_res50.yaml')
    
        model = RetinaFaceModel(cfg, training=False, iou_th=0.4, score_th=0.5)

        # load checkpoint
        checkpoint_dir = cur_dir + 'retinaface_tf2/checkpoints/' + cfg['sub_name']
        checkpoint = tf.train.Checkpoint(model=model)
        if tf.train.latest_checkpoint(checkpoint_dir):
            checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
        else:
            print('No checkpoint')
            exit()

        self.model = model
        self.cfg   = cfg


    def detect(self, img_raw):
        img_height_raw, img_width_raw, _ = img_raw.shape
        img = np.float32(img_raw.copy())

        img, pad_params = pad_input_image(img, max_steps=max(self.cfg['steps']))

        outputs = self.model(img[np.newaxis, ...]).numpy()
        outputs = recover_pad_output(outputs, pad_params)

        faces = []

        for output in outputs:
            # bbox
            x1 = int(output[0] * img_width_raw)
            y1 = int(output[1] * img_height_raw)
            x2 = int(output[2] * img_width_raw)
            y2 = int(output[3] * img_height_raw)
            bbox = np.array([x1,y1,x2,y2])

            # confidence
            score = output[15]

            # landmark
            l_eye_x   = int(output[ 4] * img_width_raw)
            l_eye_y   = int(output[ 5] * img_height_raw)
            r_eye_x   = int(output[ 6] * img_width_raw)
            r_eye_y   = int(output[ 7] * img_height_raw)
            nose_x    = int(output[ 8] * img_width_raw)
            nose_y    = int(output[ 9] * img_height_raw)
            l_mouth_x = int(output[10] * img_width_raw)
            l_mouth_y = int(output[11] * img_height_raw)
            r_mouth_x = int(output[12] * img_width_raw)
            r_mouth_y = int(output[13] * img_height_raw)
            landmarks = np.array([
                [l_eye_x, l_eye_y], 
                [r_eye_x, r_eye_y], 
                [nose_x, nose_y], 
                [l_mouth_x, l_mouth_y], 
                [r_mouth_x, r_mouth_y]
            ])

            faces.append([bbox, landmarks])

        return faces


class FaceParser():
    def __init__(self):
        import bisenet
        self.bnet = bisenet.models.pretrained_models()
        self.mask_dict = {
            'background': 0, 
            'skin' :  1, 'l_brow':  2, 'r_brow':  3, 'l_eye':  4, 'r_eye':  5,  'eye_g':  6, 
            'l_ear':  7, 'r_ear' :  8, 'ear_r' :  9, 'nose' : 10, 'mouth': 11,  'u_lip': 12, 
            'l_lip': 13, 'neck'  : 14, 'neck_l': 15, 'cloth': 16, 'hair' : 17,  'hat'  : 18
        }

    def parse(self, img, smooth=False, percent=10):
        import bisenet
        img = tf.convert_to_tensor(img)
        img_in = bisenet.data.preprocess(img, size=512)
        img_in = tf.expand_dims(img_in, axis=0)
        out, out16, out32 = self.bnet(img_in)
        label = out[0].numpy()
        masks = bisenet.data.to_mask2(img, label, smooth, percent)
        return masks

    def segment(self, img, masks, mask_names):
        black = np.zeros(img.shape, img.dtype)
        mask  = np.zeros(masks['skin'].shape, masks['skin'].dtype)
        for name in mask_names:
            mask += masks[name]
        seg_img = np.where(mask>0,img,black)
        return seg_img

    def blur_edge(self, mask, percent=10):
        import bisenet
        return bisenet.data.blur_edge(mask,percent)

def draw_box(image, box, color=(0,255,0),thickness=2):
    x, y ,w, h = box
    x_end = x+w
    y_end = y+h
    cv2.rectangle(image, pt1=(x,y), pt2=(x_end,y_end), color=color, thickness=thickness)

def draw_text(image, text, pos=(20,20), fontScale=0.7, color=(240, 31, 31)):
    cv2.putText(image, text, org=pos, fontFace=cv2.FONT_HERSHEY_SIMPLEX , fontScale=fontScale, color=(0,0,0), thickness=4, lineType=cv2.LINE_AA)
    cv2.putText(image, text, org=pos, fontFace=cv2.FONT_HERSHEY_SIMPLEX , fontScale=fontScale, color=color, thickness=2, lineType=cv2.LINE_AA)

def make_vid(in_video_path, out_video_path, model, start=0, duration=100000):
    cap = cv2.VideoCapture(in_video_path)
    frame_width  = int(cap.get(3))
    frame_height = int(cap.get(4))
    # fourcc  = cv2.VideoWriter_fourcc('M','J','P','G')
    # fourcc  = cv2.VideoWriter_fourcc(*'mp4v')
    fourcc  = cv2.VideoWriter_fourcc(*'vp80')
    out_vid = cv2.VideoWriter(out_video_path, fourcc, 30, (frame_width,frame_height))

    cnt = 0
    stop = start + (30*duration)

    while(True):
        ret, frame = cap.read()

        if ret == False:
            break

        if cnt >= start and cnt < stop:
            img = model.process_frame(frame)
            out_vid.write(img)
            # cv2.imshow('frame',img)
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     break
        cnt+=1

    cap.release()
    out_vid.release()
    cv2.destroyAllWindows()
