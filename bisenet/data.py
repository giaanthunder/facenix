import os, sys
import numpy as np
import cv2
import tensorflow as tf


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'



class MaskCeleba():
    def __init__(self, data_dir, img_resize, batch_size, part='train'):
        atts = ['skin', 'l_brow', 'r_brow', 'l_eye', 'r_eye', 'eye_g', 'l_ear', 'r_ear', 'ear_r',
                  'nose', 'mouth', 'u_lip', 'l_lip', 'neck', 'neck_l', 'cloth', 'hair', 'hat']
        # data_dir = '/data2/01_luan_van/data/CelebAMask-HQ'
    
        # create list of image_paths
        list_file    = os.path.join(data_dir, 'CelebAMask-HQ-attribute-anno.txt')
        names         = np.loadtxt(list_file, skiprows=2, usecols=[0], dtype=np.str)
        image_paths = [os.path.join(data_dir, 'CelebA-HQ-img', name) for name in names]
        
        # create list of label_paths
        label_paths = []
        for i in range(15):
            for j in range(2000):
                name = "%d/%05d_" % (i, i*2000+j)
                path = os.path.join(data_dir, 'CelebAMask-HQ-mask-anno', name)
                label_paths.append(path)
        
        if part == 'test':
            image_paths = image_paths[26999:29999]
            label_paths = label_paths[26999:29999]
            shuffle          = False
            drop_remainder = False
        if part == 'val':
            image_paths = image_paths[23999:26999]
            label_paths = label_paths[23999:26999]
            shuffle          = False
            drop_remainder = False
        if part == 'train':
            image_paths = image_paths[:23999]
            label_paths = label_paths[:23999]
            shuffle          = True
            drop_remainder = True
        
        self.count = len(image_paths)
        
        
        def load_data():
            for i in range(self.count):
                image_path = image_paths[i]
                label_path = label_paths[i]
                
                img = tf.io.read_file(image_path)
                img = tf.io.decode_jpeg(img, channels=3)
                #img = tf.image.resize(img, [1024,1024])
                #img = tf.cast(img, dtype=tf.uint8)
                
                h, w, c = img.shape
                labels = []
                for att in atts:
                    path = label_path + att + '.png'
                    if os.path.exists(path):
                        label = tf.io.read_file(path)
                        label = tf.io.decode_png(label, channels=1)
                        #label = tf.image.resize(label, [1024,1024], tf.image.ResizeMethod.NEAREST_NEIGHBOR)
                        #label = tf.cast(label, dtype=tf.uint8)
                    else:
                        label = tf.zeros([1024,1024,1], dtype=tf.uint8)
                    labels.append(label)
                label = tf.concat(labels, axis=2)

                if part == 'train':
                    img, label = ColorJitter(img, label)
                    img, label = HorizontalFlip(img, label)
                    img, label = RandomCrop(img, label)
                img, label = Standardize(img, label, img_resize)
                yield img, label

        ds = tf.data.Dataset.from_generator(load_data, output_types=(tf.float32,tf.float32))
        #ds = ds.cache(filename='./ds_cache')
        
        if shuffle:
            ds = ds.shuffle(buffer_size=4096)

        ds        = ds.repeat(-1)
        ds        = ds.batch(batch_size, drop_remainder=drop_remainder)
        self.ds = ds.prefetch(buffer_size=8)

    

# ==============================================================================
# =                                          Utilities                                              =
# ==============================================================================
@tf.function
def ColorJitter(img, label):
    img = tf.image.random_brightness(img, 0.5)
    img = tf.image.random_contrast(img, 0.5, 1.5)
    img = tf.image.random_saturation(img, 0.5, 1.5)
    return img, label

@tf.function
def HorizontalFlip(img, label):
    rand = tf.random.uniform([], 0, 1)
    if rand > 0.5:
        return img, label
    img    = tf.image.flip_left_right(img)
    label = tf.image.flip_left_right(label)
    return img, label
    
@tf.function
def RandomCrop(img, label):
    rand = tf.random.uniform([], 0.7, 1)
    img_h, img_w, _ = img.shape
    new_h, new_w, _ = tf.cast(img.shape * rand, dtype=tf.int32)
    y = tf.random.uniform([], 0, img_h-new_h, dtype=tf.int32)
    x = tf.random.uniform([], 0, img_w-new_w, dtype=tf.int32)

    img = tf.image.crop_to_bounding_box(img, y, x, new_h, new_w)
    label = tf.image.crop_to_bounding_box(img, y, x, new_h, new_w)
    return img, label
    
@tf.function
def Standardize(img, label, size):
    img = preprocess(img, size)
    
    label = tf.image.resize(label, size=[size,size], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    label = tf.cast(label, tf.int32)
    return img, label




def preprocess(img, size):
    # img = tf.image.resize(img, size=[size,size], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    img = tf.image.resize(img, size=[size,size], method=tf.image.ResizeMethod.BILINEAR)
    img = tf.cast(img, tf.float32)
    img = img/255.
    
    h, w, c = img.shape
    mean = tf.convert_to_tensor((0.485, 0.456, 0.406))
    mean = tf.reshape(mean, [1,1,-1])
    mean = tf.tile(mean, [h,w,1])
    std  = tf.convert_to_tensor((0.229, 0.224, 0.225))
    std  = tf.reshape(std, [1,1,-1])
    std  = tf.tile(std, [h,w,1])
    
    img = (img - mean) / std

    return img
    
def load_img(image_path):
    img = tf.io.read_file(image_path)
    img = tf.io.decode_jpeg(img, channels=3)
    return img
    
def to_img(x):
    img = (x + 1.) * 127.5
    img = tf.clip_by_value(img, 0., 255.)
    return img

def to_file(img, img_path):
    img = tf.cast(img, dtype=tf.uint8)
    img = tf.image.encode_jpeg(img)
    tf.io.write_file(img_path, img)
    
def to_img_file(x, img_path):
    img = to_img(x)
    to_file(img, img_path)

def to_mask(img, label):
    h, w, c = img.shape
    
    mask = tf.image.resize(label, size=[h,w], method=tf.image.ResizeMethod.BILINEAR, antialias=True)
    # mask = tf.image.resize(label, size=[h,w], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    mask = tf.argmax(mask, axis=2)
    mask_lst = []
    zeros = tf.zeros(shape=[h,w])
    for i in range(19):
        part = tf.where(mask==i, 1., zeros)
        mask_lst.append(part)
    masks = tf.stack(mask_lst, axis=2)
    return masks

def to_mask2(img, label, smooth=False, f=0.05):
    mask_dict = {'background':  0, 
        'skin' :  1, 'l_brow':  2, 'r_brow':  3, 'l_eye':  4, 'r_eye':  5,  'eye_g':  6, 
        'l_ear':  7, 'r_ear' :  8, 'ear_r' :  9, 'nose' : 10, 'mouth': 11,  'u_lip': 12, 
        'l_lip': 13, 'neck'  : 14, 'neck_l': 15, 'cloth': 16, 'hair' : 17,  'hat'  : 18
    }
    h, w, c = img.shape
    
    label = np.argmax(label, axis=2)
    masks = {}
    for name, i in mask_dict.items():
        part = np.where(label==i, 1., 0.)
        part = part.astype(np.float32)
        part = cv2.resize(part, (w,h), cv2.INTER_NEAREST)
        if smoothing:
            part = smoothing(part, f=f)
        part = np.expand_dims(part, axis=2)
        masks[name]=part
    return masks

def smoothing(mask, f=0.05):
    h, w = mask.shape
    f_size = int(h*f)
    if f_size%2 == 0:
        f_size+=1
    smooth_mask = cv2.GaussianBlur(mask, (f_size, f_size), 0)
    smooth_mask = np.where(smooth_mask>0.5,1,0)
    return smooth_mask


def colorize(img, label):
    h, w, c = label.shape

    img = tf.image.resize(img, size=[h,w], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    img = tf.cast(img, dtype=tf.float32)

    label = tf.transpose(label, [2,0,1])
    label = tf.argmax(label, axis=0)
    label = tf.stack([label,label,label], axis=2)

    seg = tf.zeros(shape=[h,w,3])
    for i in range(19):
        color = tf.random.uniform(shape=[3], minval=0, maxval=255)
        seg = tf.where(label==[i,i,i], color, seg)


    # color = [0,0,255]
    # seg = tf.where(label==[17,17,17], color, img)

    # # # shift mean
    # # hair = tf.where(label==[17,17,17], img, [0,0,0])
    # # shift_val = tf.reduce_mean(hair, axis=[0,1])
    # # mask = tf.where(label==[17,17,17], [1.,1.,1.], [0.,0.,0.])
    # # img = img - (mask*shift_val)

    # # reduce saturation
    # rest = tf.where(label==[17,17,17], [0,0,0], img)
    # hair = tf.where(label==[17,17,17], img, [0,0,0])
    # hair = tf.reduce_mean(hair, axis=2, keepdims=True)
    # hair = tf.tile(hair, multiples=[1,1,3])
    # img = rest + hair

    ratio = 0.4
    out = (1-ratio)*img + ratio*seg

    return out


# ==============================================================================
# =                                             Test                                                  =
# ==============================================================================
if __name__ == '__main__':
    print("Tensorflow ver: ",tf.__version__)
    print("Eager enable: ",tf.executing_eagerly())
    
    data = MaskCeleba(data_dir='/data2/01_luan_van/data/CelebAMask-HQ', img_resize=128, 
        batch_size=32, part='val')
    
    print(data.count)
    
    for _ in range(1):
        image_batch, label_batch = next(iter(data.ds))
        print("image: ", image_batch.shape)
        print("label: ", label_batch.shape)
#











