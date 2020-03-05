import argparse
import datetime
import json
import traceback

import numpy as np

import os, shutil, sys, math, time

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# ==============================================================================
# =                                    param                                   =
# ==============================================================================

parser = argparse.ArgumentParser()
# model
parser.add_argument('--img_size', dest='img_size', type=int, default=224)
# training
parser.add_argument('--epoch', dest='epoch', type=int, default=10, help='# of epochs')
parser.add_argument('--batch_size', dest='batch_size', type=int, default=8)
parser.add_argument('--lr', dest='lr', type=float, default=0.01, help='learning rate')
parser.add_argument('--gpu', dest='gpu', type=str, default='0', help='# of gpu computing in paralel')
parser.add_argument('--att', dest='att', type=str, default='Male', help='''Choose 1 from list below:
   '5_o_Clock_Shadow'   , 'Arched_Eyebrows'  , 'Attractive'     ,
   'Bags_Under_Eyes'    , 'Bald'             , 'Bangs'          ,
   'Big_Lips'           , 'Big_Nose'         , 'Black_Hair'     ,
   'Blond_Hair'         , 'Blurry'           , 'Brown_Hair'     ,
   'Bushy_Eyebrows'     , 'Chubby'           , 'Double_Chin'    ,
   'Eyeglasses'         , 'Goatee'           , 'Gray_Hair'      ,
   'Heavy_Makeup'       , 'High_Cheekbones'  , 'Male'           ,
   'Mouth_Slightly_Open', 'Mustache'         , 'Narrow_Eyes'    ,
   'No_Beard'           , 'Oval_Face'        , 'Pale_Skin'      ,
   'Pointy_Nose'        , 'Receding_Hairline', 'Rosy_Cheeks'    ,
   'Sideburns'          , 'Smiling'          , 'Straight_Hair'  ,
   'Wavy_Hair'          , 'Wearing_Earrings' , 'Wearing_Hat'    ,
   'Wearing_Lipstick'   , 'Wearing_Necklace' , 'Wearing_Necktie',
   'Young'              
''')

args = parser.parse_args()
# model
img_size = args.img_size

# training
epochs = args.epoch
batch_size = args.batch_size
lr_base = args.lr
att = args.att
experiment_name = args.att


# ==============================================================================
# =                                 init model                                 =
# ==============================================================================
CUR_DIR  = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(CUR_DIR)
DATA_DIR = BASE_DIR + '/data/'
# strategy
print("======= Create strategy =======")
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
import tensorflow as tf
import data, models, loss, test
strategy = tf.distribute.MirroredStrategy()
print('Number of GPUs in use:', strategy.num_replicas_in_sync)



# optimizer
print("======= Create optimizers =======")
with strategy.scope():
   lr    = tf.Variable(initial_value=lr_base, trainable=False)
   opt   = tf.optimizers.Adam(lr, beta_1 =0.9, beta_2=0.999)
   #opt   = tf.optimizers.SGD(lr)
   params= tf.Variable(initial_value=[0, 0], trainable=False, dtype=tf.int64)


# model
print("======= Create model_lst =======")
with strategy.scope():
   Cls = models.Classifier()

# ==============================================================================
# =                                    backup                                  =
# ==============================================================================
with strategy.scope():
   x = tf.ones(shape=[2,img_size,img_size,3], dtype=tf.float32)
   y = Cls(x)

# save progress
checkpoint_dir = CUR_DIR + '/output/%s' % experiment_name
checkpoint = tf.train.Checkpoint(
   params=params,
   opt=opt,
   Cls=Cls
)

# load old checkpoint
print("======= Load old save point =======")
manager = tf.train.CheckpointManager(checkpoint, checkpoint_dir, max_to_keep=1)

with strategy.scope():
   if manager.latest_checkpoint:
      checkpoint.restore(manager.latest_checkpoint)
      print("Restored from {}".format(manager.latest_checkpoint))
   else:
      print("Initializing from scratch: " + experiment_name)
start_ep, start_it = params.numpy()
# ==============================================================================
# =                                    train                                   =
# ==============================================================================
def step(model_lst, inputs):
   def single_step(model_lst, inputs):
      Cls = model_lst
      w = Cls.trainable_variables
      with tf.GradientTape() as tape:
         tape.watch(w)
         loss = losses.loss(model_lst, inputs)
         
      grad = tape.gradient(loss, w)
      dw_w = zip(grad, w)
      opt.apply_gradients(dw_w)
      
      return loss

   loss = strategy.experimental_run_v2(single_step, args=(model_lst, inputs,))
   loss = strategy.reduce(tf.distribute.ReduceOp.SUM, loss, axis=None)
   return loss


with strategy.scope():
   g_batch_size = batch_size * strategy.num_replicas_in_sync
   
   # data
   print("======= Make data: %dx%d ======="%(img_size, img_size))
   
   tr_data = data.img_ds(data_dir=DATA_DIR+'/CelebAMask-HQ/', 
         att=att, img_resize=img_size, batch_size=g_batch_size, part='train')
   tr_ds  = strategy.experimental_distribute_dataset(tr_data.ds)
   tr_ite = iter(tr_ds)


   val_data = data.img_ds(data_dir=DATA_DIR+'/CelebAMask-HQ/', 
         att=att, img_resize=img_size, batch_size=g_batch_size, part='val')
   
   # loss, validation
   losses = loss.losses(g_batch_size) 
   tester = test.tester(strategy)
   

# create tf graph
print("======= Create graph =======")
graph = tf.function(step)

# training loop
print("======= Create training loop =======")
it_per_epoch = tr_data.count // g_batch_size
max_it = it_per_epoch * epochs
acc = tf.constant(0.)
try:
   for ep in range(start_ep, epochs):
      # update alpha
      lr.assign(lr_base / (10**ep))
      for it in range(start_it, it_per_epoch):
         tik = time.time()
         it_count = ep * it_per_epoch + it
         
         with strategy.scope():
            # update alpha
            #if it % 100 == 0:
            #   lr.assign(lr_base / (10 ** (ep // 100)))
            
            # get 1 batch
            inputs = next(tr_ite)
            model_lst = Cls

            # training
            loss = graph(model_lst, inputs)
            

            # validation
            if it % 100 == 0:
               acc = tester.validation(model_lst, val_data)
               
            
         # progress display
         if it % 1 == 0:
            tok = time.time()
            duration  = tok-tik
            remain_it = max_it - it_count - 1
            time_in_sec = int(remain_it * duration)
            remain_h  = time_in_sec // 3600 
            remain_m  = time_in_sec % 3600 // 60
            remain_s  = time_in_sec % 60
            print("EPOCH %d/%d - Batch %d/%d, Time: %.3f s - Remain: %d batches, Estimate: %02d:%02d:%02d " \
                  % (ep, epochs-1, it, it_per_epoch-1, duration, remain_it, remain_h, remain_m, remain_s))
            print( "Train loss: %.3f | Valid acc: %.2f"%(loss.numpy(), acc.numpy()) )

         # save model
         if it_count % 1000 == 0 and it_count != 0:
            params.assign([start_ep, start_it])
            manager.save()
            print('Model is saved!')
      start_it = 0
   
   # recall best validation
   valid_vars = tester.best_vars
   vars = Cls.variables
   acc = tester.validation(model_lst, val_data)
   print(acc.numpy())
   for i in range(len(vars)):
      vars[i].assign(valid_vars[i])
   acc = tester.validation(model_lst, val_data)
   print(acc.numpy())
   
   params.assign([start_ep, start_it])
   manager.save()
   print('Training finished...')
   print('Model is saved!')
# ==============================================================================
# =                                  save model                                =
# ==============================================================================
except:
   traceback.print_exc()
finally:
   # recall best validation
   valid_vars = tester.best_vars
   vars = Cls.variables
   acc = tester.validation(model_lst, val_data)
   print(acc.numpy())
   for i in range(len(vars)):
      vars[i].assign(valid_vars[i])
   acc = tester.validation(model_lst, val_data)
   print(acc.numpy())
   
   params.assign([ep, it])
   manager.save()
   print('Emergency backup...')
   print('Model is saved!')







