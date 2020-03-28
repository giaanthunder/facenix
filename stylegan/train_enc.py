import argparse, datetime, json, traceback

import numpy as np


import os, shutil, sys, math, time

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# import logging
# logger = tf.get_logger()
# logger.setLevel(logging.ERROR)

# ==============================================================================
# =                                    param                                   =
# ==============================================================================

parser = argparse.ArgumentParser()
parser.add_argument('--epoch', dest='epoch', type=int, default=20, help='# of epochs')
parser.add_argument('--lr', dest='lr', type=float, default=0.02, help='learning rate')
parser.add_argument('--bs', dest='bs', type=int, default=2, help='batch size')
parser.add_argument('--gpu', dest='gpu', type=str, default='0', help='# of gpu computing in paralel')
parser.add_argument('--name', dest='name',
                    default=datetime.datetime.now().strftime("test_no_%y%m%d_%H%M%S"))

args = parser.parse_args()
epochs = args.epoch
lr_base = args.lr
batch_size = args.bs
name = args.name

# ==============================================================================
# =                                   init                                     =
# ==============================================================================
CUR_DIR  = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(CUR_DIR)
DATA_DIR = BASE_DIR + '/data/'
OUTPUT_DIR = CUR_DIR + '/output/%s/' % (name+'_enc')
SAMP_DIR = OUTPUT_DIR + '%s_sample' % (name)
CKPT_DIR = OUTPUT_DIR + 'checkpoints'

# create output dir
if not os.path.exists(OUTPUT_DIR):
   os.makedirs(OUTPUT_DIR, exist_ok=True)
# create sample dir
if not os.path.exists(SAMP_DIR):
   os.makedirs(SAMP_DIR, exist_ok=True)

# save setting information
with open(OUTPUT_DIR + 'setting.txt', 'w') as f:
   f.write(json.dumps(vars(args), indent=3, separators=(',', ':')))


# ==============================================================================
# =                                    train                                   =
# ==============================================================================
# strategy
print("======= Create strategy =======")
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
import tensorflow as tf
import data, models, loss, test

strategy = tf.distribute.MirroredStrategy()
tf.distribute.experimental_set_strategy(strategy)
print("Number of GPUs in use:", strategy.num_replicas_in_sync)



# optimizer
print("======= Create optimizers =======")
lr    = tf.Variable(initial_value=lr_base, trainable=False)
# e_opt = tf.optimizers.Adam(lr, beta_1 =0., beta_2=0.99)
e_opt = tf.optimizers.Adam(lr)
params= tf.Variable(initial_value=[0, 0], trainable=False, dtype=tf.int16)


# model
print("======= Create model =======")
Gen = models.pretrained_models()
Enc = models.EncoderNN()

def e_step(model_lst, inputs):
   def single_step(model_lst, inputs):
      Gen, Enc = model_lst

      with tf.GradientTape() as e_tape:
         enc_loss = losses.e_loss(model_lst, inputs)
         
      w    = Enc.trainable_variables
      grad = e_tape.gradient(enc_loss, w)
      dw_w = zip(grad, w)
      e_opt.apply_gradients(dw_w)
      
      return enc_loss

   loss = strategy.experimental_run_v2(single_step, args=(model_lst, inputs,))
   loss = strategy.reduce(tf.distribute.ReduceOp.SUM, loss, axis=None)
   return loss



# ==============================================================================
# =                              backup & restore                              =
# ==============================================================================

checkpoint = tf.train.Checkpoint(e_opt=e_opt, params=params, Enc=Enc)

# load old checkpoint
manager = tf.train.CheckpointManager(checkpoint, CKPT_DIR, max_to_keep=3)
if manager.latest_checkpoint:
   checkpoint.restore(manager.latest_checkpoint)
   print("Restored from {}".format(manager.latest_checkpoint))
else:
   print("Initializing from scratch. Experiment name:", name)


# ==============================================================================
# =                                    train                                   =
# ==============================================================================

try:
   tf.random.set_seed(np.random.randint(1 << 31))
   start_ep, start_it = params.numpy()
   up_alp_interval = 10

   g_batch_size = batch_size * strategy.num_replicas_in_sync
   losses = loss.losses(g_batch_size, tf.distribute.has_strategy()) 
   tester = test.tester(SAMP_DIR, batch_size)
   
   # data
   print("======= Make data =======")
   tr_data = data.img_ds(data_dir=DATA_DIR, img_resize=1024, batch_size=g_batch_size)
   tr_ite = iter(strategy.experimental_distribute_dataset(tr_data.ds))

   # create tf graph
   e_graph = tf.function(e_step)


   # training loop
   print("======= Create training loop =======")
   it_per_ep  = tr_data.count // g_batch_size
   it_per_phs = it_per_ep * epochs

   for ep in range(start_ep, epochs):
      for it in range(start_it, it_per_ep):
         tik = time.time()
         it_phs_cnt = it_per_ep * ep + it

         # get 1 batch
         inputs = next(tr_ite)
         model_lst = (Gen, Enc)

         # train E
         e_loss = e_step(model_lst, inputs)
         
         # save sample
         if it % 100 == 0:
            strategy.experimental_run_v2(tester.make_enc_sample, 
                  args=(model_lst, inputs, ep, it,))

         # progress display
         if it % 1 == 0:
            tok = time.time()
            duration  = tok-tik
            remain_it = it_per_phs - it_phs_cnt - 1
            
            total_time= remain_it * duration

            remain_h  = total_time // 3600 
            remain_m  = total_time % 3600 // 60
            
            print("EPOCH %d/%d - Batch %d/%d, Time: %.3f s - Remain: %d batches, %02d:%02d " \
                  % (ep, epochs-1, it, it_per_ep-1, duration, remain_it, remain_h, remain_m))
            print("e_loss: %.3f" % e_loss.numpy())

         # save model
         if it_phs_cnt%1000 == 0 and it_phs_cnt != 0:
            params.assign([ep, it])
            manager.save()
            print('Model is saved!')
      start_it = 0
   start_ep = 0
   
   params.assign([0, 0])
   manager.save()
   print('Model is saved!')
# ==============================================================================
# =                                  save model                                =
# ==============================================================================
except:
    traceback.print_exc()
finally:
   params.assign([ep, it])
   manager.save()
   print('Model is saved!')






