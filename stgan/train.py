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
att_default = ['Bald', 'Bangs', 'Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Bushy_Eyebrows', 'Eyeglasses', 'Male', 'Mouth_Slightly_Open', 'Mustache', 'No_Beard', 'Pale_Skin', 'Young']
parser.add_argument('--atts', dest='atts', default=att_default, choices=data.att_dict.keys(), nargs='+', help='attributes to learn')
parser.add_argument('--img_size', dest='img_size', type=int, default=128)
parser.add_argument('--z_dim', dest='z_dim', type=int, default=512)
# training
parser.add_argument('--epoch', dest='epoch', type=int, default=1, help='# of epochs')
parser.add_argument('--batch_size', dest='batch_size', type=int, default=16)
parser.add_argument('--lr', dest='lr', type=float, default=0.0002, help='learning rate')
parser.add_argument('--n_d', dest='n_d', type=int, default=5, help='# of d updates per g update')
parser.add_argument('--gpu', dest='gpu', type=str, default='0', help='# of gpu computing in paralel')
parser.add_argument('--experiment_name', dest='experiment_name',
                    default=datetime.datetime.now().strftime("test_no_%y%m%d_%H%M%S"))

args = parser.parse_args()
# model
atts = args.atts
n_att = len(atts)
img_size = args.img_size
z_dim = args.z_dim

# training
mode = args.mode
epochs = args.epoch
batch_size = args.batch_size
lr_base = args.lr
n_d = args.n_d
experiment_name = args.experiment_name

# ==============================================================================
# =                                   init                                     =
# ==============================================================================
CUR_DIR  = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(CUR_DIR)
# DATA_DIR = BASE_DIR + '/data/'
DATA_DIR = '/data2/01_luan_van/data/'

# save setting information
output_dir = CUR_DIR + '/output/%s' % experiment_name
if not os.path.exists(output_dir):
   os.makedirs(output_dir, exist_ok=True)

with open(CUR_DIR + '/output/%s/setting.txt' % experiment_name, 'w') as f:
   f.write(json.dumps(vars(args), indent=4, separators=(',', ':')))

# save sample
sample_dir = CUR_DIR + '/output/%s/%s_sample' % (experiment_name,experiment_name)
if not os.path.exists(sample_dir):
   os.makedirs(sample_dir, exist_ok=True)

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

g_batch_size = batch_size * strategy.num_replicas_in_sync


# optimizer
print("======= Create optimizers =======")
#with strategy.scope():
lr    = tf.Variable(initial_value=lr_base, trainable=False)
g_opt = tf.optimizers.Adam(lr, beta_1 =0., beta_2=0.99)
d_opt = tf.optimizers.Adam(lr, beta_1 =0., beta_2=0.99)
params= tf.Variable(initial_value=[0, 3], trainable=False, dtype=tf.int64)


# model
print("======= Create model_lst =======")
#with strategy.scope():
Gen = models.Generator()
Dis = models.Discriminator(n_att)
Enc = models.Encoder()
Stu = models.STU()
   

      
# data
print("======= Make data: %dx%d ======="%(img_size, img_size))
tr_count, tr_data = data.img_ds(data_dir=DATA_DIR, 
      atts=atts, img_resize=img_size, batch_size=g_batch_size, part='train')
tr_data = strategy.experimental_distribute_dataset(tr_data)
tr_ite = iter(tr_data)

# ==============================================================================
# =                                    backup                                  =
# ==============================================================================
# def init_vars(model_lst, inputs):
#    Gen, Dis, Enc, Stu = model_lst
#    x, a = inputs
#    z      = Enc(x)
#    z_stu  = Stu(z, a)
#    x_fake = Gen(z_stu, a)
#    d, att = Dis(x)

# # with strategy.scope():
# inputs = next(tr_ite)
# model_lst = [Gen, Dis, Enc, Stu]
# strategy.experimental_run_v2(init_vars, args=(model_lst, inputs,))

x = tf.ones(shape=[1,128,128,3], dtype=tf.float32)
a = tf.ones(shape=[1,13], dtype=tf.float32)

z      = Enc(x)
z_stu  = Stu(z, a)
x_fake = Gen(z_stu, a)
d, att = Dis(x)



# tạo chỗ lưu tiến trình
checkpoint_dir = CUR_DIR + '/output/%s/trained_model' % experiment_name
checkpoint = tf.train.Checkpoint(
   params=params,
   d_opt=d_opt, g_opt=g_opt,
   Gen=Gen, Dis=Dis, Enc=Enc, Stu=Stu 
)

# load checkpoint cũ
print("======= Load old save point =======")
manager = tf.train.CheckpointManager(checkpoint, checkpoint_dir, max_to_keep=3)

if manager.latest_checkpoint:
   checkpoint.restore(manager.latest_checkpoint)
   print("Restored from {}".format(manager.latest_checkpoint))
else:
   print("Experiment name:", experiment_name)
   print("Initializing from scratch.")
start_ep, start_it = params.numpy()
# ==============================================================================
# =                                    train                                   =
# ==============================================================================

def d_step(model_lst, inputs):
   def single_step(model_lst, inputs):
      Gen, Dis, Enc, Stu = model_lst
      with tf.GradientTape() as d_tape:
         dis_loss = losses.d_loss(model_lst, inputs)
         
      w = Dis.trainable_variables
      grad = d_tape.gradient(dis_loss, w)
      dw_w = zip(grad, w)
      d_opt.apply_gradients(dw_w)
      
      return dis_loss

   x_real, a = inputs
   # convert [-1, 1] to [-0.5, 0.5]
   a = (a * 2 - 1) * 0.5
   b = tf.random.shuffle(a)
   inputs = [x_real, a, b]
   loss = strategy.experimental_run_v2(single_step, args=(model_lst, inputs,))
   loss = strategy.reduce(tf.distribute.ReduceOp.SUM, loss, axis=None)
   return loss


def g_step(model_lst, inputs):
   def single_step(model_lst, inputs):
      Gen, Dis, Enc, Stu = model_lst
      with tf.GradientTape() as g_tape:
         gen_loss = losses.g_loss(model_lst, inputs)
         
      w = Gen.trainable_variables
      w.extend(Enc.trainable_variables)
      w.extend(Stu.trainable_variables)
      grad = g_tape.gradient(gen_loss, w)
      dw_w = zip(grad, w) 
      
      g_opt.apply_gradients(dw_w)
      
      return gen_loss

   x_real, a = inputs
   # convert [0, 1] to [-0.5, 0.5]
   a = (a * 2 - 1) * 0.5
   b = tf.random.shuffle(a)
   inputs = [x_real, a, b]
   loss = strategy.experimental_run_v2(single_step, args=(model_lst, inputs,))
   loss = strategy.reduce(tf.distribute.ReduceOp.SUM, loss, axis=None)
   return loss

try:
   #with strategy.scope():
   #tf.random.set_seed(np.random.randint(1 << 31))
   losses = loss.losses(g_batch_size, True) 
   tester = test.tester(sample_dir, batch_size, z_dim)


   
   # create tf graph
   print("======= Create graph =======")
   d_graph = tf.function(d_step)
   g_graph = tf.function(g_step)
   #d_graph = d_step
   #g_graph = g_step

   # training loop
   print("======= Create training loop =======")
   it_per_epoch = tr_count // (g_batch_size*n_d)
   max_it = it_per_epoch * epochs
   
   for ep in range(start_ep, epochs):
      for it in range(start_it, it_per_epoch):
         tik = time.time()
         it_count = ep * it_per_epoch + it
         
         #with strategy.scope():
         # update alpha
         lr.assign(lr_base / (10 ** (ep // 100)))
         
         # get 1 batch
         inputs = next(tr_ite)
         model_lst = (Gen, Dis, Enc, Stu)

         # train G
         g_loss = g_step(model_lst, inputs)
         
         # train D
         d_loss = d_step(model_lst, inputs)
         for i in range(n_d-1):
            inputs = next(tr_ite)
            d_loss = d_step(model_lst, inputs)

         # save sample
         if it % 100 == 0:
            x_real, a = inputs
            # convert [0, 1] to [-0.5, 0.5]
            a = (a * 2 - 1) * 0.5
            b = tf.random.shuffle(a)
            inputs = [x_real, a, b]
            strategy.experimental_run_v2(tester.make_sample, args=(
                  model_lst, inputs, 128, ep, it,))
            
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
            print("g_loss: %.3f - d_loss: %.3f" % (g_loss.numpy(), d_loss.numpy()))

         # save model
         if it_count % 1000 == 0 and it_count != 0:
            params.assign([start_ep, start_it])
            manager.save()
            print('Model is saved!')
      start_it = 0
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
   params.assign([ep, it])
   manager.save()
   print('Emergency backup...')
   print('Model is saved!')







