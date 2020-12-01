import os, sys, math, time
import argparse
import datetime
import json
import traceback

import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import logging
logger = tf.get_logger()
logger.disabled = True
logger.setLevel(logging.FATAL)

# ==============================================================================
# =                                    param                                   =
# ==============================================================================

parser = argparse.ArgumentParser()
parser.add_argument('--epoch', dest='epoch', type=int, default=12, help='# of epochs')
parser.add_argument('--gpu', dest='gpu', type=str, default='0', help='# of gpu computing in paralel')
parser.add_argument('--experiment_name', dest='experiment_name',
                          default=datetime.datetime.now().strftime("test_no_%y%m%d_%H%M%S"))

args = parser.parse_args()
epochs = args.epoch
experiment_name = args.experiment_name

# ==============================================================================
# =                                   init                                     =
# ==============================================================================
CUR_DIR  = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(CUR_DIR)
DATA_DIR = BASE_DIR + '/data/'
OUTPUT_DIR = CUR_DIR + '/output/%s/' % experiment_name
SAMP_DIR = OUTPUT_DIR + '%s_sample' % (experiment_name)
CKPT_DIR = OUTPUT_DIR + 'trained_model'

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

# strategy = tf.distribute.MirroredStrategy()
# tf.distribute.experimental_set_strategy(strategy)
# print("Number of GPUs in use:", strategy.num_replicas_in_sync)



# optimizer
print("======= Create optimizers =======")
lr    = tf.Variable(initial_value=0., trainable=False)
g_opt = tf.optimizers.Adam(lr, beta_1 =0., beta_2=0.99)
d_opt = tf.optimizers.Adam(lr, beta_1 =0., beta_2=0.99)
params= tf.Variable(initial_value=[0, 0, 0], trainable=False, dtype=tf.int16)


def d_step(model_lst, inputs):
    def single_step(model_lst, inputs):
        Gen, Dis = model_lst

        with tf.GradientTape() as d_tape:
            dis_loss = losses.d_loss(model_lst, inputs)
            
        w     = Dis.trainable_variables
        grad = d_tape.gradient(dis_loss, w)
        dw_w = zip(grad, w)
        d_opt.apply_gradients(dw_w)
        
        return dis_loss

    loss = single_step(model_lst, inputs)
    # loss = strategy.experimental_run_v2(single_step, args=(model_lst, inputs,))
    # loss = strategy.reduce(tf.distribute.ReduceOp.SUM, loss, axis=None)
    return loss


def g_step(model_lst, inputs):
    def single_step(model_lst, inputs):
        Gen, Dis = model_lst

        with tf.GradientTape() as g_tape:
            gen_loss = losses.g_loss(model_lst, inputs)
            
        w     = Gen.trainable_variables
        grad = g_tape.gradient(gen_loss, w)
        dw_w = zip(grad, w)
        g_opt.apply_gradients(dw_w)
        
        return gen_loss

    z, x_real = inputs
    bs = z.shape[0]
    z = tf.random.normal([bs,512])
    inputs = [z, x_real]
    loss = single_step(model_lst, inputs)
    # loss = strategy.experimental_run_v2(single_step, args=(model_lst, inputs,))
    # loss = strategy.reduce(tf.distribute.ReduceOp.SUM, loss, axis=None)
    return loss




# model
print("======= Create model_lst =======")
Gen = models.G_style()
Dis = models.D_basic()
Gen.is_training = True


# ==============================================================================
# =                              backup & restore                              =
# ==============================================================================

checkpoint = tf.train.Checkpoint(d_opt=d_opt, g_opt=g_opt, lr=lr, params=params, Gen=Gen, Dis=Dis)


# load old checkpoint
manager = tf.train.CheckpointManager(checkpoint, CKPT_DIR, max_to_keep=600)
print("Number of checkpoints", len(manager.checkpoints))
latest = manager.latest_checkpoint
# latest = manager.checkpoints[41]
if latest:
    restore_from_interrupt = True
    checkpoint.restore(latest)
    print("Restored from {}".format(latest))
else:
    restore_from_interrupt = False
    print("Initializing from scratch. Experiment name:", experiment_name)




train_phases = [ 
    (  8,32,0.0020, 600,False), 
    ( 16,32,0.0020, 600,True), ( 16, 4,0.0010,600,False),
    ( 32,16,0.0020, 600,True), ( 32, 4,0.0010,600,False),
    ( 64,16,0.0020, 600,True), ( 64, 4,0.0010,600,False),
    (128, 8,0.0020, 600,True), (128, 4,0.0010,600,False),
    (256, 4,0.0020, 600,True), (256, 4,0.0010,600,False),
    (512, 4,0.0020, 600,True), (512, 4,0.0010,600,False)
]

# collect executed time
Gen1 = models.G_style()
Dis1 = models.D_basic()
Gen1.synthesis.alpha.assign(0.9)
Dis1.alpha.assign(0.9)
Gen1.is_training = True
time_lst = {}
for cur_res, batch_size, _, _,_ in train_phases:
    print(cur_res)
    # g_batch_size = batch_size * strategy.num_replicas_in_sync
    # losses = loss.losses2(g_batch_size, True) 
    g_batch_size = batch_size
    losses = loss.losses2() 
    z = tf.ones([g_batch_size, 512])
    x = tf.ones([g_batch_size, cur_res, cur_res, 3])
    inputs = (z, x)

    g_graph = tf.function(g_step)
    d_graph = tf.function(d_step)

    Gen1.synthesis.lod = int(math.log2(cur_res)) - 2
    Dis1.lod = int(math.log2(cur_res)) - 2

    model_lst = (Gen1, Dis1)
    g_loss = g_graph(model_lst, inputs)
    d_loss = d_graph(model_lst, inputs)

    tik = time.time()
    g_loss = g_graph(model_lst, inputs)
    d_loss = d_graph(model_lst, inputs)
    tok = time.time()
    duration = tok - tik
    time_lst[cur_res] = duration
print("Time collect:", time_lst)
del Gen1
del Dis1


# ==============================================================================
# =                                  train                                     =
# ==============================================================================


tf.random.set_seed(np.random.randint(1 << 31))
phs_num = len(train_phases)
start_phs, start_ep, start_it = params.numpy()
# start_phs, start_ep, start_it = [7,0,0]
up_alp_interval = 10
save_ep_interval = 4
disp_interval = 2
nan_error = False

try:
    # phase train, 8 phases
    for phs in range(start_phs, phs_num):
        cur_res, batch_size, l_rate, k_imgs, update_alp = train_phases[phs]
        lr.assign(l_rate)


        # g_batch_size = batch_size * strategy.num_replicas_in_sync
        # losses = loss.losses2(g_batch_size, tf.distribute.has_strategy()) 
        g_batch_size = batch_size
        losses = loss.losses2() 
        tester = test.tester(SAMP_DIR, batch_size)
        
        # data
        print("======= Make data: %dx%d ======="%(cur_res, cur_res))
        tr_data = data.img_ds(data_dir=DATA_DIR, img_resize=cur_res, batch_size=g_batch_size)
        # tr_ite  = iter(strategy.experimental_distribute_dataset(tr_data.ds))
        tr_ite  = iter(tr_data.ds)

        # create tf graph
        d_graph = tf.function(d_step)
        g_graph = tf.function(g_step)


        # training loop
        print("======= Create training loop =======")
        epochs     = k_imgs*1000//tr_data.count
        it_per_ep  = tr_data.count // g_batch_size
        it_per_phs = it_per_ep * epochs
        it_all_phs = it_per_phs * len(train_phases)

        it_stop = it_per_phs//2
        
        Gen.synthesis.lod = int(math.log2(cur_res)) - 2
        Dis.lod = int(math.log2(cur_res)) - 2

        # epoch train
        for ep in range(start_ep, epochs):
            tr_data = data.img_ds(data_dir=DATA_DIR, img_resize=cur_res, batch_size=g_batch_size)
            tr_ite  = iter(tr_data.ds)

            for it in range(start_it, it_per_ep):
                tik = time.time()
                it_phs_cnt = it_per_ep * ep + it
                
                # update alpha
                if update_alp and (it_phs_cnt%up_alp_interval == 0 or restore_from_interrupt):
                    if it_phs_cnt < it_stop:
                        alpha = 1. - (it_phs_cnt/it_per_phs) * 2.
                        Gen.synthesis.alpha.assign(alpha)
                        Dis.alpha.assign(alpha)
                    else:
                        Gen.synthesis.alpha.assign(0.)
                        Dis.alpha.assign(0.)
                        update_alp = False

                    if restore_from_interrupt:
                        restore_from_interrupt = False


                # get 1 batch
                inputs = next(tr_ite)
                model_lst = (Gen, Dis)
                
                # train G
                g_loss = g_graph(model_lst, inputs)
                if tf.math.is_nan(g_loss):
                    print('G LOSS IS NAN')
                    nan_error = True
                    exit()

                # train D
                d_loss = d_graph(model_lst, inputs)
                if tf.math.is_nan(d_loss):
                    print('D LOSS IS NAN')
                    nan_error = True
                    exit()
                
                # save sample
                if it % 100 == 0:
                    # strategy.experimental_run_v2(tester.make_sample, 
                    #         args=(model_lst, inputs, cur_res, ep, it,))
                    tester.make_sample(model_lst, inputs, cur_res, ep, it, phs)

                # progress display
                if it % int(disp_interval/time_lst[cur_res]) == 0:
                    tok = time.time()
                    duration  = tok-tik
                    remain_it = it_per_phs - it_phs_cnt - 1
                    
                    ph_time    = remain_it * duration
                    
                    total_time = ph_time
                    for p in range(phs+1, phs_num):
                        res, bs, _, _, _ = train_phases[p]
                        t = time_lst[res]
                        # n_it = (tr_data.count // (bs*strategy.num_replicas_in_sync)) * epochs
                        n_it = (tr_data.count // bs) * epochs
                        total_time += t*n_it
                        remain_it += n_it
                    remain_h  = total_time // 3600 
                    remain_m  = total_time % 3600 // 60
                    
                    print("EPOCH %d/%d - Batch %d/%d, Time: %.3f s - Remain: %d batches, %02d:%02d " \
                            % (ep, epochs-1, it, it_per_ep-1, duration, remain_it, remain_h, remain_m))
                    print("g_loss: %.3f - d_loss: %.3f | res: %d"%
                        (g_loss.numpy(), d_loss.numpy(), cur_res))


            start_it = 0
            # save model
            if ep%save_ep_interval == 0 and ep != start_ep:
                params.assign([phs, ep, it])
                manager.save()
                print('Model save after %d epoch...'%save_ep_interval)

        start_ep = 0
        params.assign([0, 0, 0])
        manager.save()
        print('Model save after phase end...')

    start_phs = 0
    
    params.assign([0, 0, 0])
    manager.save()
    print('Model save after train...')
# ==============================================================================
# =                                             save model                                          =
# ==============================================================================
except:
     traceback.print_exc()
finally:
    if not nan_error: 
        params.assign([phs, ep, it])
        manager.save()
        print('Model save after exception...')






