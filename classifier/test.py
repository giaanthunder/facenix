import argparse
import datetime
import json
import traceback

import numpy as np
import tensorflow as tf
import data, models, loss, test

import os, shutil, sys, math, time, copy

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import logging
logger = tf.get_logger()
logger.setLevel(logging.ERROR)


class tester():
   def __init__(self, strategy):
      self.strategy     = strategy
      self.graph = tf.function(self.step)
      #self.graph = self.step
      self.best_vars = None
      self.best_acc  = 0

   def validation(self, model_lst, val_data):
      val_ds = self.strategy.experimental_distribute_dataset(val_data.ds)
      val_ite = iter(val_ds)
      
      correct = 0
      for it in range(val_data.n_batch):
         with self.strategy.scope():
            inputs = next(val_ite)
            correct += self.graph(model_lst, inputs)
      acc = correct/val_data.count
      
      if acc >= self.best_acc:
         self.best_vars = []
         vars = model_lst.variables
         for var in vars:
            self.best_vars.append(tf.Variable(initial_value=var))
         self.best_acc   = acc
         
      return acc
      
   
   def step(self, model_lst, inputs):
      def single_step(model_lst, inputs):
         img, label = inputs
         Cls = model_lst
         
         pred = Cls(img, is_training=False)
         #print(pred.numpy())
         pred = tf.math.softmax(pred)
         #print(pred.numpy())
         pred = tf.argmax(pred, axis=1)
         label = tf.argmax(label, axis=1)
         #print(pred.numpy())
         #print(label.numpy())
         #sys.exit()
         #pred = tf.round(pred)
         #label= tf.round(label)
         correct = tf.where(pred==label, 1., 0.)
         correct = tf.reduce_sum(correct)
         
         return correct

      correct = self.strategy.experimental_run_v2(single_step, args=(model_lst, inputs,))
      correct = self.strategy.reduce(tf.distribute.ReduceOp.SUM, correct, axis=None)
      return correct
   
   
if __name__ == '__main__':
   parser = argparse.ArgumentParser()
   parser.add_argument('--img_size', dest='img_size', type=int, default=224)
   parser.add_argument('--batch_size', dest='batch_size', type=int, default=4)
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
   parser.add_argument('--name', dest='experiment_name', 
                    default=datetime.datetime.now().strftime('test_no_%y%m%d_%H%M%S'))

   args = parser.parse_args()
   img_size = args.img_size
   batch_size = args.batch_size
   att = args.att
   experiment_name = args.experiment_name + '_' + args.att
 
   os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
   strategy = tf.distribute.MirroredStrategy()
   print('Number of GPUs in use:', strategy.num_replicas_in_sync)
   

   ## optimizer
   #print("======= Create optimizers =======")
   #with strategy.scope():
   #   lr    = tf.Variable(initial_value=lr_base, trainable=False)
   #   opt   = tf.optimizers.Adam(lr, beta_1 =0.9, beta_2=0.999)
   #   params= tf.Variable(initial_value=[0, 0], trainable=False, dtype=tf.int64)


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

      
      
   exp_names = [
      'test_no_191130_153931_5_o_Clock_Shadow',
      'test_no_191130_173200_Arched_Eyebrows',
      'test_no_191130_192434_Attractive',
      'test_no_191130_211658_Bags_Under_Eyes',
      'test_no_191130_230938_Bald',
      'test_no_191201_010207_Bangs',
      'test_no_191201_025431_Big_Lips',
      'test_no_191201_044659_Big_Nose',
      'test_no_191201_063935_Black_Hair',
      'test_no_191201_140327_Blond_Hair',
      'test_no_191201_155600_Blurry',
      'test_no_191201_174858_Brown_Hair',
      'test_no_191201_194126_Bushy_Eyebrows',
      'test_no_191201_213408_Chubby',
      'test_no_191201_224850_Double_Chin',
      'test_no_191202_000351_Eyeglasses',
      'test_no_191202_011824_Goatee',
      'test_no_191202_023326_Gray_Hair',
      'test_no_191202_034802_Heavy_Makeup',
      'test_no_191202_050302_High_Cheekbones',
      'test_no_191202_061733_Male',
      'test_no_191202_073232_Mouth_Slightly_Open',
      'test_no_191202_084716_Mustache',
      'test_no_191202_132715_Narrow_Eyes',
      'test_no_191202_144235_No_Beard',
      'test_no_191202_160547_Oval_Face',
      'test_no_191202_172048_Pale_Skin',
      'test_no_191202_183530_Pointy_Nose',
      'test_no_191202_195034_Receding_Hairline',
      'test_no_191202_210525_Rosy_Cheeks',
      'test_no_191202_222029_Sideburns',
      'test_no_191202_233716_Smiling',
      'test_no_191203_005220_Straight_Hair',
      'test_no_191203_020702_Wavy_Hair',
      'test_no_191203_032149_Wearing_Earrings',
      'test_no_191203_043623_Wearing_Hat',
      'test_no_191203_055127_Wearing_Lipstick',
      'test_no_191203_070559_Wearing_Necklace',
      'test_no_191203_082056_Wearing_Necktie',
      'test_no_191204_181330_Young',
   ]
      
      
      
      
      
      
      
      
      
      
   for experiment_name in exp_names:
      att = experiment_name[22:]
      # tạo chỗ lưu tiến trình
      checkpoint_dir = './output/%s' % experiment_name
      checkpoint = tf.train.Checkpoint(
         #params=params,
         #opt=opt,
         Cls=Cls
      )

      # load checkpoint cũ
      #print("======= Load old save point =======")
      manager = tf.train.CheckpointManager(checkpoint, checkpoint_dir, max_to_keep=1)

      if manager.latest_checkpoint:
         checkpoint.restore(manager.latest_checkpoint)
         print("Restored from {}".format(manager.latest_checkpoint))
      else:
         print("ERROR: expeciment %s not found"%checkpoint_dir)
         sys.exit(1)
    
      with strategy.scope():
         g_batch_size = batch_size * strategy.num_replicas_in_sync
         
         # data
         #print("======= Make data: %dx%d ======="%(img_size, img_size))
         data_dir = '/data2/01_luan_van/data/CelebAMask-HQ/'
         test_data = data.img_ds(data_dir=data_dir, 
               att=att, img_resize=img_size, batch_size=g_batch_size, part='test')
         tester = test.tester(strategy)
         model_lst = Cls

         acc = tester.validation(model_lst, test_data)
         print( "Test acc: %.2f"%(acc.numpy()) )





