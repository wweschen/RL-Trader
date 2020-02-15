import os
import pprint
import inspect
import sys
import tensorflow as tf

import errno
import datetime


pp = pprint.PrettyPrinter().pprint

def class_vars(obj):
  return {k:v for k, v in inspect.getmembers(obj)
      if not k.startswith('__') and not callable(k)}

class BaseModel(object):
  """Abstract object representing an Reader model."""
  def __init__(self, config):
    self._saver = None
    self.config = config

    try:
      self._attrs = config.__dict__['__flags']
    except:
      self._attrs = class_vars(config)
    pp(self._attrs)

    self.config = config

    for attr in self._attrs:
      name = attr if not attr.startswith('_') else attr[1:]
      setattr(self, name, getattr(self.config, attr))

  def save_model(self, step=None):
    print(" [*] Saving checkpoints...")
    model_name = type(self).__name__

    if not os.path.exists(self.checkpoint_dir):
      os.makedirs(self.checkpoint_dir)
    self.saver.save(self.sess, self.checkpoint_dir, global_step=step)

  def load_model(self):
    print(" [*] Loading checkpoints...")

    ckpt = tf.train.get_checkpoint_state(self.checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
      ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
      fname = os.path.join(self.checkpoint_dir, ckpt_name)
      self.saver.restore(self.sess, fname)
      print(" [*] Load SUCCESS: %s" % fname)
      return True
    else:
      print(" [!] Load FAILED: %s" % self.checkpoint_dir)
      return False

  @property
  def checkpoint_dir(self):
    return  self.config.env_name+'/checkpoints/'

  @property
  def model_dir(self):
    model_dir = self.config.env_name
    # for k, v in self._attrs.items():
    #   if not k.startswith('_') and k not in ['display']:
    #     model_dir += "-%s-%s" % (k, ",".join([str(i) for i in v])
    #         if type(v) == list else v)
    return model_dir + '/models/'

  @property
  def saver(self):
    if self._saver == None:
      self._saver = tf.train.Saver(max_to_keep=10)
    return self._saver

  def log_trade(self,action,date,time,unit,order_price,current_price,gain):
    saction = 'Bought' if action==1 else 'Sold  '
    sdate = date.strftime('%Y-%m-%d')
    stime =time.strftime('%H:%M:%S')

    path = os.path.join(self.config.env_name,'trade_logs',sdate ,"trade_log.csv")
    if not os.path.exists(os.path.dirname(path)):
      try:
        os.makedirs(os.path.dirname(path))
      except OSError as exc:  # Guard against race condition
        if exc.errno != errno.EEXIST:
          raise
    if os.path.exists(path):
      append_write = 'a'  # append if already exists
    else:
      append_write = 'w'  # make a new file if not

    with open(path, append_write) as f:
       f.write('{},{},{},{},{:0.2f},{:0.2f},{:0.2f}\n'.format(sdate,stime,saction,unit,order_price,current_price,gain))