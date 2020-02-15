import time
import numpy as np
import os.path
import csv
import errno
from collections import defaultdict

import sys
if (sys.version_info[0]==2):
  import cPickle
elif (sys.version_info[0]==3):
  import _pickle as cPickle



def rgb2gray(image):
  return np.dot(image[...,:3], [0.299, 0.587, 0.114])

def timeit(f):
  def timed(*args, **kwargs):
    start_time = time.time()
    result = f(*args, **kwargs)
    end_time = time.time()

    print("   [-] %s : %2.5f sec" % (f.__name__, end_time - start_time))
    return result
  return timed

def get_time():
  return time.strftime("%Y-%m-%d_%H:%M:%S", time.gmtime())

@timeit
def save_pkl(obj, path):
    if not os.path.exists(os.path.dirname(path)):
        try:
            os.makedirs(os.path.dirname(path))
        except OSError as exc:  # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise

    with open(path, 'wb') as f:
        cPickle.dump(obj, f)
        print("  [*] save %s" % path)

@timeit
def load_pkl(path):
    if os.path.exists(path):
      with open(path,'rb') as f:
        obj = cPickle.load(f)
        print("  [*] load %s" % path)
        return obj

@timeit
def save_npy(obj, path):
  np.save(path, obj)
  print("  [*] save %s" % path)

@timeit
def load_npy(path):
  obj = np.load(path)
  print("  [*] load %s" % path)
  return obj

def QFunc():
    return np.zeros(3)

@timeit
def save_dict_cvs(dict,path):
    if not os.path.exists(os.path.dirname(path)):
        try:
            os.makedirs(os.path.dirname(path))
        except OSError as exc:  # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise
    with open(path, 'w') as f:
        writer = csv.writer(f)
        for k,v in dict.items():
            writer.writerow([k] + v)

@timeit
def load_dict_cvs(dict, path):
    if os.path.exists(path):
        with open(path, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                dict[row[0]] = row[1:]

def pprint_dict(d, indent=0):
   for key, value in d.items():
      print('\t' * indent + str(key))
      if isinstance(value, dict):
          pprint_dict(value, indent+1)
      else:
         print('\t' * (indent+1) + str(value))