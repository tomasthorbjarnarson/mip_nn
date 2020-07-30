import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math
import random
import copy
from keras.datasets import mnist,cifar10
from globals import EPSILON

def get_unique_examples(train_x, train_y, N):
  # When working with minimal data, get as representative examples as possible
  ex_per_class = math.ceil(N / 10)
  indices = []
  for i, ex in enumerate(train_x):
    if len(indices) < ex_per_class*10 and np.count_nonzero(train_y[indices] == train_y[i]) < ex_per_class:
      indices.append(i)

  return indices[0:N]

def normalize(matrix, row=False):
  # Have same precision as global minimum comparison
  precision = int(np.log10(EPSILON))*-1
  if row:
    # Normalize each column
    matrix = (matrix - np.min(matrix, axis=0)) / (np.max(matrix, axis=0) - np.min(matrix, axis=0))
  else:
    # Normalize every element in matrix
    matrix = (matrix - np.min(matrix)) / (np.max(matrix) - np.min(matrix))
  # Round to precision so sign calculations are easier
  matrix = np.around(matrix, decimals=precision)
  return matrix

def oh_encode(labels):
  classes = int(np.max(labels))
  encoded = np.zeros((len(labels), classes+1)) - 1

  for i,val in enumerate(labels):
    encoded[i][int(val)] = 1
  return encoded

def get_architecture(data, hls):
  """data is dict with train/test/val data. hls is list of size of hidden layers"""
  in_neurons = [data["train_x"].shape[1]]
  out_neurons = [data["oh_train_y"].shape[1]]

  return in_neurons + hls + out_neurons

def get_batches(data, batch_size):
  batches = []
  for i in range(0, len(data["train_x"]), batch_size):
    tmp_data = copy.deepcopy(data)
    tmp_data["train_x"] = data["train_x"][i:i+batch_size]
    tmp_data["train_y"] = data["train_y"][i:i+batch_size]
    tmp_data["oh_train_y"] = data["oh_train_y"][i:i+batch_size]

    batches.append(tmp_data)
  
  return batches

def get_training_batches(data, batch_size):
  train_batches = []
  for i in range(0, len(data["train_x"]), batch_size):
    tmp_data = {}
    tmp_data["train_x"] = data["train_x"][i:i+batch_size]
    tmp_data["train_y"] = data["train_y"][i:i+batch_size]
    tmp_data["oh_train_y"] = data["oh_train_y"][i:i+batch_size]
    tmp_data["val_x"] = data["val_x"]
    tmp_data["val_y"] = data["val_y"]
    tmp_data["oh_val_y"] = data["oh_val_y"]

    train_batches.append(tmp_data)
  
  return train_batches
   

def load_data(dataset, N, seed):
  """Return list of batches of dataset. If batch_size == 0 : all N elements are in first batch"""
  random.seed(seed)
  if dataset == "mnist":
    data = load_keras(mnist, N)
  elif dataset == "cifar":
    data = load_keras(cifar10, N)
  elif dataset == "adult":
    data = load_adult(N)
  elif dataset == "heart":
    data = load_heart(N)
  else:
    raise Exception("Dataset %s not known" % dataset)

  return data

def load_keras(dataset, N):

  (train_x, train_y), (test_x, test_y) = dataset.load_data()
  
  train_x = normalize(train_x)
  test_x = normalize(test_x)

  pixels = np.prod(train_x.shape[1:])
  train_x = train_x.reshape(-1, pixels)
  test_x = test_x.reshape(-1, pixels)

  train_indices = [x for x in range(len(train_x))]
  random.shuffle(train_indices)

  train_x = train_x[train_indices]
  train_y = train_y[train_indices]

  max_num = len(train_x)
  if N < 500:
    train_indices = get_unique_examples(train_x, train_y, N)
  else:
    train_indices = range(N)
  val_indices = [i for i in range(max_num) if i not in train_indices]

  data = {}
  data["train_x"] = train_x[train_indices]
  data["train_y"] = train_y[train_indices]
  data["oh_train_y"] = oh_encode(train_y[train_indices])

  data["val_x"] = train_x[val_indices]
  data["val_y"] = train_y[val_indices]
  data["oh_val_y"] = oh_encode(train_y[val_indices])

  data["test_x"] = test_x
  data["test_y"] = test_y
  data["oh_test_y"] = oh_encode(test_y)

  return data

def load_heart(N):

  heart = pd.read_csv('data/heart.csv')
  cp = pd.get_dummies(heart['cp'], prefix='cp')
  thal = pd.get_dummies(heart['thal'], prefix='thal')
  slope = pd.get_dummies(heart['slope'], prefix='slope')
  frames = [heart, cp, thal, slope]
  heart = pd.concat(frames, axis=1)
  heart = heart.drop(columns=['cp', 'thal', 'slope'])

  train_size = math.floor(0.8*len(heart))

  heart_y = heart['target'].to_numpy()
  heart_x = heart.drop(columns=['target']).to_numpy()

  all_indices = [x for x in range(len(heart))]
  random.shuffle(all_indices)
  heart_x = heart_x[all_indices]
  heart_y = heart_y[all_indices]
  
  heart_x = normalize(heart_x, True)

  val_size = math.floor(0.1*train_size)

  if N > train_size - val_size:
    raise Exception("N larger than training sample: %s > %s" % (N, train_size - val_size))

  val_indices = all_indices[0:val_size]
  train_indices = all_indices[val_size:val_size + N]
  test_indices = all_indices[train_size:]

  data = {}
  data["train_x"] = heart_x[train_indices]
  data["train_y"] = heart_y[train_indices]
  data["oh_train_y"]= oh_encode(heart_y[train_indices])

  data["val_x"] = heart_x[val_indices]
  data["val_y"] = heart_y[val_indices]
  data["oh_val_y"]= oh_encode(heart_y[val_indices])

  data["test_x"] = heart_x[test_indices]
  data["test_y"] = heart_y[test_indices]
  data["oh_test_y"]= oh_encode(heart_y[test_indices])

  return data

def load_adult(N):

  columns = ['age', 'workclass', 'fnlwgt', 'education', 'education.num', 'marital.status',
             'occupation', 'relationship', 'race', 'sex', 'capital.gain', 'capital.loss',
             'hours.per.week', 'native.country', 'income']

  categorical = ['workclass', 'education', 'marital.status', 'occupation', 'relationship',
                 'race', 'sex', 'native.country']

  adult_train = pd.read_csv('data/adult.data')
  adult_test = pd.read_csv('data/adult.test')
  adult_train.columns = columns
  adult_test.columns = columns
  train_len = len(adult_train)
  all_data =  [adult_train, adult_test]
  all_data = pd.concat(all_data)

  frames = [all_data]

  for cat in categorical:
    tmp = pd.get_dummies(all_data[cat], prefix=cat)
    frames.append(tmp)

  # Female and Male become cols 64 and 65, respectively

  all_data = pd.concat(frames, axis=1)
  all_data = all_data.drop(columns=categorical)
  all_data['income'] = all_data['income'].map({' <=50K': 0, ' >50K': 1, ' <=50K.': 0, ' >50K.': 1})
  all_data_y = all_data['income'].to_numpy()
  all_data_x = all_data.drop(columns=['income']).to_numpy()
  all_data_x = normalize(all_data_x, True)

  train_x, test_x = all_data_x[:train_len], all_data_x[train_len:]
  train_y, test_y = all_data_y[:train_len], all_data_y[train_len:]

  train_indices = [x for x in range(train_len)]
  random.shuffle(train_indices)
  train_x = train_x[train_indices]
  train_y = train_y[train_indices]

  train_size = math.floor(0.8*train_len)
  val_size = len(train_indices) - train_size

  val_indices = train_indices[0:val_size]
  train_indices = train_indices[val_size:val_size + N]

  data = {}
  data["train_x"] = train_x[train_indices]
  data["train_y"] = train_y[train_indices]
  data["oh_train_y"]= oh_encode(train_y[train_indices])

  data["val_x"] = train_x[val_indices]
  data["val_y"] = train_y[val_indices]
  data["oh_val_y"]= oh_encode(train_y[val_indices])

  data["test_x"] = test_x
  data["test_y"] = test_y
  data["oh_test_y"]= oh_encode(test_y)

  return data

def imshow(example):
  plt.imshow(example.reshape([28, 28]))
  plt.show()