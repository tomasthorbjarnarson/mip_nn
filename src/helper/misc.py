import numpy as np
import random
import math
import copy

seed = 0

def infer_and_accuracy(set_x, set_y, varMatrices, architecture):
  inferred = inference(set_x, varMatrices, architecture)
  accuracy = calc_accuracy(inferred, set_y)
  return accuracy

def numpy_sign(varMatrix):
  signVarMatrix = varMatrix
  signVarMatrix[varMatrix >= 0] = 1
  signVarMatrix[varMatrix < 0] = -1
  return signVarMatrix

def inference(set_x, varMatrices, architecture):
  N_test, input_size = np.shape(set_x)
    
  infer = set_x

  for lastLayer, neurons_out in enumerate(architecture[1:]):
    layer = lastLayer + 1
    infer = np.dot(infer,varMatrices["w_%s" % layer])
    infer += varMatrices["b_%s" % layer]
    if layer < len(architecture) - 1:
      infer = numpy_sign(infer)

  output = all_ok(infer)
  #output = all_good(infer)
  return output

def all_good(infer):
  output = []
  for row in infer:
    label = np.argwhere(row >= 0)
    if len(label) == 1 and len(label[0]) == 1:
      label = label[0][0]
    else:
      label = -1
    output.append(label)
  return output

def all_ok(infer):
  random.seed(seed)
  output = []
  for row in infer:
    label = np.argwhere(row == np.max(row))
    label = random.choice(label)[0]
    output.append(label)
  return output

def calc_accuracy(inferred, y):
  acc = 0
  for i, label in enumerate(inferred):
    if label == y[i]:
      acc += 1
  acc = acc/len(y)
  return acc*100

def get_bound_matrix(network_vars, bound):
  """network_vars contains varMatrices of all batches"""
  all_vars = {}
  bound_matrix = {}
  for key in network_vars[0]:
    all_vars[key] = np.stack([tmp[key] for tmp in network_vars])
    if "w_" in key or "b_" in key:
      tmp_min = np.min(all_vars[key],axis=0)
      tmp_max = np.max(all_vars[key],axis=0)
      tmp_min[tmp_min >= 0] = -bound
      tmp_max[tmp_max <= 0] = bound
      bound_matrix["%s_%s" % (key,"lb")] = tmp_min
      bound_matrix["%s_%s" % (key,"ub")] = tmp_max

  return bound_matrix

def get_alt_bound_matrix(network_vars, bound):
  """network_vars contains varMatrices of all batches"""
  all_vars = {}
  bound_matrix = {}
  for key in network_vars[0]:
    all_vars[key] = np.stack([tmp[key] for tmp in network_vars])
    if "w_" in key or "b_" in key:
      bound_matrix["%s_%s" % (key,"lb")] = np.zeros_like(all_vars[key][0]) - bound
      bound_matrix["%s_%s" % (key,"ub")] = np.zeros_like(all_vars[key][0]) + bound
      vars_sum = all_vars[key].sum(axis=0) / all_vars[key].shape[0]
      vars_eq = np.equal(vars_sum, all_vars[key][0])
      bound_matrix["%s_%s" % (key,"lb")][vars_eq] = all_vars[key][0][vars_eq]
      bound_matrix["%s_%s" % (key,"ub")][vars_eq] = all_vars[key][0][vars_eq]

  return bound_matrix

#def get_mean_bound_matrix(network_vars, bound, diff=0):
#  new_bound = int(bound - 2**np.floor(np.log2(bound)))
#  if diff == 0:
#    diff = new_bound
#  mean_vars = get_mean_vars(network_vars)
#  bound_matrix = {}
#  for key in network_vars[0]:
#    if "w_" in key or "b_" in key:
#      bound_matrix["%s_%s" % (key,"lb")] = np.maximum(mean_vars[key] - diff, -bound)
#      bound_matrix["%s_%s" % (key,"ub")] = np.minimum(mean_vars[key] + diff, bound)
#
#  return bound_matrix

def get_mean_bound_matrix(network_vars, bound, diff=0, weights=[]):
  if len(weights) == 0:
    mean_vars = get_from_vars(network_vars, np.mean)
  else:
    mean_vars = get_weighted_mean_vars(network_vars, weights)
  std_vars = get_from_vars(network_vars, np.std)
  print("In get bound matrix")
  bound_matrix = {}
  for key in network_vars[0]:
    if "w_" in key or "b_" in key:
      if diff == 0:
        bound_matrix["%s_%s" % (key,"lb")] = np.maximum(mean_vars[key] - std_vars[key], -bound)
        bound_matrix["%s_%s" % (key,"ub")] = np.minimum(mean_vars[key] + std_vars[key], bound)
      else:
        bound_matrix["%s_%s" % (key,"lb")] = np.maximum(mean_vars[key] - diff, -bound)
        bound_matrix["%s_%s" % (key,"ub")] = np.minimum(mean_vars[key] + diff, bound)

  return bound_matrix

def get_weighted_mean_bound_matrix(network_vars, bound, diff, weighted_avg):
  bound_matrix = {}
  for key in network_vars[0]:
    if "w_" in key or "b_" in key:
      bound_matrix["%s_%s" % (key,"lb")] = np.maximum(weighted_avg[key] - diff, -bound)
      bound_matrix["%s_%s" % (key,"ub")] = np.minimum(weighted_avg[key] + diff, bound)

  return bound_matrix

def get_from_vars(network_vars, get_func=np.mean):
  """network_vars contains varMatrices of all batches"""
  all_vars = {}
  mean_vars = {}

  for key in network_vars[0]:
    all_vars[key] = np.stack([tmp[key] for tmp in network_vars])
    mean_vars[key] = get_func(all_vars[key], axis=0)
    mean_vars[key][mean_vars[key] < 0] -= 1e-5
    mean_vars[key][mean_vars[key] >= 0] += 1e-5
    mean_vars[key] = np.round(mean_vars[key])

  return mean_vars

def get_mean_vars(network_vars):
  """network_vars contains varMatrices of all batches"""
  all_vars = {}
  mean_vars = {}
  for key in network_vars[0]:
    all_vars[key] = np.stack([tmp[key] for tmp in network_vars])
    mean_vars[key] = np.mean(all_vars[key], axis=0)
    mean_vars[key][mean_vars[key] < 0] -= 1e-5
    mean_vars[key][mean_vars[key] >= 0] += 1e-5
    mean_vars[key] = np.round(mean_vars[key])

  return mean_vars

def get_weighted_mean_vars(network_vars, weights):
  weighted_avg = {}
  weights = np.array(weights)
  weights -= np.min(weights)
  weights /= np.max(weights)
  #weights += 1
  for i,var in enumerate(network_vars):
    for key in var:
      if key not in weighted_avg:
        weighted_avg[key] = weights[i]*var[key]
      else:
        weighted_avg[key] += weights[i]*var[key]

  for key in weighted_avg:
      weighted_avg[key] = np.round(weighted_avg[key]/sum(weights))

  return weighted_avg

def get_median_vars(network_vars):
  """network_vars contains varMatrices of all batches"""
  all_vars = {}
  mean_vars = {}
  for key in network_vars[0]:
    all_vars[key] = np.stack([tmp[key] for tmp in network_vars])
    mean_vars[key] = np.median(all_vars[key], axis=0)
    mean_vars[key][mean_vars[key] < 0] -= 1e-5
    mean_vars[key][mean_vars[key] >= 0] += 1e-5
    mean_vars[key] = np.round(mean_vars[key])

  return mean_vars

def clear_print(text):
  print("====================================")
  print(text)
  print("====================================")

# [108-16-2], 3
def get_network_size(architecture, bound):
  bound_bits = math.floor(math.log2(bound))+ 1 + 1# extra + 1 because negative is allowed
  total_weights = 0
  total_biases = 0
  for i, layer_size in enumerate(architecture[1:]):
    last_layer_size = architecture[i]
    total_weights += layer_size*last_layer_size
    total_biases += layer_size

  return (total_biases + total_weights)*bound_bits / 8 # / 8 for bytes instead of bits

def strip_network(varMatrices, arch):
  if 'H_1' not in varMatrices:
    print("Nothing to strip")
    return varMatrices,arch
  stripped = copy.deepcopy(varMatrices)
  new_arch = copy.deepcopy(arch)
  for lastLayer, neurons_out in enumerate(arch[1:-1]):
    layer = lastLayer + 1
    nextLayer = layer+1
    h = np.array([int(v) for v in stripped['H_%s' % layer]])
    h = np.array(h == 1)
    stripped['w_%s' % layer] = stripped['w_%s' % layer][:,h]
    stripped['b_%s' % layer] = stripped['b_%s' % layer][h]
    stripped['w_%s' % nextLayer] = stripped['w_%s' % nextLayer][h,:]
    new_arch[layer] = h.sum()
  return stripped, new_arch


# Print iterations progress
def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end = printEnd)
    # Print New Line on Complete
    if iteration == total: 
        print()


def extract_params(varMatrices, get_keys=["w", "b"]):
  tmp = {}
  keys = []
  for key in varMatrices:
    if key[0] in get_keys:
      keys.append(key)
      tmp[key] = varMatrices[key]
  return tmp, keys