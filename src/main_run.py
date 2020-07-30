from mip.get_nn import get_nn
from helper.misc import inference, infer_and_accuracy, clear_print, get_network_size,strip_network
from helper.data import load_data, get_architecture
from helper.fairness import equalized_odds, demographic_parity
from gd.gd_nn import GD_NN
import argparse
import numpy as np
from pdb import set_trace


if __name__ == '__main__':

  parser = argparse.ArgumentParser(description='Optional app description')

  parser.add_argument('--hls', default='16', type=str)
  parser.add_argument('--ex', default=10, type=int)
  parser.add_argument('--focus', default=0, type=int)
  parser.add_argument('--time', default=1, type=float)
  parser.add_argument('--seed', default=0, type=int)
  parser.add_argument('--loss', default="min_w", type=str)
  parser.add_argument('--data', default="mnist", type=str)
  parser.add_argument('--bound', default=1, type=int)
  parser.add_argument('--fair', default="", type=str)
  parser.add_argument('--reg', default=0, type=float)
  parser.add_argument('--save', action='store_true', help="An optional flag to save data")
  args = parser.parse_args()
  
  hls = [int(x) for x in args.hls.split("-") if len(args.hls) > 0]

  N = args.ex
  focus = args.focus
  train_time = args.time
  seed = args.seed
  loss = args.loss
  data_name = args.data
  bound = args.bound
  reg = args.reg
  fair = args.fair

  print(args)

  data = load_data(data_name, N, seed)

  architecture = get_architecture(data, hls)
  print("architecture", architecture)

  print_str = "Architecture: %s. N: %s. Loss: %s. Bound: %s"
  clear_print(print_str % ("-".join([str(x) for x in architecture]), N, loss, bound))

  if loss == "gd_nn":
    lr = 1e-2
    nn = GD_NN(data, N, architecture, lr, bound, seed)
    nn.train(60*train_time)
  else:
    nn = get_nn(loss, data, architecture, bound, reg, fair)
    nn.train(train_time*60, focus)

  obj = nn.get_objective()
  print("Objective value: ", obj)

  varMatrices = nn.extract_values()

  train_acc = infer_and_accuracy(nn.data['train_x'], nn.data["train_y"], varMatrices, nn.architecture)
  test_acc = infer_and_accuracy(nn.data['test_x'], nn.data["test_y"], varMatrices, nn.architecture)

  print("Training accuracy: %s " % (train_acc))
  print("Testing accuracy: %s " % (test_acc))


  w1 = varMatrices['w_1']
  b1 = varMatrices['b_1']
  if len(architecture) > 2:
    w2 = varMatrices['w_2']
    b2 = varMatrices['b_2']
    train = nn.data['train_x']

    tmp_inf = np.dot(train, w1) + b1
    tmp_inf[tmp_inf >= 0] = 1
    tmp_inf[tmp_inf < 0] = -1
    inf = np.dot(tmp_inf, varMatrices['w_2']) + varMatrices['b_2']
    norm = 2*inf / ((hls[0]+1)*bound)

  net_size = get_network_size(architecture, bound)
  print("Network memory: %s Bytes" % net_size)

  stripped,new_arch = strip_network(varMatrices, architecture)
  new_net_size = get_network_size(new_arch, bound)
  if new_net_size != net_size:
    print("New Network memory: %s Bytes" % new_net_size)

    stripped_train_acc = infer_and_accuracy(nn.data['train_x'], nn.data["train_y"], stripped, new_arch)
    stripped_test_acc = infer_and_accuracy(nn.data['test_x'], nn.data["test_y"], stripped, new_arch)

    print("Stripped Training accuracy: %s " % (stripped_train_acc))
    print("Stripped Testing accuracy: %s " % (stripped_test_acc))

  if fair:
    female_train = data['train_x'][:,64]
    male_train = data['train_x'][:,65]
    labels_train = np.array(inference(data['train_x'], varMatrices, architecture))
    female_perc_train = (female_train*labels_train).sum() / labels_train.sum()
    print("female_perc_train", female_perc_train)
    male_perc_train = (male_train*labels_train).sum() / labels_train.sum()
    print("male_perc_train", male_perc_train)

    female_test = data['test_x'][:,64]
    male_test = data['test_x'][:,65]
    labels_test = np.array(inference(data['test_x'], varMatrices, architecture))

  if fair == "EO":

    clear_print("Equalized Odds:")

    tr_p111, tr_p101, tr_p110, tr_p100 = equalized_odds(data['train_x'], labels_train, data['train_y'])
    
    print("train_p111: %.3f" % (tr_p111))
    print("train_p101: %.3f" % (tr_p101))
    print("train_p110: %.3f" % (tr_p110))
    print("train_p100: %.3f" % (tr_p100))
    p111, p101, p110, p100 = equalized_odds(data['test_x'], labels_test, data['test_y'])

    print("test_p111: %.3f" % (p111))
    print("test_p101: %.3f" % (p101))
    print("test_p110: %.3f" % (p110))
    print("test_p100: %.3f" % (p100))

    print("NN p111: %.3f" % (nn.female_pred1_true1.getValue()))
    print("NN p101: %.3f" % (nn.male_pred1_true1.getValue()))
    print("NN p110: %.3f" % (nn.female_pred1_true0.getValue()))
    print("NN p100: %.3f" % (nn.male_pred1_true0.getValue()))

  elif fair == "DP":

    clear_print("Demographic Parity:")

    tr_p11, tr_p10 = demographic_parity(data['train_x'], labels_train, data['train_y'])
    print("train_p11: %.3f" % (tr_p11))
    print("train_p10: %.3f" % (tr_p10))

    p11, p10 = demographic_parity(data['test_x'], labels_test, data['test_y'])
    print("test_p11: %.3f" % (p11))
    print("test_p10: %.3f" % (p10))

    print("NN p11: %.3f" % (nn.female_pred1.getValue()))
    print("NN p10: %.3f" % (nn.male_pred1.getValue()))

  set_trace()
