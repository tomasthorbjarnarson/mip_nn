import tensorflow as tf
import numpy as np

from gd.gd_nn import GD_NN
from helper.data import load_data, get_architecture
from helper.misc import infer_and_accuracy, clear_print

tf.logging.set_verbosity(tf.logging.ERROR)


def test_stuff():
  random_seed = 31567478618
  tf.set_random_seed(random_seed)
  #seed = 5369
  seed = random_seed
  N = 25000
  data = load_data("adult", N, seed)
  #data = load_data("mnist", N, seed)

  hls = [16]

  architecture = get_architecture(data, hls)
  lr = 1e-1
  bound = 1
  time = 60
  batch_size = 100

  print_str = "Architecture: %s. N: %s. LR: %s. Bound: %s"
  clear_print(print_str % ("-".join([str(x) for x in architecture]), N, lr, bound))


  #nn = BNN(data, N, architecture, lr, seed)
  nn = GD_NN(data, N, architecture, lr, bound, seed, batch_size)
  nn.train(max_time=time*60)
  nn_y_pred = nn.y_pred.eval(session=nn.sess, feed_dict={nn.x: data['train_x']})
  #nn_loss = nn.loss.eval(session=nn.sess, feed_dict={nn.x: nn.X_train, nn.y: nn.oh_y_train})
  nn_loss = nn.get_objective()
  print("nn_loss", nn_loss)
  nn_runtime = nn.get_runtime()
  print("nn_runtime", nn_runtime)
  varMatrices = nn.extract_values()

  train_acc = infer_and_accuracy(data['train_x'], data['train_y'], varMatrices, architecture)
  test_acc = infer_and_accuracy(data['test_x'], data['test_y'], varMatrices, architecture)
  print("train_acc", train_acc)
  print("test_acc", test_acc)


  loss = np.square(np.maximum(0, 0.5 - nn_y_pred*data['oh_train_y'])).sum()
  print("loss", loss)

  w1 = varMatrices['w_1']
  b1 = varMatrices['b_1']
  w2 = varMatrices['w_2']
  b2 = varMatrices['b_2']

  x = data['test_x']
  y = data['test_y']
  foo = np.dot(x, w1) + b1
  bar = 1/(1+np.exp(-foo))
  tmp = np.dot(bar, w2) + b2
  acc = np.equal(np.argmax(tmp, 1), y).sum()/len(y)

  from pdb import set_trace
  set_trace()

def batch_train():
  N = 25000
  hls = [16]
  epochs = 10000
  
  lr = 1e-1
  bound = 15
  time = 60
  batch_size = 100

  train_accs = []
  test_accs = []
  times = []

  seeds = [1348612,7864568,9434861,3618393,93218484358]
  for seed in seeds:
    tf.set_random_seed(seed)
    data = load_data("adult", N, seed)
    architecture = get_architecture(data, hls)

    print_str = "Architecture: %s. N: %s. LR: %s. Bound: %s. Seed: %s."
    clear_print(print_str % ("-".join([str(x) for x in architecture]), N, lr, bound, seed))
    nn = GD_NN(data, N, architecture, lr, bound, seed, batch_size)
    nn.train(max_time=time*60)

    nn_runtime = nn.get_runtime()
    varMatrices = nn.extract_values()

    train_acc = infer_and_accuracy(data['train_x'], data['train_y'], varMatrices, architecture)
    test_acc = infer_and_accuracy(data['test_x'], data['test_y'], varMatrices, architecture)
    print("train_acc", train_acc)
    print("test_acc", test_acc)

    train_accs.append(train_acc)
    test_accs.append(test_acc)
    times.append(nn_runtime)

  clear_print("Train: %s +/- %s" % (np.mean(train_accs), np.std(train_accs)))
  clear_print("Test: %s +/- %s" % (np.mean(test_accs), np.std(test_accs)))
  clear_print("Time: %s +/- %s" % (np.mean(times), np.std(times)))

if __name__ == '__main__':
  #batch_train()
  test_stuff()