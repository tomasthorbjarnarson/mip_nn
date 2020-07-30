from time import time
import tensorflow as tf
import numpy as np
from globals import TARGET_ERROR, MARGIN
"""
Ported from Icarte's formulation from https://bitbucket.org/RToroIcarte/bnn/
Modified to allow for any integer precision, i.e. 1 (binary) or 3,7,15 etc
"""
class GD_NN:
  def __init__(self, data, N, architecture, lr, bound, seed=1, batch_size=0, max_epoch=10000000000):
    tf.set_random_seed(seed)
    self.sess = tf.Session()
    self.lr = lr
    self.bound = bound
    self.architecture = architecture
    if batch_size == 0:
      self.batch_size = N
    else:
      self.batch_size = batch_size
    self.N = N
    self.max_epoch = max_epoch

    self.cutoff = N*TARGET_ERROR*MARGIN*MARGIN
    print("cutoff", self.cutoff)
    self.out_bound = (architecture[-2]+1)*bound

    self.runtime = 0

    self.data = data

    self.x = tf.placeholder("float", [None, architecture[0]], name="x")
    self.y = tf.placeholder("float", [None, architecture[-1]], name="y")

    self.train_dict = {self.x: self.data['train_x'], self.y: self.data['oh_train_y']}
    self.val_dict = {self.x: self.data['val_x'], self.y: self.data['oh_val_y']}

    self.weights = []
    self.biases = []

    x = self.x
    for layer, n_in in enumerate(architecture[0:-1]):
      n_hl = architecture[layer+1]
      w = self.step_unit(self.weight_variable([n_in, n_hl]))
      b = self.step_unit(self.bias_variable([n_hl]))
      x = tf.add(tf.matmul(x, w), b)
      if layer < len(architecture) - 2:
        x = self.binary_tanh_unit(x + 0.001)
      self.weights.append(w)
      self.biases.append(b)

    self.y_pred = tf.divide(tf.multiply(x,2),self.out_bound)

    self.loss = tf.reduce_sum(tf.square(tf.math.maximum(0., MARGIN - tf.multiply(self.y,self.y_pred))))
    correct_prediction = tf.equal(tf.argmax(self.y_pred, 1), tf.argmax(self.y, 1))
    self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

    global_step = tf.Variable(0, trainable=False)
    learn = tf.compat.v1.train.exponential_decay(lr, global_step, 1000, 0.9)
    self.optimizer = tf.train.AdamOptimizer(learning_rate=learn).minimize(self.loss, global_step=global_step)

  def train(self, max_time, focus=0):
    start_time = time()
    self.sess.run(tf.global_variables_initializer())
    loss_count = 500
    losses = np.random.rand(loss_count)
    loss_i = 0
    epoch = 0
    finished = False

    while not finished:
      for start,end in zip(range(0,self.N,self.batch_size), range(self.batch_size, self.N+1,self.batch_size)):
        batch_dict = {self.x: self.data['train_x'][start:end], self.y: self.data['oh_train_y'][start:end]}
        _, loss = self.sess.run([self.optimizer, self.loss], feed_dict=batch_dict)

      loss = self.loss.eval(session=self.sess, feed_dict = self.train_dict)
      end_time = time()
      losses[loss_i] = loss
      loss_i = (loss_i + 1) % loss_count
      finished = loss <= self.cutoff or np.std(losses) <= 1e-5 or (end_time - start_time) >= max_time or epoch >= self.max_epoch
      if epoch % 5*(25000/self.N) == 0 or finished:
        train_perf = self.sess.run(self.accuracy, feed_dict = self.train_dict)
        val_perf = self.sess.run(self.accuracy, feed_dict = self.val_dict)
        print("Epoch: %s. Train: %% %.2f. Val: %% %.2f. Loss: %s. Std: %s" % (epoch, 100*train_perf, 100*val_perf, loss, np.std(losses)))

      epoch += 1

    end_time = time()
    self.runtime = end_time - start_time

  def get_objective(self):
    loss = self.sess.run(self.loss, feed_dict=self.train_dict)
    return float(loss)

  def get_runtime(self):
    return self.runtime

  def extract_values(self):
    varMatrices = {}
    for layer, w in enumerate(self.weights):
      varMatrices["w_%s" % (layer + 1)] = w.eval(session = self.sess)
    for layer, b in enumerate(self.biases):
      varMatrices["b_%s" % (layer + 1)] = b.eval(session = self.sess)
    return varMatrices

  def weight_variable(self, shape):
    initial = tf.truncated_normal(shape, stddev=self.bound/2)
    #initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

  def bias_variable(self, shape):
    initial = tf.truncated_normal(shape, stddev=self.bound/2)
    #initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

  def hard_sigmoid(self, x):
    return tf.clip_by_value((x + 1.)/2., 0, 1)

  def round_through(self, x):
    rounded = tf.round(x)
    return x + tf.stop_gradient(rounded-x)

  def binary_tanh_unit(self, x):
    return 2.*self.round_through(self.hard_sigmoid(x))-1

  def step_unit(self, x):
    if self.bound < 1:
      return x
    return self.round_through(tf.clip_by_value(x, -self.bound, self.bound))
