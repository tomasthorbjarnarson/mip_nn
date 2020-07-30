import numpy as np
from mip.mip_nn import MIP_NN
from globals import EPSILON, BIN

class MIN_W(MIP_NN):
  def __init__(self, data, architecture, bound, reg, fair):

    MIP_NN.__init__(self, data, architecture, bound, reg, fair)
    self.add_abs_params()
    self.add_output_constraints()
    self.calc_objective()
    self.cutoff = 0

  def add_abs_params(self):
    self.abs_weights = {}
    self.abs_biases = {}

    for lastLayer, neurons_out in enumerate(self.architecture[1:]):
      layer = lastLayer + 1
      neurons_in = self.architecture[lastLayer]

      self.abs_weights[layer] = np.full((neurons_in, neurons_out), None)
      self.abs_biases[layer] = np.full(neurons_out, None)

      for j in range(neurons_out):
        for i in range(neurons_in):
          if layer == 1 and self.dead[i]:
            # Dead inputs should have 0 weight and therefore 0 absolute weight
            self.abs_weights[layer][i,j] = 0
          else:
            self.abs_weights[layer][i,j] = self.add_var(BIN,"abs(w)_%s-%s_%s" % (layer,i,j))
            self.add_constraint(self.abs_weights[layer][i,j] >= self.weights[layer][i,j])
            self.add_constraint(-self.abs_weights[layer][i,j] <= self.weights[layer][i,j])
        # Bias only for each output neuron
        self.abs_biases[layer][j] = self.add_var(BIN,"abs(b)_%s-%s" % (layer,j))
        self.add_constraint(self.abs_biases[layer][j] >= self.biases[layer][j])
        self.add_constraint(-self.abs_biases[layer][j] <= self.biases[layer][j])

  def add_output_constraints(self):
    layer = len(self.architecture) - 1
    lastLayer = layer - 1
    neurons_in = self.architecture[lastLayer]
    neurons_out = self.architecture[layer]

    for k in range(self.N):
      for j in range(neurons_out):
        inputs = []
        for i in range(neurons_in):
          if layer == 1:
            inputs.append(self.train_x[k,i]*self.weights[layer][i,j])
          else:
            inputs.append(self.var_c[layer][k,i,j])
        pre_activation = sum(inputs) + self.biases[layer][j]
        if self.oh_train_y[k,j] > 0:
          self.add_constraint(pre_activation >= 0)
        else:
          self.add_constraint(pre_activation <= -EPSILON)

  def calc_objective(self):
    self.obj = 0
    for layer in self.abs_weights:
      self.obj += self.abs_weights[layer].sum()
    for layer in self.abs_biases:
      self.obj += self.abs_biases[layer].sum()

    self.set_objective()
