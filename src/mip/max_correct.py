import numpy as np
from mip.mip_nn import MIP_NN
from globals import EPSILON, BIN, TARGET_ERROR

class MAX_CORRECT(MIP_NN):
  def __init__(self, data, architecture, bound, reg, fair):

    MIP_NN.__init__(self, data, architecture, bound, reg, fair)

    self.init_output()
    self.add_output_constraints()
    self.cutoff = self.N*TARGET_ERROR
    #self.calc_objective()
    if reg:
      self.add_regularizer()
    if reg == -1:
      self.calc_multi_obj()
    else:
      self.calc_objective()

  def init_output(self):
    self.output = np.full((self.N, self.architecture[-1]), None)
    for k in range(self.N):
      for j in range(self.architecture[-1]):
        self.output[k,j] = self.add_var(BIN, name="output_%s-%s" % (j,k))
      self.add_constraint(self.output[k,:].sum() == 1)

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
        pre_activation = 2*pre_activation/self.out_bound
        self.add_constraint((self.output[k,j] == 1) >> (pre_activation >= 0))
        self.add_constraint((self.output[k,j] == 0) >> (pre_activation <= -EPSILON))

  def calc_objective(self):
    self.obj = self.N
    for k in range(self.N):
      for j in range(self.architecture[-1]):
        if self.oh_train_y[k,j] > 0:
          self.obj -= self.output[k,j]

    if self.reg:
      for layer in self.H:
        self.obj += self.reg*self.H[layer].sum()

    self.set_objective()

  def calc_multi_obj(self):
    self.obj = self.N
    for k in range(self.N):
      for j in range(self.architecture[-1]):
        if self.oh_train_y[k,j] > 0:
          self.obj -= self.output[k,j]
    self.m.setObjectiveN(self.obj, 0, 2)
    self.m.ObjNAbsTol = self.cutoff

    if self.reg:
      regObj = 0
      for layer in self.H:
        regObj += self.H[layer].sum()
      self.m.setObjectiveN(regObj, 1, 1)

  def extract_values(self, get_func=lambda z: z.x):
    varMatrices = MIP_NN.extract_values(self, get_func)
    varMatrices["output"] = self.get_val(self.output, get_func)

    if self.reg:
      for layer in self.H:
        varMatrices["H_%s" % layer] = self.get_val(self.H[layer], get_func)

    return varMatrices
