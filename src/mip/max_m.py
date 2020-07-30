import gurobipy as gp
import numpy as np
from mip.mip_nn import MIP_NN
from globals import EPSILON, CONT, GUROBI_ENV, LOG

class MAX_M(MIP_NN):
  def __init__(self, data, architecture, bound, reg, fair):

    model = gp.Model("Gurobi_NN", env=GUROBI_ENV)
    if not LOG:
      model.setParam("OutputFlag", 0)

    self.N = len(data["train_x"])
    self.architecture = architecture
    self.data = data
    self.train_x = data["train_x"]
    self.oh_train_y = data["oh_train_y"]

    self.bound = bound
    self.reg = reg
    self.fair = fair

    self.m = model

    self.init_params()
    self.add_margins()
    self.add_examples()
    self.add_output_constraints()
    self.calc_objective()
    self.cutoff = 0

  def add_margins(self):
    self.margins = {}

    for lastLayer, neurons_out in enumerate(self.architecture[1:]):
      layer = lastLayer + 1

      self.margins[layer] = np.full(neurons_out, None)

      for j in range(neurons_out):
        self.margins[layer][j] = self.add_var(CONT,"margin_%s-%s" % (layer,j), lb=0)

  def add_examples(self):

    for lastLayer, neurons_out in enumerate(self.architecture[1:]):
      layer = lastLayer + 1
      neurons_in = self.architecture[lastLayer]
      
      for k in range(self.N):
        for j in range(neurons_out):
          inputs = []
          for i in range(neurons_in):
            if layer == 1:
              inputs.append(self.train_x[k,i]*self.weights[layer][i,j])
            else:
              self.add_constraint(self.var_c[layer][k,i,j] - self.weights[layer][i,j] + 2*self.bound*self.act[lastLayer][k,i] <= 2*self.bound)
              self.add_constraint(self.var_c[layer][k,i,j] + self.weights[layer][i,j] - 2*self.bound*self.act[lastLayer][k,i] <= 0*self.bound)
              self.add_constraint(self.var_c[layer][k,i,j] - self.weights[layer][i,j] - 2*self.bound*self.act[lastLayer][k,i] >= -2*self.bound)
              self.add_constraint(self.var_c[layer][k,i,j] + self.weights[layer][i,j] + 2*self.bound*self.act[lastLayer][k,i] >= 0*self.bound)
              inputs.append(self.var_c[layer][k,i,j])
          pre_activation = sum(inputs) + self.biases[layer][j]

          if layer < len(self.architecture) - 1:
            self.add_constraint((self.act[layer][k,j] == 1) >> (pre_activation >= self.margins[layer][j]))
            self.add_constraint((self.act[layer][k,j] == 0) >> (pre_activation <= -EPSILON - self.margins[layer][j]))

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
          self.add_constraint(pre_activation >= self.margins[layer][j])
        else:
          self.add_constraint(pre_activation <= -EPSILON - self.margins[layer][j])

  def calc_objective(self):
    self.obj = 0
    for layer in self.margins:
      self.obj += self.margins[layer].sum()

    self.set_objective(sense="max")

  def extract_values(self, get_func=lambda z: z.x):
    varMatrices = MIP_NN.extract_values(self, get_func)
    for layer in self.margins:
      varMatrices["margins_%s" % layer] = self.get_val(self.margins[layer], get_func)

    return varMatrices