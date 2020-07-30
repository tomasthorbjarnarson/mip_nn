import numpy as np
from mip.mip_nn import MIP_NN
from globals import BIN, CONT, TARGET_ERROR, MARGIN, EPSILON

class SAT_MARGIN(MIP_NN):
  def __init__(self, data, architecture, bound, reg, fair):

    MIP_NN.__init__(self, data, architecture, bound, reg, fair)

    # Cutoff set so as not too optimize fully.
    self.cutoff = self.N*TARGET_ERROR
    self.init_output()
    if reg:
      self.add_regularizer()
    if reg == -1:
      self.calc_multi_obj()
    else:
      self.calc_objective()
    self.add_output_constraints()

  def init_output(self):
    self.output = np.full((self.N, self.architecture[-1]), None)
    for k in range(self.N):
      for j in range(self.architecture[-1]):
        self.output[k,j] = self.add_var(BIN, name="output_%s-%s" % (j,k))

  def add_output_constraints(self):
    layer = len(self.architecture) - 1
    lastLayer = layer - 1
    neurons_in = self.architecture[lastLayer]
    neurons_out = self.architecture[layer]

    if self.reg:
      self.margin = self.add_var(CONT, name="margin", bound=self.out_bound)
      self.new_out_bound = (self.H[len(self.architecture)-2].sum() + 1)*self.bound
      self.add_constraint(self.margin >= 0.5*self.new_out_bound/2)

    for k in range(self.N):
      for j in range(neurons_out):
        inputs = []
        for i in range(neurons_in):
          if layer == 1:
            inputs.append(self.train_x[k,i]*self.weights[layer][i,j])
          else:
            inputs.append(self.var_c[layer][k,i,j])
        pre_activation = sum(inputs) + self.biases[layer][j]
        if self.reg:
          self.add_constraint((self.output[k,j] == 1) >> (pre_activation*self.data['oh_train_y'][k,j] >= self.margin))
          self.add_constraint((self.output[k,j] == 0) >> (pre_activation*self.data['oh_train_y'][k,j] <= self.margin - EPSILON))
        else: 
          pre_activation = 2*pre_activation/self.out_bound
          # HYPERPARAMETER 0.5
          self.add_constraint((self.output[k,j] == 1) >> (pre_activation*self.data['oh_train_y'][k,j] >= MARGIN))
          self.add_constraint((self.output[k,j] == 0) >> (pre_activation*self.data['oh_train_y'][k,j] <= MARGIN - EPSILON))

  def calc_objective(self):
    self.obj = np.prod(self.output.shape) - self.output.sum()

    if self.reg:
      for layer in self.H:
        self.obj += self.reg*self.H[layer].sum()

    self.set_objective()
    

  def calc_multi_obj(self):
    self.obj = np.prod(self.output.shape) - self.output.sum()
    self.m.setObjectiveN(self.obj, 0, 2)
    #self.m.ObjNAbsTol = self.cutoff

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
