import numpy as np
from mip.mip_nn import MIP_NN
from globals import CONT, TARGET_ERROR, MARGIN

class MIN_HINGE(MIP_NN):
  def __init__(self, data, architecture, bound, reg, fair):

    MIP_NN.__init__(self, data, architecture, bound, reg, fair)

    self.init_output()
    self.add_output_constraints()
    if reg:
      self.add_regularizer()
    self.calc_objective()
    # Cutoff set so as not too optimize fully.
    # Target: >=90% accuracy. 0.25 = (0.5-0)^2 (see objective function)
    self.cutoff = self.N*TARGET_ERROR*MARGIN*MARGIN

  def init_output(self):
    self.output = np.full((self.N, self.architecture[-1]), None)
    for k in range(self.N):
      for j in range(self.architecture[-1]):
        self.output[k,j] = self.add_var(CONT, bound=self.out_bound, name="output_%s-%s" % (j,k))

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
        # Approximately normalize to between -1 and 1
        pre_activation = 2*pre_activation/self.out_bound
        self.add_constraint(self.output[k,j] == pre_activation*self.oh_train_y[k,j])

  def calc_objective(self):
    def hinge(u):
      return np.square(np.maximum(0, (MARGIN - u)))
    npts = int(2*self.out_bound+1)
    lb = -1
    ub = 1
    ptu = []
    pthinge = []
    for i in range(npts):
      ptu.append(lb + (ub - lb) * i / (npts-1))
      pthinge.append(hinge(ptu[i]))

    if self.reg and len(self.architecture) > 2:
      alpha = 1/sum(self.architecture[1:-1])
      self.obj = 0
      for layer in self.H:
        self.obj += self.H[layer].sum()*alpha
      self.add_constraint(self.obj >= alpha*self.data['oh_train_y'].shape[1])

      self.set_objective()

    for k in range(self.N):
      for j in range(self.architecture[-1]):
        self.m.setPWLObj(self.output[k,j], ptu, pthinge)

  def extract_values(self, get_func=lambda z: z.x):
    varMatrices = MIP_NN.extract_values(self, get_func)
    varMatrices["output"] = self.get_val(self.output, get_func)

    if self.reg:
      for layer in self.H:
        varMatrices["H_%s" % layer] = self.get_val(self.H[layer], get_func)

    return varMatrices
