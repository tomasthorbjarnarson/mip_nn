import gurobipy as gp
from gurobipy import GRB
import numpy as np
from helper.misc import infer_and_accuracy, inference
from helper.fairness import demographic_parity
from globals import INT, BIN, CONT, LOG, DO_CUTOFF, GUROBI_ENV

vtypes = {
  INT: GRB.INTEGER,
  BIN: GRB.BINARY,
  CONT: GRB.CONTINUOUS
}

def get_gurobi_nn(NN, data, architecture, bound, reg=False, fair="", batch=False):
  # Init a NN using Gurobi API according to the NN supplied
  class Gurobi_NN(NN):
    def __init__(self, data, architecture, bound, reg, fair, batch):
      model = gp.Model("Gurobi_NN", env=GUROBI_ENV)
      if not LOG:
        model.setParam("OutputFlag", 0)
      NN.__init__(self, model, data, architecture, bound, reg, fair, batch)
      
    def add_var(self, precision, name, bound=None, lb=None, ub=None):
      if precision not in vtypes:
        raise Exception('Parameter precision not known: %s' % precision)

      if precision == BIN:
        return self.m.addVar(vtype=GRB.BINARY, name=name)
      else:
        if not bound:
          if lb != None and ub != None:
            return self.m.addVar(vtype=vtypes[precision], lb=lb, ub=ub, name=name)
          elif lb != None:
            return self.m.addVar(vtype=vtypes[precision], lb=lb, name=name)
          elif ub != None:
            return self.m.addVar(vtype=vtypes[precision], ub=ub, name=name)
        else:
            return self.m.addVar(vtype=vtypes[precision], lb=-bound, ub=bound, name=name)

    def add_constraint(self, constraint):
      self.m.addConstr(constraint)

    def set_objective(self, sense="min"):
      if sense == "min":
        self.m.setObjective(self.obj, GRB.MINIMIZE)
      else:
        self.m.setObjective(self.obj, GRB.MAXIMIZE)
      
    def train(self, time=None, focus=None):
      if time:
        self.m.setParam('TimeLimit', time)
      if focus:
        self.m.setParam('MIPFocus', focus)
      #self.m.setParam('Threads', 1)
      self.m._lastobjbst = GRB.INFINITY
      self.m._lastobjbnd = -GRB.INFINITY
      self.m._progress = []
      self.m._val_acc = 0
      # Needed to access values for NN objects
      self.m._self = self
      self.m.update()
      self.m.optimize(mycallback)

      # Add last value after optimisation finishes
      #gap = 1 - self.m.ObjBound/(self.m.ObjVal + 1e-15)
      #if gap != self.m._progress[-1][4]:
      #  self.m._progress.append((self.m.NodeCount, self.m.ObjVal, self.m.ObjBound, self.m.Runtime, gap))

    def get_objective(self):
      return self.m.ObjVal

    def get_runtime(self):
      return self.m.Runtime

    def get_data(self):
      data = {
        'obj': self.m.ObjVal,
        'bound': self.m.ObjBound,
        'gap': self.m.MIPGap,
        'nodecount': self.m.NodeCount,
        'num_vars': self.m.NumVars,
        'num_int_vars': self.m.NumIntVars - self.m.NumBinVars,
        'num_binary_vars': self.m.NumBinVars,
        'num_constrs': self.m.NumConstrs,
        'num_nonzeros': self.m.NumNZs,
        'periodic': self.m._progress,
        'variables': self.extract_values(),
      }
      return data

    # get_func needed to know how to access the variable
    def get_val(self, maybe_var, get_func):
      tmp = np.zeros(maybe_var.shape)
      for index, count in np.ndenumerate(maybe_var):
        try:
          # Sometimes solvers have "integer" values like 1.000000019, round it to 1
          if maybe_var[index].VType in ['I', 'B']:
            tmp[index] = round(get_func(maybe_var[index]))
          else:
            tmp[index] = get_func(maybe_var[index])
        except:
          tmp[index] = 0
      return tmp

  return Gurobi_NN(data, architecture, bound, reg, fair, batch)


def mycallback(model, where):
  if where == GRB.Callback.MIP:
    nodecnt = model.cbGet(GRB.Callback.MIP_NODCNT)
    objbst = model.cbGet(GRB.Callback.MIP_OBJBST) + 1e-15
    objbnd = model.cbGet(GRB.Callback.MIP_OBJBND)
    runtime = model.cbGet(GRB.Callback.RUNTIME)
    gap = 1 - objbnd/objbst
    if objbst < model._lastobjbst or objbnd > model._lastobjbnd:
      model._lastobjbst = objbst
      model._lastobjbnd = objbnd
      model._progress.append((nodecnt, objbst, objbnd, runtime, gap, model._val_acc))
  elif where == GRB.Callback.MIPSOL:
    nodecnt = model.cbGet(GRB.Callback.MIPSOL_NODCNT)
    objbst = model.cbGet(GRB.Callback.MIPSOL_OBJBST) + 1e-15
    objbnd = model.cbGet(GRB.Callback.MIPSOL_OBJBND)
    runtime = model.cbGet(GRB.Callback.RUNTIME)
    gap = 1 - objbnd/objbst
    model._lastobjbst = objbst
    model._lastobjbnd = objbnd

    data = model._self.data
    architecture = model._self.architecture
    varMatrices = model._self.extract_values(get_func=model.cbGetSolution)
    train_acc = infer_and_accuracy(data['train_x'], data['train_y'], varMatrices, architecture)
    val_acc = infer_and_accuracy(data['val_x'], data['val_y'], varMatrices, architecture)
    if LOG:
      print("Train accuracy: %s " % (train_acc))
      print("Validation accuracy: %s " % (val_acc))
      if model._self.reg:
        for layer in model._self.H:
          hl = varMatrices["H_%s" % layer].sum()
          print("Hidden layer %s length: %s" % (layer, int(hl)))

      labels_train = np.array(inference(data['train_x'], varMatrices, architecture))
      #print(model._self.female_pred1.getValue(), model._self.male_pred1.getValue())
      tr_p11, tr_p10 = demographic_parity(data['train_x'], labels_train, data['train_y'])
      print("train_p11: %.3f" % (tr_p11))
      print("train_p10: %.3f" % (tr_p10))
      print("ratio: %.3f" % np.minimum((tr_p11/tr_p10),(tr_p10/tr_p11)))

    model._progress.append((nodecnt, objbst, objbnd, runtime, gap, val_acc))
    model._val_acc = val_acc

    # ModelSense == 1 makes sure it is minimization
    if DO_CUTOFF and int(objbst) <= model._self.cutoff and model.ModelSense == 1:# and model._self.reg <= 0:
      if model._self.reg == -1:
        #print("Cutoff first optimization from cutoff value: %s" % model._self.cutoff)
        model.cbStopOneMultiObj(0)
      elif model._self.reg > 0:
        hls = 0
        for layer in model._self.H:
          hls += varMatrices["H_%s" % layer].sum()
        if hls == architecture[-1] and int(objbst) - hls <= model._self.cutoff:
          model.terminate()

      else:
        #print("Terminate from cutoff value: %s" % model._self.cutoff)
        model.terminate()

    #cutoff = np.prod(model._output.shape)*np.square(0.1)
    #if model._output[0,0].VType == 'C' and objbst - objbnd < cutoff:
    #  print("Terminate after cutoff value: %s" % cutoff)
    #  model.terminate()
#
    #elif model.Params.TimeLimit and runtime >= 0.5*model.Params.TimeLimit and train_acc >= 98:
    #  print("Terminate after time: %s with train acc: %s" % (runtime, train_acc))
    #  model.terminate()