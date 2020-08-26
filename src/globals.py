import gurobipy as gp

ARCHITECTURES = {
  1: [784, 10],
  2: [784, 16, 10],
  3: [784, 16, 16, 10]
}

INT = 0
BIN = 1
CONT = 2

EPSILON = 1e-4

LOG = False

MULTIOBJ = False

TARGET_ERROR = 0.1 # 90% Accuracy

MARGIN = 0.5

DO_CUTOFF = True

GUROBI_ENV = gp.Env()
