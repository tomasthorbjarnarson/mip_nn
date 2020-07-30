from helper.NumpyEncoder import NumpyEncoder
from datetime import datetime
import matplotlib.pyplot as plt
import json
import math
import pathlib

def get_file_locations(architecture, num_examples, solver):
  arch_str = '-'.join([str(z) for z in architecture])
  plot_dir = "results/plots/%s/%s/%s" % (arch_str, num_examples, solver)
  json_dir = "results/json/%s/%s/%s" % (arch_str, num_examples, solver)
  pathlib.Path(plot_dir).mkdir(parents=True, exist_ok=True)
  pathlib.Path(json_dir).mkdir(parents=True, exist_ok=True)
  return plot_dir, json_dir

class DataSaver:
  def __init__(self, nn, architecture, num_examples, focus, solver):
    self.nn = nn
    self.architecture = architecture
    self.num_examples = num_examples
    self.focus = focus
    self.time_elapsed = math.floor(nn.get_runtime())

    now = datetime.now()
    nowStr = now.strftime("%d %b %H:%M")
    self.title = '%s_Time:%s-Focus:%s' % (nowStr,self.time_elapsed,focus)
    self.plot_dir, self.json_dir = get_file_locations(architecture, num_examples, solver)

  def plot_periodic(self, data):
    per_filtered = [z for z in data if z[4] < 0.95]
    x = [z[3] for z in per_filtered]
    y = [z[1] for z in per_filtered]
    y2 = [z[2] for z in per_filtered]

    plt.plot(x,y, label="Best objective")
    plt.plot(x,y2, label="Best bound")
    plt.legend()
    plt.xlabel("Time [s]")
    plt.ylabel("Sum of absolute weights")
    plt.title(self.title)
    last_acc = 0
    for sol in per_filtered:
      if sol[5] != last_acc:
        plt.annotate("%.2f" % sol[5],(sol[3], sol[1]))
        last_acc = sol[5]
    plt.savefig("%s/%s.png" % (self.plot_dir, self.title), bbox_inches='tight')
    plt.show()


  def save_json(self, train_acc, test_acc):
    now = datetime.now()
    nowStr = now.strftime("%d/%m/%Y %H:%M:%S")

    data = self.nn.get_data()

    data.update({
      'datetime': nowStr,
      'architecture': self.architecture,
      'num_examples': self.num_examples,
      'time': self.time_elapsed,
      'MIPFocus': self.focus,
      'trainingAcc': train_acc,
      'testingAcc': test_acc,
    })

    with open('%s/%s.json' % (self.json_dir, self.title), 'w') as f:
      json.dump(data, f, cls=NumpyEncoder)