import matplotlib.pyplot as plt
import numpy as np
import pathlib
import json
import seaborn as sns
from datetime import datetime
from helper.misc import infer_and_accuracy, clear_print
from helper.data import load_data, get_architecture
from mip.get_nn import get_nn
from gd.gd_nn import GD_NN

colors = sns.color_palette("Set1", 6)
loss_colors = {
  "max_correct": colors[0],
  "min_hinge": colors[1],
  "sat_margin": colors[2],
  "gd_nn": colors[3],
  "min_w": colors[4],
  "max_m": colors[5]
}


class Exp1():
  def __init__(self, short, show):
    self.losses = ["max_correct", "sat_margin", "min_hinge", "min_w", "max_m", "gd_nn"]
    self.bound = 1
    self.seeds = [821323421,465426341,99413,1436061,7775501]
    self.dataset = "mnist"
    self.num_examples = 100
    self.max_runtime = 2*60
    self.hls = [16]
    self.short = short
    self.show = show

    if short:
      self.seeds = self.seeds[:2]
      self.num_examples = 50
      self.max_runtime = 5*60

    self.max_time_left = len(self.losses)*len(self.seeds)*self.max_runtime

    self.results = {}

  def run_all(self):
    for loss in self.losses:
      self.results[loss] = {}
      json_dir = "results/json/extra_exp"
      pathlib.Path(json_dir).mkdir(parents=True, exist_ok=True)

      json_path = "%s/%s" % (json_dir, loss)
      if self.short:
        json_path += "-short"
      json_path += ".json"

      if pathlib.Path(json_path).is_file():
        print("Path %s exists" % json_path)
        with open(json_path, "r") as f:
          data = json.loads(f.read())
          self.results[loss] = data["results"]
          
      else:
        self.run_nn(loss)
        with open(json_path, "w") as f:
          data = {"results": self.results[loss], "ts": datetime.now().strftime("%d-%m-%H:%M")}
          json.dump(data, f)

  def run_nn(self, loss):
    nn_results = {
      "train_accs": [],
      "val_accs": [],
      "test_accs": [],
      "runtimes": [],
      "objs": []
    }


    self.print_max_time_left()
    N = self.num_examples

    for s in self.seeds:
      clear_print("%s:  N: %s. Seed: %s. Bound: 1." % (loss, N, s))
      data = load_data(self.dataset, N, s)
      arch = get_architecture(data, self.hls)
      if loss == "gd_nn":
        lr = 1e-2
        nn = GD_NN(data, N, arch, lr, self.bound, s, max_epoch=10000)
        nn.train(60*self.max_runtime)
      else:
        nn = get_nn(loss, data, arch, self.bound, 0, "")
        nn.m.setParam('MIPFocus', 0)
        nn.train(60*self.max_runtime, 1)
      obj = nn.get_objective()
      runtime = nn.get_runtime()
      varMatrices = nn.extract_values()
      train_acc = infer_and_accuracy(nn.data["train_x"], nn.data["train_y"], varMatrices, nn.architecture)
      val_acc = infer_and_accuracy(nn.data["val_x"], nn.data["val_y"], varMatrices, nn.architecture)
      test_acc = infer_and_accuracy(nn.data["test_x"], nn.data["test_y"], varMatrices, nn.architecture)

      clear_print("Runtime was: %s" % (runtime))
      print("")

      nn_results["train_accs"].append(train_acc)
      nn_results["val_accs"].append(val_acc)
      nn_results["test_accs"].append(test_acc)
      nn_results["runtimes"].append(runtime)
      nn_results["objs"].append(obj)

      del nn

      self.max_time_left -= self.max_runtime

    self.results[loss] = nn_results

  def plot_results(self):
    settings = ["train_accs", "test_accs", "runtimes"]
    
    print_str = """
    \\begin{table*}[t]
      \\centering
      \\captionsetup{justification=centering}
      \\begin{tabular}{llll}
      \\toprule
      & Training Accuracy \\% & Testing Accuracy \\% & Runtime [s]\\\\
      \\midrule
    """
    table_results = []
    for loss in self.losses:
      add_res = [loss.replace("_", "\_")]
      for setting in settings:
        add_res += [np.mean(self.results[loss][setting]), np.std(self.results[loss][setting])]
      add_str = "%s & %.1f $\\pm$ %.1f & %.1f $\\pm$ %.1f & %.1f $\\pm$ %.1f \\\\" % tuple(add_res)
      table_results.append(add_str)
      print_str += add_str + "\n"


    print_str += """
      \\bottomrule
      \\end{tabular}
      \\caption{Comparing to Icarte's MIP models on MNIST with 100 samples.}
      \\label{tab:exp2}
      \\end{table*}
    """ % table_results


    print(print_str)


  def print_max_time_left(self):
    time_left = self.max_time_left
    days = time_left // (60*24)
    time_left -= days*60*24
    hours = time_left // 60
    time_left -= hours*60
    minutes = time_left % 60

    clear_print("Max time left: %s days, %s hours, %s minutes" % (days, hours, minutes))

def get_mean_std(results):
  mean = np.array([np.mean(z) for z in results])
  std = np.array([np.std(z) for z in results])
  return mean, std

def get_plot_settings(setting, max_runtime):
  titles = {
    "train_accs": "Train",
    "test_accs": "Test",
    "runtimes": "Runtime"
  }

  ylabels = {
    "train_accs": "Accuracy %",
    "test_accs": "Accuracy %",
    "runtimes": "Runtime [s]"
  }

  ylims = {
    "train_accs": [60,102],
    "test_accs": [60,102],
    "runtimes": [0, (max_runtime+2*60)*60]
  }

  return titles[setting], ylabels[setting], ylims[setting]