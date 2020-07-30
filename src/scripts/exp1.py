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

colors = sns.color_palette("Set1", 4)
loss_colors = {
  "max_correct": colors[0],
  "min_hinge": colors[1],
  "sat_margin": colors[2],
  "gd_nn": colors[3],
}

bound_styles = {
  1: "-.",
  3: "-",
  7: "--",
  15: ":"
}

class Exp1():
  def __init__(self, short, show):
    self.losses = ["max_correct", "min_hinge", "sat_margin", "gd_nn"]
    self.bounds = [1,3,7,15]
    self.seeds = [821323421,465426341,99413,1436061,7775501]
    self.dataset = "adult"
    self.num_examples = [40,80,120,160,200,240,280]
    self.max_runtime = 10*60
    self.hls = [16]
    self.short = short
    self.show = show

    if short:
      self.bounds = [1,3,7]
      self.seeds = self.seeds[:2]
      self.num_examples = [40,80,120]
      self.max_runtime = 15

    self.max_time_left = len(self.losses)*len(self.num_examples)*len(self.seeds)*len(self.bounds)*self.max_runtime

    self.results = {}

  def run_all(self):
    for loss in self.losses:
      self.results[loss] = {}
      for bound in self.bounds:
        json_dir = "results/json/exp1"
        pathlib.Path(json_dir).mkdir(parents=True, exist_ok=True)

        name = "%s-%s" % (loss, bound)
        json_path = "%s/%s" % (json_dir, name)
        if self.short:
          json_path += "-short"

        if pathlib.Path(json_path).is_file():
          print("Path %s exists" % json_path)
          with open(json_path, "r") as f:
            data = json.loads(f.read())
            self.results[loss][str(bound)] = data["results"]
        else:
          self.run_nn(loss, bound)
          with open(json_path, "w") as f:
            data = {"results": self.results[loss][str(bound)], "ts": datetime.now().strftime("%d-%m-%H:%M")}
            json.dump(data, f)

  def run_nn(self, loss, bound):
    nn_results = {
      "train_accs": {},
      "val_accs": {},
      "test_accs": {},
      "runtimes": {},
      "objs": {}
    }
    for N in self.num_examples:
      nn_results["train_accs"][N] = []
      nn_results["val_accs"][N] = []
      nn_results["test_accs"][N] = []
      nn_results["runtimes"][N] = []
      nn_results["objs"][N] = []
      self.print_max_time_left()

      for s in self.seeds:
        clear_print("%s:  N: %s. Seed: %s. Bound: %s." % (loss, N, s, bound))
        data = load_data(self.dataset, N, s)
        arch = get_architecture(data, self.hls)
        if loss == "gd_nn":
          lr = 1e-2
          nn = GD_NN(data, N, arch, lr, bound, s, max_epoch=10000)
          nn.train(60*self.max_runtime)
        else:
          nn = get_nn(loss, data, arch, bound, 0, "")
          nn.train(60*self.max_runtime, 1)
        obj = nn.get_objective()
        runtime = nn.get_runtime()
        varMatrices = nn.extract_values()
        train_acc = infer_and_accuracy(nn.data["train_x"], nn.data["train_y"], varMatrices, nn.architecture)
        val_acc = infer_and_accuracy(nn.data["val_x"], nn.data["val_y"], varMatrices, nn.architecture)
        test_acc = infer_and_accuracy(nn.data["test_x"], nn.data["test_y"], varMatrices, nn.architecture)

        clear_print("Runtime was: %s" % (runtime))
        print("")

        nn_results["train_accs"][N].append(train_acc)
        nn_results["val_accs"][N].append(val_acc)
        nn_results["test_accs"][N].append(test_acc)
        nn_results["runtimes"][N].append(runtime)
        nn_results["objs"][N].append(obj)

        self.max_time_left -= self.max_runtime

    self.results[loss][str(bound)] = nn_results

  def plot_results(self):
    settings = ["train_accs", "test_accs", "runtimes"]
    for ind, setting in enumerate(settings):
      plt.figure(ind)
      sns.set_style("darkgrid")
      for i, loss in enumerate(self.losses):
        for bound in self.bounds:
          x = self.num_examples
          y,err = get_mean_std(self.results[loss][str(bound)][setting].values())
          plt.plot(x,y, label="%s-%s" % (loss,bound), color=loss_colors[loss], ls=bound_styles[bound])

      plt.legend()
      plt.xlabel("Number of examples")
      title, ylabel, ylim  = get_plot_settings(setting, self.max_runtime)
      plt.title(title)
      plt.ylabel(ylabel)
      plt.ylim(ylim)

      plot_dir = "results/plots/exp1"
      plot_dir = "%s/%s" % (plot_dir, setting)
      pathlib.Path(plot_dir).mkdir(parents=True, exist_ok=True)
      plot_name = ""
      if self.short:
        plot_name = "short"
      plot_name = "%s_TS:%s" % (plot_name, datetime.now().strftime("%d-%m-%H:%M"))

      if not self.show:
        plt.savefig("%s/%s.png" % (plot_dir, plot_name), bbox_inches="tight")

    if self.show:
      plt.show()

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
    "train_accs": [0,105],
    "test_accs": [0,105],
    "runtimes": [0, (max_runtime+2*60)*60]
  }

  return titles[setting], ylabels[setting], ylims[setting]