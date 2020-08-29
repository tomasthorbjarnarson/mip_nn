import matplotlib.pyplot as plt
import pathlib
import json
import seaborn as sns
import numpy as np
from datetime import datetime
from helper.misc import infer_and_accuracy, clear_print
from helper.data import load_data, get_architecture
from mip.get_nn import get_nn

colors = sns.color_palette("Set1", 4)
reg_colors = {
  "0": colors[0],
  "0.1": colors[1],
  "0.01": colors[2],
  "1": colors[3]
}

class Exp2():
  def __init__(self, short, show):
    self.loss = "sat_margin"
    self.regs = [0, 0.1, 0.01]
    self.seeds = [821323421]#,465426341,99413,1436061,7775501]
    self.dataset = "adult"
    self.num_examples = 800
    self.max_runtime = 10*60
    self.bound = 15
    self.hls = [16]
    self.short = short
    self.show = show

    if short:
      self.seeds = self.seeds[:3]
      self.regs = [0, 0.1, 0.01]
      self.hls = [16]
      self.num_examples = 500
      self.max_runtime = 5

    self.max_time_left = len(self.seeds)*len(self.regs)*self.max_runtime

    self.results = {}

  def run_all(self):
    self.results[self.loss] = {}
    for reg in self.regs:
      json_dir = "results/json/exp2"
      pathlib.Path(json_dir).mkdir(parents=True, exist_ok=True)

      name = "%s-%s" % (self.loss, reg)
      json_path = "%s/%s" % (json_dir, name)
      if self.short:
        json_path += "-short"
      json_path += ".json"

      self.json_path = json_path

      seeds = self.seeds
      if pathlib.Path(json_path).is_file():
        print("Path %s exists" % json_path)
        with open(json_path, "r") as f:
          data = json.loads(f.read())
          self.results[self.loss][str(reg)] = data["results"]
          seeds = [s for s in self.seeds if s not in data["seeds"]]
          if data["N"] != self.num_examples:
            raise Exception("Not the correct file for %s" % self.num_examples)
      if len(seeds) != 0:
        self.run_nn(self.loss, reg, seeds)
        with open(json_path, "w") as f:
          data = {
            "results": self.results[self.loss][str(reg)],
            "ts": datetime.now().strftime("%d-%m-%H:%M"),
            "N": self.num_examples,
            "seeds": seeds
          }
          json.dump(data, f)

  def run_nn(self, loss, reg, seeds):
    if str(reg) not in self.results[loss]:
      nn_results = {
        "train_accs": {},
        "val_accs": {},
        "test_accs": {},
        "runtimes": {},
        "objs": {},
        "HL": {}
      }
    else:
      nn_results = self.results[loss][str(reg)]

    N = self.num_examples
    self.print_max_time_left()

    tmp_seeds = [s for s in self.seeds if s not in seeds]
    print("tmp_seeds", tmp_seeds)

    for s in seeds:
      tmp_seeds.append(s)
      clear_print("%s:  N: %s. Seed: %s. Reg: %s." % (loss, N, s, reg))
      data = load_data(self.dataset, N, s)
      arch = get_architecture(data, self.hls)
      nn = get_nn(loss, data, arch, self.bound, reg, "")
      nn.train(60*self.max_runtime, 1)
      obj = nn.get_objective()
      runtime = nn.get_runtime()
      varMatrices = nn.extract_values()
      train_acc = infer_and_accuracy(nn.data["train_x"], nn.data["train_y"], varMatrices, nn.architecture)
      val_acc = infer_and_accuracy(nn.data["val_x"], nn.data["val_y"], varMatrices, nn.architecture)
      test_acc = infer_and_accuracy(nn.data["test_x"], nn.data["test_y"], varMatrices, nn.architecture)

      clear_print("Runtime was: %s" % (runtime))
      print("")

      #nn_results["train_accs"][N].append(train_acc)
      #nn_results["val_accs"][N].append(val_acc)
      #nn_results["test_accs"][N].append(test_acc)
      #nn_results["runtimes"][N].append(runtime)
      #nn_results["objs"][N].append(obj)

      nn_results["train_accs"][s] = train_acc
      nn_results["val_accs"][s] = val_acc
      nn_results["test_accs"][s] = test_acc
      nn_results["runtimes"][s] = runtime
      nn_results["objs"][s] = obj

      if reg:
        hl = [int(v.sum()) for (k,v) in varMatrices.items() if "H_" in k]
        nn_results["HL"][s] = sum(hl)
      else:
        nn_results["HL"][s] = sum(arch[1:-1])

      self.max_time_left -= self.max_runtime

      self.results[loss][str(reg)] = nn_results
      with open(self.json_path, "w") as f:
        data = {
          "results": self.results[self.loss][str(reg)],
          "ts": datetime.now().strftime("%d-%m-%H:%M"),
          "N": N,
          "seeds": tmp_seeds
        }
        json.dump(data, f)


  def plot_results(self):
    self.print_results()
    return
    plt.rcParams.update({'font.size': 16})
    settings = ["train_accs", "test_accs", "runtimes"]
    markers = ['o', 'v', '+', '*']
    for ind, setting in enumerate(settings):
      plt.figure(ind, figsize=(8,6))
      sns.set_style("darkgrid")
      for i,reg in enumerate(self.regs):
        x = self.results[self.loss][str(reg)]["HL"]
        y = list(self.results[self.loss][str(reg)][setting].values())
        plt.scatter(x,y, label=get_reg_label(reg), color=reg_colors[str(reg)], marker=markers[i])

      plt.legend()
      plt.xlabel("Number of neurons in hidden layer")
      title, ylabel, ylim  = get_plot_settings(setting, self.max_runtime)
      plt.title(title)
      plt.ylabel(ylabel)
      plt.ylim(ylim)

      plot_dir = "results/plots/exp2"
      plot_dir = "%s/%s" % (plot_dir, setting)
      pathlib.Path(plot_dir).mkdir(parents=True, exist_ok=True)
      plot_name = "%s-%s" % (self.loss, reg)
      if self.short:
        plot_name += "-short"
      plot_name = "%s_TS:%s" % (plot_name, datetime.now().strftime("%d-%m-%H:%M"))

      if not self.show:
        plt.savefig("%s/%s.png" % (plot_dir, plot_name), bbox_inches="tight")

    if self.show:
      plt.show()

  def print_results(self):
    settings = ["train_accs", "test_accs", "runtimes"]
    for ind, setting in enumerate(settings):
      for i,reg in enumerate(self.regs):
        hls = self.results[self.loss][str(reg)]["HL"]
        res = list(self.results[self.loss][str(reg)][setting].values())
        print(setting, reg, np.mean(res), np.std(res), np.mean(hls), np.std(hls))

  def print_max_time_left(self):
    time_left = self.max_time_left
    days = time_left // (60*24)
    time_left -= days*60*24
    hours = time_left // 60
    time_left -= hours*60
    minutes = time_left % 60

    clear_print("Max time left: %s days, %s hours, %s minutes" % (days, hours, minutes))

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

def get_reg_label(reg):
  if reg == 0:
    return "No model compression"
  elif reg == -1:
    return "Hierarchical optimization"
  else:
    return "Weighted optimization, alpha=%s" % reg