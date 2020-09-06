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
  1: "-",
  3: "--",
  7: "-.",
  15: ":"
}

alphas = {
  1: 0.2,
  3: 0.3,
  7: 0.4,
  15: 0.5
}

class Exp1():
  def __init__(self, short, show):
    self.losses = ["max_correct", "sat_margin", "gd_nn", "min_hinge"]
    self.bounds = [1,3,7,15]
    self.seeds = [821323421,465426341,99413,1436061,7775501]
    self.dataset = "adult"
    self.num_examples = [40,80,120,160,200,240,280]
    self.max_runtime = 10*60
    self.hls = [16]
    self.short = short
    self.show = show

    if short:
      self.bounds = [1,3,7, 15]
      self.seeds = self.seeds[:3]
      self.num_examples = [50,100,150,200]
      self.max_runtime = 30

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

        num_examples = self.num_examples
        if pathlib.Path(json_path).is_file():
          print("Path %s exists" % json_path)
          with open(json_path, "r") as f:
            data = json.loads(f.read())
            self.results[loss][str(bound)] = data["results"]
            num_examples = [i for i in self.num_examples if str(i) not in data["results"]["train_accs"].keys()]
            print("num_examples", num_examples)
            
        if len(num_examples) != 0:
          self.run_nn(loss, bound, num_examples)
          with open(json_path, "w") as f:
            data = {"results": self.results[loss][str(bound)], "ts": datetime.now().strftime("%d-%m-%H:%M")}
            json.dump(data, f)

  def run_nn(self, loss, bound, num_examples):
    if self.results[loss] == {}:
      nn_results = {
        "train_accs": {},
        "val_accs": {},
        "test_accs": {},
        "runtimes": {},
        "objs": {}
      }
    else:
      nn_results = self.results[loss][str(bound)]

    for N in num_examples:
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

        del nn

        self.max_time_left -= self.max_runtime

    self.results[loss][str(bound)] = nn_results

  def plot_results(self):
    settings = ["train_accs", "test_accs", "runtimes"]
    sns.set_style("darkgrid")
    plt.rcParams.update({'font.size': 12})
    for ind, setting in enumerate(settings):
      if setting == "train_accs" or setting == "test_accs":
        self.subplot(ind, setting)
      else:
        lines, labels = self.regular_plot(ind, setting)

    plt.rcParams.update({'font.size': 16})
    legendfig = plt.figure(ind+1, figsize=(8,6))
    legendfig.legend(lines[0:4], labels[0:4], 'upper left', bbox_to_anchor=(0.2,1.0))#, ncol=len(self.losses)//2)
    legendfig.legend(lines[4:8], labels[4:8], 'upper left', bbox_to_anchor=(0.2,0.7))#, ncol=len(self.losses)//2)
    legendfig.legend(lines[8:12], labels[8:12], 'upper left', bbox_to_anchor=(0.6,1.0))#, ncol=len(self.losses)//2)
    legendfig.legend(lines[12:16], labels[12:16], 'upper left', bbox_to_anchor=(0.6,0.7))#, ncol=len(self.losses)//2)

    if not self.show:
      legendfig.savefig("results/plots/exp1/legend.png")

    if self.show:
      plt.show()

  def regular_plot(self, ind, setting):
    plt.figure(ind, figsize=(8,6), dpi=100)
    lines = []
    labels = []
    for i, loss in enumerate(self.losses):
      for bound in self.bounds:
        x = self.num_examples
        #y,err = get_mean_std(self.results[loss][str(bound)][setting].values())
        y,err = get_mean_std([self.results[loss][str(bound)][setting][str(x)] for x in self.num_examples])
        label = "%s P=%s" % (loss,bound)
        lines += plt.plot(x,y, label=label, color=loss_colors[loss], ls=bound_styles[bound])
        plt.fill_between(x, y - err, y + err, alpha=alphas[bound], facecolor=loss_colors[loss])
        labels.append(label)

    #plt.legend(ncol=len(self.losses), loc="upper left", bbox_to_anchor=(1, 0.5))
    plt.xlabel("Number of examples")
    title, ylabel, ylim  = get_plot_settings(setting, self.max_runtime)
    #plt.title(title)
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

    return lines, labels

  def subplot(self, ind, setting):
    ratio = 4
    f, (ax,ax2) = plt.subplots(2,1, num=ind, sharex=True, figsize=(8,6), dpi=100, gridspec_kw={'height_ratios': [ratio, 1]})
    for i, loss in enumerate(self.losses):
      for bound in self.bounds:
        x = self.num_examples
        #y,err = get_mean_std(self.results[loss][str(bound)][setting].values())
        y,err = get_mean_std([self.results[loss][str(bound)][setting][str(x)] for x in self.num_examples])
        ax.plot(x,y, label="%s-%s" % (loss,bound), color=loss_colors[loss], ls=bound_styles[bound])
        ax.fill_between(x, y - err, y + err, alpha=alphas[bound], facecolor=loss_colors[loss])
        ax2.plot(x,y, label="%s-%s" % (loss,bound), color=loss_colors[loss], ls=bound_styles[bound])


    #ax2.legend(ncol=len(self.losses), loc="lower center")
    plt.xlabel("Number of examples")
    title, ylabel, ylim  = get_plot_settings(setting, self.max_runtime)
    #ax.set_title(title)
    ax.set_ylabel(ylabel)
    if setting in ["train_accs", "test_accs"]:
      ax.set_ylim(58,102)
      ax2.set_ylim(0,12)
    else:
      plt.ylim(ylim)
    ax.spines["bottom"].set_visible(False)
    ax2.spines["top"].set_visible(False)
    ax.xaxis.tick_top()
    ax.tick_params(labeltop=False)
    ax2.xaxis.tick_bottom()
    ax2.set_yticks([0,5,10])

    #f.tight_layout()
    d = .015  # how big to make the diagonal lines in axes coordinates
    # arguments to pass to plot, just so we don't keep repeating them
    kwargs = dict(transform=ax.transAxes, color='k', clip_on=False)
    ax.plot((-d, +d), (-d, +d), **kwargs)        # top-left diagonal
    ax.plot((1 - d, 1 + d), (-d, +d), **kwargs)  # top-right diagonal

    kwargs.update(transform=ax2.transAxes)  # switch to the bottom axes
    ax2.plot((-d, +d), (1 - ratio*d, 1 + ratio*d), **kwargs) # bottom-left diagonal
    ax2.plot((1 - d, 1 + d), (1 - ratio*d, 1 + ratio*d), **kwargs) # bottom-right diagonal

    plt.subplots_adjust(hspace=0.1)

    plot_dir = "results/plots/exp1"
    plot_dir = "%s/%s" % (plot_dir, setting)
    pathlib.Path(plot_dir).mkdir(parents=True, exist_ok=True)
    plot_name = ""
    if self.short:
      plot_name = "short"
    plot_name = "%s_TS:%s" % (plot_name, datetime.now().strftime("%d-%m-%H:%M"))

    if not self.show:
      plt.savefig("%s/%s.png" % (plot_dir, plot_name), bbox_inches="tight")



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