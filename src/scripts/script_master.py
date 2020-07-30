import matplotlib.pyplot as plt
import numpy as np
import pathlib
import json
import seaborn as sns
from datetime import datetime
from helper.misc import infer_and_accuracy, clear_print, inference
from helper.data import load_data, get_architecture
from helper.fairness import equalized_odds, demographic_parity
from mip.get_nn import get_nn

focus = 1

#add_prec = lambda x: ["%s-%s" % (x,i) for i in [1,3,7,15,31]]
#all_possible_losses = [add_prec(loss) for loss in list(milps.keys())+list(gds.keys())]
colors = sns.color_palette("Set1", 6 + 3)
loss_colors = {
  "max_correct": colors[0],
  "min_hinge": colors[1],
  "sat_margin": colors[2],
  "gd_nn": colors[3],
  "min_w": colors[4],
  "max_m": colors[6]
}
#loss_colors = dict(zip(list(milps.keys())+list(gds.keys()), colors))


class Script_Master():
  def __init__(self, script_name, losses, dataset, num_examples, max_runtime,
               seeds, hls, bounds, regs=[], fair=False, lr=1e-3, show=True):
    self.script_name = script_name
    self.losses = losses

    self.dataset = dataset
    self.num_examples = num_examples
    self.max_runtime = max_runtime
    self.seeds = seeds
    self.bounds = bounds
    self.regs = regs
    self.loss_colors = loss_colors
    self.fair = fair

    ok = False
    if len(bounds) == 1:
      self.bound = bounds[0]
      ok = True
    if len(losses) == 1 and len(bounds) > 1:
      self.losses = []
      new_colors = sns.color_palette("bright", len(bounds))
      i = 0
      for bound in bounds:
        loss_name = "%s-bound=%s" % (losses[0], bound)
        self.losses.append(loss_name)
        self.loss_colors[loss_name] = new_colors[i]
        i += 1
      ok = True
    if len(losses) == 1 and len(regs) > 0:
      self.bound = bounds[0]
      self.losses = []
      new_colors = sns.color_palette("bright", len(regs))
      i = 0
      for reg in regs:
        loss_name = "%s-reg=%s" % (losses[0],reg)
        self.losses.append(loss_name)
        self.loss_colors[loss_name] = new_colors[i]
        i += 1
      ok = True
    if len(losses) == 1 and fair:
      self.bound = bounds[0]
      self.losses = []
      new_colors = sns.color_palette("bright", 3)
      i = 0
      for f in ["", "EO", "DP"]:
        loss_name = "%s-fair=%s" % (losses[0], f)
        self.losses.append(loss_name)
        self.loss_colors[loss_name] = new_colors[i]
        i += 1


    if not ok:
      raise Exception("Losses %s, Bounds %s and Regs %s incompatible" % (losses,bounds,regs))

    self.lr = lr
    self.show = show

    self.results = {}
    self.json_names = {}
    self.plot_names = {}
    self.hls = {}

    for hl in hls:
      hl_key = '-'.join([str(x) for x in hl])
      self.results[hl_key] = {}
      self.json_names[hl_key] = {}
      self.hls[hl_key] = hl

    self.max_time_left = len(self.losses)*len(num_examples)*len(seeds)*len(hls)*max_runtime

    for hl_key in self.json_names:
      if len(bounds) == 1:
        name = "Time:%s_HLs:%s_|S|:%s_Prec:%s" % (max_runtime, hl_key, len(seeds), self.bound)
      else:
        name = "Time:%s_HLs:%s_|S|:%s_|Prec|:%s" % (max_runtime, hl_key, len(seeds), len(bounds))

      for loss in self.losses:
        self.json_names[hl_key][loss] = "%s-%s" % (loss, name)
      self.plot_names[hl_key] = name

    self.json_dir = "results/json/%s_%s" % (script_name, dataset)
    self.plot_dir = "results/plots/%s_%s" % (script_name, dataset)
    pathlib.Path(self.json_dir).mkdir(parents=True, exist_ok=True)
    pathlib.Path(self.plot_dir).mkdir(parents=True, exist_ok=True)


  def run_all(self):
    for hl_key in self.json_names:
      for loss in self.losses:
        json_path = "%s/%s.json" % (self.json_dir, self.json_names[hl_key][loss])
        if pathlib.Path(json_path).is_file():
          print("Path %s exists" % json_path)
          with open(json_path, "r") as f:
            data = json.loads(f.read())
            self.results[hl_key][loss] = data["results"]
          self.max_time_left -= len(self.seeds)*self.max_runtime
        else:
          self.run_experiment(hl_key, loss)
          with open(json_path, "w") as f:
            data = {"results": self.results[hl_key][loss], "ts": datetime.now().strftime("%d-%m-%H:%M")}
            json.dump(data, f)

  def plot_all(self):
    settings = ["train_accs", "test_accs", "runtimes"]
    i = 1
    for hl_key in self.results:
      for setting in settings:
        self.plot_results(hl_key, setting, i)
        i += 1

  def run_experiment(self, hl_key, loss):
    nn_results = {
      "train_accs": {},
      "val_accs": {},
      "test_accs": {},
      "runtimes": {},
      "objs": {}
    }
    og_loss = loss
    # If there are multiple precisions for the same loss
    if "-bound=" in og_loss:
      loss, bound = og_loss.split("-bound=")
      bound = int(bound)
    else:
      bound = self.bound

    if "-reg=" in og_loss:
      loss,reg = og_loss.split("-reg=")
      reg = float(reg)
    else:
      reg = 0

    if "-fair=" in og_loss:
      loss, fair = og_loss.split("-fair=")
    else:
      fair = ""

    # Make a copy
    num_examples = list(self.num_examples)
    for N in num_examples:
      nn_results["train_accs"][N] = []
      nn_results["val_accs"][N] = []
      nn_results["test_accs"][N] = []
      nn_results["runtimes"][N] = []
      nn_results["objs"][N] = []
      nn_results["HL"] = []
      nn_results["train_EO"] = []
      nn_results["test_EO"] = []
      nn_results["train_DP"] = []
      nn_results["test_DP"] = []
      optimal_reached = []
      self.print_max_time_left()
      for s in self.seeds:
        clear_print("%s:  HLs: %s. N: %s. Seed: %s. Bound: %s. Reg: %s. Fair: %s" % (loss, hl_key, N, s, bound, reg, fair))
        data = load_data(self.dataset, N, s)
        arch = get_architecture(data, self.hls[hl_key])
        nn = get_nn(loss, data, N, arch, bound, reg, fair, s)
        nn.train(60*self.max_runtime, focus)
        obj = nn.get_objective()
        runtime = nn.get_runtime()
        varMatrices = nn.extract_values()
        train_acc = infer_and_accuracy(nn.data["train_x"], nn.data["train_y"], varMatrices, nn.architecture)
        val_acc = infer_and_accuracy(nn.data["val_x"], nn.data["val_y"], varMatrices, nn.architecture)
        test_acc = infer_and_accuracy(nn.data["test_x"], nn.data["test_y"], varMatrices, nn.architecture)

        optimal_reached.append(obj <= nn.cutoff)
        clear_print("Runtime was: %s" % (runtime))
        print("")

        nn_results["train_accs"][N].append(train_acc)
        nn_results["val_accs"][N].append(val_acc)
        nn_results["test_accs"][N].append(test_acc)
        nn_results["runtimes"][N].append(runtime)
        nn_results["objs"][N].append(obj)
        if reg:
          hl = [int(v.sum()) for (k,v) in varMatrices.items() if "H_" in k]
          nn_results["HL"].append(sum(hl))
        else:
          nn_results["HL"].append(sum(arch[1:-1]))

        if self.fair:
          labels_train = np.array(inference(data['train_x'], varMatrices, arch))
          labels_test = np.array(inference(data['test_x'], varMatrices, arch))
          tr_p111, tr_p101, tr_p110, tr_p100 = equalized_odds(data['train_x'], labels_train, data['train_y'])
          tst_p111, tst_p101, tst_p110, tst_p100 = equalized_odds(data['test_x'], labels_test, data['test_y'])

          nn_results["train_EO"].append({'p111': tr_p111, 'p101': tr_p101, 'p110': tr_p110, 'p100': tr_p100})
          nn_results["test_EO"].append({'p111': tst_p111, 'p101': tst_p101, 'p110': tst_p110, 'p100': tst_p100})

          tr_p11, tr_p10 = demographic_parity(data['train_x'], labels_train, data['train_y'])
          tst_p11, tst_p10 = demographic_parity(data['test_x'], labels_test, data['test_y'])

          nn_results["train_DP"].append({'p11': tr_p11, 'p10': tr_p10})
          nn_results["test_DP"].append({'p11': tst_p11, 'p10': tst_p10})

        self.max_time_left -= self.max_runtime

      if self.script_name == "push" and any(optimal_reached):
        num_examples.append(N+num_examples[0])
        if self.max_time_left <= 0:
          self.max_time_left = self.max_runtime*len(self.seeds)

    self.results[hl_key][og_loss] = nn_results

  def plot_results(self, hl_key, setting, index):
    plt.figure(index)
    sns.set_style("darkgrid")
    for i,loss in enumerate(self.losses):
      loss_label = loss
      if "-bound=" in loss_label:
        loss_label = "P=%s" % loss_label.split("-bound=")[-1]
      if loss_label == "min_w":
        loss_label = "min-weight"
      if loss_label == "max_m":
        loss_label = "max-margin"
      if "_" in loss_label:
        loss_label = loss_label.replace("_","-")
      x = [int(z) for z in self.results[hl_key][loss][setting].keys()]
      y, err = get_mean_std(self.results[hl_key][loss][setting].values())
      plt.plot(x,y, label=loss_label, color = loss_colors[loss])
      plt.fill_between(x, y - err, y + err, alpha=0.3, facecolor=loss_colors[loss])

    if len(self.bounds) == 1:
      title = "%s for %s dataset with bound %s" % (self.get_plot_title(setting), self.dataset, self.bound)
    else:
      title = "%s for %s dataset with different bounds" % (self.get_plot_title(setting), self.dataset)

    plt.legend()
    plt.xlabel("Number of examples")
    plt.ylabel(self.get_plot_ylabel(setting))
    plt.title(self.get_plot_title(setting))
    plt.ylim(self.get_plot_ylim(setting))

    plot_dir = "%s/%s" % (self.plot_dir, setting)
    pathlib.Path(plot_dir).mkdir(parents=True, exist_ok=True)
    file_name = self.plot_names[hl_key]
    title = "%s_TS:%s" % (file_name, datetime.now().strftime("%d-%m-%H:%M"))
    if self.show:
      plt.show()
    else:
      plt.savefig("%s/%s.png" % (plot_dir,title),  bbox_inches='tight')

  def subplot_results(self):
    settings = ["train_accs", "test_accs", "runtimes"]   
    j = 0
    for hl_key in self.results:
      fig, axs = plt.subplots(2,2)
      axs = axs.flatten()
      for setting in settings:
        for i,loss in enumerate(self.losses):
          x = [int(z) for z in self.results[hl_key][loss][setting].keys()]
          y, err = get_mean_std(self.results[hl_key][loss][setting].values())
          axs[j].plot(x,y, label=loss, color = loss_colors[loss])
          axs[j].fill_between(x, y - err, y + err, alpha=0.3, facecolor=loss_colors[loss])
        axs[j].set_xlabel("Number of examples")
        axs[j].set_ylabel(self.get_plot_ylabel(setting))
        axs[j].set_title(self.get_plot_title(setting))
        axs[j].set_ylim(self.get_plot_ylim(setting))
        j += 1

      handles, labels = axs[0].get_legend_handles_labels()
      axs[-1].axis('off')
      axs[-1].legend(handles, labels, loc='upper left')

      plot_dir = "%s/%s" % (self.plot_dir, setting)
      pathlib.Path(plot_dir).mkdir(parents=True, exist_ok=True)
      file_name = self.plot_names[hl_key]
      title = "%s_TS:%s" % (file_name, datetime.now().strftime("%d-%m-%H:%M"))
      if self.show:
        plt.show()
      else:
        plt.savefig("%s/%s.png" % (plot_dir,title),  bbox_inches='tight')

  def plot_reg_results(self):
    def get_reg_label(reg):
      if reg == 0:
        return "No model compression"
      elif reg == -1:
        return "Hierarchical optimization"
      else:
        return "Weighted optimization, alpha=%s" % reg
    settings = ["train_accs", "test_accs", "runtimes"]
    markers = ['o', 'v', '+', '*', 'P', '^', 'v'][0:len(self.losses)]
    j = 0
    sns.set_style("darkgrid")

    for hl_key in self.results:
      #fig, axs = plt.subplots(2,2, figsize=(12,10))
      #axs = axs.flatten()
      for setting in settings:
        plt.figure(j)
        for i,loss in enumerate(self.losses):
          _,reg = loss.split("-reg=")
          reg = float(reg)
          x = self.results[hl_key][loss]["HL"]
          y = list(self.results[hl_key][loss][setting].values())
          #axs[j].scatter(x,y, label=get_reg_label(reg), color=loss_colors[loss], marker=markers[i])
          plt.scatter(x,y, label=get_reg_label(reg), color=loss_colors[loss], marker=markers[i])
          #if reg == 0 and setting == "test_accs":
          #  x = [0,np.max(x)]
          #  y = [np.min(y),np.min(y)]
          #  #axs[j].plot(x,y, color=loss_colors[loss], linestyle="--")
          #  plt.plot(x,y, color=loss_colors[loss], linestyle="--")
        #axs[j].set_xlabel("Number of neurons in hidden layer(s)")
        #axs[j].set_ylabel(self.get_plot_ylabel(setting))
        #axs[j].set_title(self.get_plot_title(setting))
        #axs[j].set_ylim(self.get_plot_ylim(setting))
        plt.xlabel("Number of neurons in hidden layer(s)")
        plt.ylabel(self.get_plot_ylabel(setting))
        plt.title(self.get_plot_title(setting))
        plt.ylim(self.get_plot_ylim(setting))
        plt.legend(labels=[get_reg_label(0), get_reg_label(-1), get_reg_label(1), get_reg_label(0.1)], loc="lower center")#, bbox_to_anchor=(0,0,1,1))

        j += 1
      #handles, labels = axs[0].get_legend_handles_labels()
      #handles, labels = plt.get_legend_handles_labels()
      #axs[-1].axis('off')
      #axs[-1].legend(handles, labels, loc='upper left')
      #plot_dir = "%s" % (self.plot_dir)
        plot_dir = "%s/%s" % (self.plot_dir, setting)
        pathlib.Path(plot_dir).mkdir(parents=True, exist_ok=True)
        file_name = self.plot_names[hl_key]
        title = "%s_TS:%s" % (file_name, datetime.now().strftime("%d-%m-%H:%M"))
        if self.show:
          plt.show()
        else:
          plt.savefig("%s/%s.png" % (plot_dir,title),  bbox_inches='tight')

  def visualize_fairness(self, fair_type):
    small_between = "----------\n"
    between = "===============\n"
    EO_str = "P111: {p111:.3f}. P101: {p101:.3f}. P110: {p110:.3f}. P100: {p100:.3f}.\n"
    DP_str = "P11: {p11:.3f}. P10: {p10:.3f}.\n"
    if fair_type == "EO":
      format_str = EO_str
    elif fair_type == "DP":
      format_str = DP_str
    else:
      raise Exception("Unkown fairness definition: %s" % fair_type)
    print_str = between
    for hl_key in self.results:
      all_bars = []
      for i,loss in enumerate(self.losses):
        _,fair = loss.split("-fair=")
        if fair == "":
          fair = "None"
        if fair == fair_type or fair == "None":
          loss_result = self.results[hl_key][loss]
          ex = self.num_examples[0]
          if str(ex) in loss_result["train_accs"]:
            ex = str(ex)
          train = np.mean(loss_result["train_accs"][ex])
          train_std = np.std(loss_result["train_accs"][ex])
          test = np.mean(loss_result["test_accs"][ex])
          test_std = np.std(loss_result["test_accs"][ex])
          run = np.mean(loss_result["runtimes"][ex])
          run_std = np.std(loss_result["runtimes"][ex])
          print_str += "Fairness definition: %s.\n" % fair
          print_str += "Train acc: %.3f +/- %.3f.\n" % (train, train_std)
          print_str += "Test acc: %.3f +/- %.3f.\n" % (test, test_std)
          print_str += "Runtime: %.3f +/- %.3f.\n" % (run, run_std)
          print_str += small_between

          train_fair = self.results[hl_key][loss]["train_%s" % fair_type]
          test_fair = self.results[hl_key][loss]["test_%s" % fair_type]
          if fair_type == "EO":
            plt.figure()
            width = 0.35
            bars = []
            for seed,_ in enumerate(train_fair):
              tr_seed = train_fair[seed]
              tst_seed = test_fair[seed]
              tmp = (np.abs(tr_seed['p111'] - tr_seed['p101']), np.abs(tst_seed['p111'] - tst_seed['p101']),"true", fair)
              tmp2 = (np.abs(tr_seed['p110'] - tr_seed['p100']), np.abs(tst_seed['p110'] - tst_seed['p100']),"false", fair)
              bars.append(tmp)
              bars.append(tmp2)
            all_bars.append(bars)
            tr_true_pos_diffs = [np.abs(x['p111'] - x['p101']) for x in train_fair]
            tr_false_pos_diffs = [np.abs(x['p110'] - x['p100']) for x in train_fair]
            tst_true_pos_diffs = [np.abs(x['p111'] - x['p101']) for x in test_fair]
            tst_false_pos_diffs = [np.abs(x['p110'] - x['p100']) for x in test_fair]
            ind = np.arange(len(tr_true_pos_diffs)*2)
            plt.bar(ind-width/2, tr_true_pos_diffs+tr_false_pos_diffs, width, label="Train")
            plt.bar(ind+width/2, tst_true_pos_diffs+tst_false_pos_diffs, width, label="Test")
            
            label1 = "Seed %s True Positives"
            label2 = "Seed %s False Positives"
            plt.xticks(ind-0.7, (label1 % 1, label1 % 2, label1 % 3, label2 % 1, label2 %2, label2 % 3), rotation=15)
            plt.legend()
            plt.title("Fairness constraint: %s " % fair)
            #plt.show()
          else:
            tr_pos = [np.abs(x['p11'] - x['p10']) for x in train_fair]
            tst_pos = [np.abs(x['p11'] - x['p10']) for x in test_fair]
          for j, seed in enumerate(self.seeds):
            print_str += "Seed: %s.\n" % seed
            print_str += "Train: " + format_str.format(**train_fair[j])
            print_str += "Test: " + format_str.format(**test_fair[j])
            if j != len(self.seeds)-1:
              print_str += small_between
          print_str += between

    print(print_str)

    #true_pos_diffs = [np.abs(x['p111'] - x['p101']) for x in self.results[hl_key][loss]["train_%s" % fair_type]]
    #false_pos_diffs = [np.abs(x['p110'] - x['p100']) for x in self.results[hl_key][loss]["train_%s" % fair_type]]
    #all_bars = list(zip(all_bars[0], all_bars[1]))
    #s1 = np.array(all_bars[0:2])
    #s2 = all_bars[2:4]
    #s3 = all_bars[4:6]
    #s1_bars = np.array(s1)[:,:,:2].flatten().astype(np.float)
    #colors = ["green" if i % 2 == 0 else "red" for i,_ in enumerate(s1_bars)]
    #plt.figure()
    #plt.bar(range(len(s1_bars)), s1_bars, color=colors) 
    #plt.show()

  def make_tables(self):

    from pdb import set_trace

    settings = ["train_accs", "test_accs", "runtimes"]

    def get_losses_name(loss):
      losses_names = {
        "min_w": "Min-weight",
        "max_m": "Max-margin",
        "max_correct": "Max-correct",
        "min_hinge": "Min-hinge",
        "sat_margin": "Sat-margin",
        "gd_nn": "GD baseline"
      }

      if loss in losses_names:
        return losses_names[loss]
      elif "-bound=" in loss:
        loss, P = loss.split("-bound=")
        if loss in losses_names:
          return "%s, $P=%s$" % (losses_names[loss], P)
      elif "-reg=" in loss:
        loss, reg = loss.split("-reg=")
        if loss in losses_names:
          return "%s, $reg=%s$" % (losses_names[loss], reg)



    def gen_table_start(setting, num_examples):
      ex_len = len(num_examples)
      table_start = """\\begin{table}[H]
      \\footnotesize
      \\centering
      \\begin{tabular}{ll"""

      if setting == "acc":
        cols = 2
      else:
        cols = 1
      num_cols = cols*ex_len
      table_start += "l"*num_cols + "}\n"

      if setting == "acc":
        table_start += """\\toprule
        {} & {} & \\multicolumn{%s}{c}{Training Accuracy [\\%%]} & \\multicolumn{%s}{c}{Testing Accuracy [\\%%]} \\\\
        \\cmidrule(lr){3-%s}
        \\cmidrule(lr){%s-%s}
        """ % (ex_len, ex_len, 3+ex_len-1,3+ex_len,3+2*ex_len-1)
      elif setting in ["train", "test", "time"]:
        #table_start += """\\toprule
        #{} & {} & \\multicolumn{%s}{c}{Runtime [s]} \\\\
        #\\cmidrule(lr){3-%s}
        #""" % (ex_len, 3+ex_len-1)
        table_start += "\\toprule \n"

      table_start += "{} & Samples &" + " & ".join([(" & ".join(str(x) for x in num_examples))]*cols) + "\\\\ \n \\midrule \n"

      return table_start


    for hl_key in self.results:
      tr_table = gen_table_start("train", self.num_examples)
      tst_table = gen_table_start("test", self.num_examples)
      time_table = gen_table_start("time", self.num_examples)
      for i,seed in enumerate(self.seeds):
        start_multi = True
        tr_table += "\\multirow{%s}{*}{Seed %s} & " % (len(self.losses),i+1)
        tst_table += "\\multirow{%s}{*}{Seed %s} & " % (len(self.losses),i+1)
        time_table += "\\multirow{%s}{*}{Seed %s} & " % (len(self.losses),i+1)
        for loss in self.losses:
          if start_multi:
            tr_table += get_losses_name(loss)
            tst_table += get_losses_name(loss)
            time_table += get_losses_name(loss)
            start_multi = False
          else:
            tr_table += "{} & " + get_losses_name(loss)
            tst_table += "{} & " + get_losses_name(loss)
            time_table += "{} & " + get_losses_name(loss)
          train_res = self.results[hl_key][loss]["train_accs"]
          test_res = self.results[hl_key][loss]["test_accs"]
          time_res = self.results[hl_key][loss]["runtimes"]
          for ex in train_res:
            tr_table += " & %.1f" % train_res[ex][i]
          for ex in test_res:
            tst_table += " & %.1f" % test_res[ex][i]
          for ex in time_res:
            time_table += " & %.1f" % time_res[ex][i]
          #set_trace()
          tr_table += "\\\\ \n"
          tst_table += "\\\\ \n"
          time_table += "\\\\ \n"
        if i < len(self.seeds)-1:
          tr_table += "\\midrule \n"
          tst_table += "\\midrule \n"
          time_table += "\\midrule \n"
      tr_table += "\\bottomrule \n \\end{tabular} \n \\caption{Training accuracy [\\%%] on %s dataset - One hidden layer} \n \\label{Foo} \n \\end{table}" % (self.dataset)
      tst_table += "\\bottomrule \n \\end{tabular} \n \\caption{Testing accuracy [\\%%] on %s dataset - One hidden layer} \n \\label{Foo} \n \\end{table}" % (self.dataset)
      time_table += "\\bottomrule \n \\end{tabular} \n \\caption{Runtime [s] on %s dataset - One hidden layer} \n \\label{Foo} \n \\end{table}" % (self.dataset)
      with open("tablestuff-%s.txt" % hl_key, "w") as f:
        f.write(tr_table + "\n\n\n" + tst_table + "\n\n\n" + time_table)
      set_trace()

    
    #""Models & \\multicolumn{%s}{c}{Seed 1} & \\multicolumn{%s}{c}{Seed 2} & \\multicolumn{%s}{c}{Seed 3} \\\\
    # \\cmidrule(lr){2-%s}
    # \\cmidrule(lr){%s-%s}
    # \\cmidrule(lr){%s-%s}
    # """ % (ex_len, ex_len, ex_len, 2+ex_len-1,2+ex_len,2+2*ex_len-1,2+2*ex_len,2+3*ex_len-1)
    



  def print_max_time_left(self):
    time_left = self.max_time_left
    days = time_left // (60*24)
    time_left -= days*60*24
    hours = time_left // 60
    time_left -= hours*60
    minutes = time_left % 60

    clear_print("Max time left: %s days, %s hours, %s minutes" % (days, hours, minutes))

  def get_plot_title(self, setting):
    titles = {
      "train_accs": "Train",
      "test_accs": "Test",
      "runtimes": "Runtime"
    }
    return titles[setting]

  def get_plot_ylabel(self, setting):
    ylabels = {
      "train_accs": "Accuracy %",
      "test_accs": "Accuracy %",
      "runtimes": "Runtime [s]"
    }
    return ylabels[setting]

  def get_plot_ylim(self, setting):
    ylims = {
      "train_accs": [0,105],
      "test_accs": [0,105],
      "runtimes": [0, (self.max_runtime+2*60)*60]
    }
    return ylims[setting]

def get_mean_std(results):
  mean = np.array([np.mean(z) for z in results])
  std = np.array([np.std(z) for z in results])
  return mean, std



    #ratio = 8
    #f, (ax,ax2) = plt.subplots(2,1,gridspec_kw={'height_ratios': [ratio, 1]})

    #plt.subplots_adjust(wspace=0, hspace=0)
      #if setting in ['test_accs']:
      #  ax.plot(x,y, label=loss, color = loss_colors[loss])
      #  ax2.plot(x,y, label=loss, color = loss_colors[loss])
      #  ax.fill_between(x, y - err, y + err, alpha=0.3, facecolor=loss_colors[loss])
      #  ax2.fill_between(x, y - err, y + err, alpha=0.3, facecolor=loss_colors[loss])
      #  ax.set_ylim(60,100)
      #  ax2.set_ylim(0,5)
      #  ax2.set_yticks([0,5])
      #  ax.spines['bottom'].set_visible(False)
      #  ax2.spines['top'].set_visible(False)
      #  ax.xaxis.tick_top()
      #  ax.tick_params(labeltop=False)  # don't put tick labels at the top
      #  ax2.xaxis.tick_bottom()
      #  d = 0.015
      #  kwargs = dict(transform=ax.transAxes, color='k', clip_on=False)
      #  #break_x = np.array([0,0,0.01,-0.01,0.01,-0.01,0,0])+0.01
      #  #break_y = np.array([0,0.03,0.06,0.09,0.12,0.15,0.18,0.21])
      #  #ax.plot(break_x, break_y, **kwargs)
      #  ax.plot((-d, +d), (-d, +d), **kwargs) # top-left diagonal
      #  ax.plot((1 - d, 1 + d), (-d, +d), **kwargs) # top-right diagonal
      #  kwargs.update(transform=ax2.transAxes)
      #  tmp = 1
      #  ax2.plot((-d, +d), (1 - ratio*d, 1 + ratio*d), **kwargs) # bottom-left diagonal
      #  ax2.plot((1 - d, 1 + d), (1 - ratio*d, 1 + ratio*d), **kwargs) # bottom-right diagonal
