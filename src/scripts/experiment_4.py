import numpy as np
import tensorflow as tf
import pathlib
import json
from datetime import datetime
from mip.get_nn import get_nn
import time
from multiprocessing import Pool
import os
from helper.misc import infer_and_accuracy, clear_print, printProgressBar
from helper.misc import extract_params, get_weighted_mean_bound_matrix, get_weighted_mean_vars
from helper.data import load_data, get_architecture, get_training_batches
from helper.NumpyEncoder import NumpyEncoder
from gd.gd_nn import GD_NN

new_seeds = [8353134,14365767666,223454657,9734234,753283393493482349,473056832,3245823,
             3842134,132414364572435,798452456,2132413245,788794342,134457678,213414,69797949393,
             34131413,46658765,1341324]

class Exp4():
  def __init__(self, short, show, loss, dataset, batch_size):
    self.loss = loss
    self.seeds = [821323421,465426341,99413,1436061,7775501]
    self.dataset = dataset
    if dataset == "adult":
      self.num_examples = 25000
    else:
      self.num_examples = 50000
    self.max_runtime = 15
    self.hls = [16]
    self.batch_size = batch_size
    self.bound = 15
    self.epochs = self.bound+1
    self.short = short
    self.show = show

    # Dummy data to load correct architecture
    tmp_data = load_data(self.dataset, 1, 0)
    self.architecture = get_architecture(tmp_data, self.hls)

    if short:
      self.seeds = self.seeds[:2]
      self.num_examples = 2500
      self.bound = 15
      self.epochs = np.minimum(self.bound+1, 3)

    self.num_batches = self.num_examples/self.batch_size

    self.num_processes = len(os.sched_getaffinity(0))//2
    print("Number of processes to run on: %s." % self.num_processes)

    self.results = {}

  def run_all(self):
    
    self.results[self.loss] = {}
    self.results["gd_nn"] = {}
    for s in self.seeds:
      json_dir = "results/json/exp3"
      pathlib.Path(json_dir).mkdir(parents=True, exist_ok=True)

      name = "%s-%s" % (self.loss, s)
      gd_name = "gd_nn-%s" % (s)
      json_path = "%s/%s" % (json_dir, name)
      gd_json_path = "%s/%s" % (json_dir, gd_name)
      if self.short:
        json_path += "-short"
        gd_json_path += "-short"

      if pathlib.Path(json_path).is_file():
        print("Path %s exists" % json_path)
        with open(json_path, "r") as f:
          data = json.loads(f.read())
          self.results[self.loss][str(s)] = data["results"]
      else:
        self.run_nn(self.loss, s)
        with open(json_path, "w") as f:
          data = {"results": self.results[self.loss][str(s)], "ts": datetime.now().strftime("%d-%m-%H:%M")}
          json.dump(data, f, cls=NumpyEncoder)

      if pathlib.Path(gd_json_path).is_file():
        print("Path %s exists" % gd_json_path)
        with open(gd_json_path, "r") as f:
          data = json.loads(f.read())
          self.results["gd_nn"][str(s)] = data["results"]
      else:
        self.run_gd(s)
        with open(gd_json_path, "w") as f:
          data = {"results": self.results["gd_nn"][str(s)], "ts": datetime.now().strftime("%d-%m-%H:%M")}
          json.dump(data, f, cls=NumpyEncoder)

  def run_nn(self, loss, seed):
    N = self.num_examples
    print_str = "Architecture: %s. N: %s. Loss: %s. Bound: %s. Seed: %s."
    clear_print(print_str % ("-".join([str(x) for x in self.architecture]), N, loss, self.bound, seed))

    epoch_start = time.time()
    bound_matrix = {}

    all_avgs = []
    all_train_accs = []
    all_val_accs = []
    all_stds = []

    for epoch in range(self.epochs):
      # Load all data with new seed every epoch
      data = load_data(self.dataset, N, seed+new_seeds[epoch-1])
      # Distribute shuffled data into batches
      batches = get_training_batches(data, self.batch_size)

      clear_print("EPOCH %s" % epoch)
      printProgressBar(0, N/self.batch_size)
      network_vars = self.run_batches(batches, bound_matrix)

      std_vars, total_std = get_std(network_vars)
      print("Total std: %.2f" % total_std)
      all_stds.append(total_std)

      val_accs = []
      for var in network_vars:
        val_acc = infer_and_accuracy(data["val_x"], data["val_y"], var, self.architecture)
        val_accs.append(val_acc)
      #print("val_accs", val_accs)
      weighted_avg = get_weighted_mean_vars(network_vars, val_accs)

      bound_matrix = get_weighted_mean_bound_matrix(network_vars, self.bound, self.bound - epoch, weighted_avg)

      mean_train_acc = infer_and_accuracy(data["train_x"], data["train_y"], weighted_avg, self.architecture)
      mean_val_acc = infer_and_accuracy(data["val_x"], data["val_y"], weighted_avg, self.architecture)

      all_avgs.append(weighted_avg)
      all_train_accs.append(mean_train_acc)
      all_val_accs.append(mean_val_acc)

      clear_print("Training accuracy for mean parameters: %s" % (mean_train_acc))
      clear_print("Validation accuracy for mean parameters: %s" % (mean_val_acc))

    total_time = time.time() - epoch_start
    print("Time to run all epochs: %.3f" % (total_time))

    best_ind = np.argmax(all_val_accs)
    best_avg = all_avgs[best_ind]

    final_train_acc = infer_and_accuracy(data["train_x"], data["train_y"], best_avg, self.architecture)
    final_val_acc = infer_and_accuracy(data["val_x"], data["val_y"], best_avg, self.architecture)
    final_test_acc = infer_and_accuracy(data["test_x"], data["test_y"], best_avg, self.architecture)

    clear_print("Final Training accuracy for mean parameters: %s" % (final_train_acc))
    clear_print("Final Validation accuracy for mean parameters: %s" % (final_val_acc))
    clear_print("Final Testing accuracy for mean parameters: %s" % (final_test_acc))

    self.results[loss][str(seed)] = {
      "all_avgs": all_avgs,
      "all_train_accs": all_train_accs,
      "all_val_accs": all_val_accs,
      "all_stds": all_stds,
      "final_train_acc": final_train_acc,
      "final_val_acc": final_val_acc,
      "final_test_acc": final_test_acc,
      "runtime": total_time
    }

  def run_gd(self, seed):
    lr = 1e-1
    N = self.num_examples

    tf.set_random_seed(seed)
    data = load_data("adult", N, seed)

    print_str = "Architecture: %s. N: %s. LR: %s. Bound: %s. Seed: %s."
    clear_print(print_str % ("-".join([str(x) for x in self.architecture]), N, lr, self.bound, seed))
    nn = GD_NN(data, N, self.architecture, lr, self.bound, seed, self.batch_size, max_epoch=1000)
    nn.train(self.max_runtime*60)

    nn_runtime = nn.get_runtime()
    varMatrices = nn.extract_values()

    train_acc = infer_and_accuracy(data['train_x'], data['train_y'], varMatrices, self.architecture)
    test_acc = infer_and_accuracy(data['test_x'], data['test_y'], varMatrices, self.architecture)
    print("train_acc", train_acc)
    print("test_acc", test_acc)

    self.results["gd_nn"][str(seed)] = {
      "final_train_acc": train_acc,
      "final_test_acc": test_acc,
      "runtime": nn_runtime
    }

  def plot_results(self):
    clear_print("Batch training for MIP model %s" % self.loss)
    test_accs = [self.results[self.loss][str(s)]["final_test_acc"] for s in self.seeds]
    runtimes = [self.results[self.loss][str(s)]["runtime"] for s in self.seeds]

    clear_print("Average testing accuracy for all seeds: %s +/- %s" % (np.mean(test_accs), np.std(test_accs)))
    clear_print("Average runtime [s] for all seeds: %s +/- %s" % (np.mean(runtimes), np.std(runtimes)))

    clear_print("Batch training for GD")
    test_accs = [self.results["gd_nn"][str(s)]["final_test_acc"] for s in self.seeds]
    runtimes = [self.results["gd_nn"][str(s)]["runtime"] for s in self.seeds]

    clear_print("Average testing accuracy for all seeds: %s +/- %s" % (np.mean(test_accs), np.std(test_accs)))
    clear_print("Average runtime [s] for all seeds: %s +/- %s" % (np.mean(runtimes), np.std(runtimes)))


  def run_batch(self, batch_data):
    batch = batch_data[0]
    bound_matrix = batch_data[1]
    batch_num = batch_data[2]
    nn = get_nn(self.loss, batch, self.architecture, self.bound, 0, "")
    nn.m.setParam('Threads', 2)
    nn.update_bounds(bound_matrix)
    nn.train(self.max_runtime*60, 1)
    runtime = nn.get_runtime()
    varMatrices = nn.extract_values()
    printProgressBar(batch_num+1, self.num_batches)

    del nn.m
    del nn

    return runtime, extract_params(varMatrices, get_keys=["w","b","H"])[0]

  def run_batches(self, batches, bound_matrix):
    batch_start = time.time()    

    pool = Pool(processes=self.num_processes)
    batch_data = []
    for i,batch in enumerate(batches):
      batch_data.append([batch, bound_matrix, i])
    output = pool.imap_unordered(self.run_batch, batch_data)
    pool.close()
    pool.join()
    print("")

    batch_end = time.time()
    batch_time = batch_end - batch_start
    print("Time to run batches: %.3f" % (batch_time))
    #runtimes = [x[0] for x in output]
    network_vars = [x[1] for x in output]

    return network_vars



def get_std(network_vars):
  std_vars = {}
  std_sum = 0
  for key in network_vars[0]:
    if 'w_' in key or 'b_' in key:
      tmp_vars = np.stack([tmp[key] for tmp in network_vars])
      std_vars[key] = np.std(tmp_vars, axis=0)
      std_sum += std_vars[key].sum()
  return std_vars, std_sum