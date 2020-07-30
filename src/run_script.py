from scripts.script_master import Script_Master
from time import time
import argparse

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Optional app description')

  parser.add_argument('--script', type=str)
  parser.add_argument('--losses', default="", type=str)
  parser.add_argument('--data', default="adult", type=str)
  parser.add_argument('--show', action='store_true')
  parser.add_argument('--short', action='store_true')
  parser.add_argument('--table', action='store_true')
  args = parser.parse_args()
  script = args.script
  data = args.data
  show = args.show
  short = args.short
  table = args.table
  print("short", short)
  print("show", show)
  losses = args.losses.split(",")

  fair = False
  max_time = 10*60
  seeds = [1348612,7864568,9434861]
  hls = [[16]]
  bounds = [1]
  focus = 1
  regs = []

  if data == "adult":
    num_examples = [40, 80, 120, 160, 200, 240, 280]
  elif data == "heart":
    num_examples = [40, 80, 120, 160, 200]
  elif data == "mnist":
    num_examples = [20,40,60,80,100]
    hls = [[], [16], [16,16]]
    max_time = 2*60


  if script == "precision":
    bounds = [1,3,7,15, 31]
    bounds = [1,3,7,15]
    if short:
      bounds = [1,3,7]
  elif script =="push":
    num_examples = [20]
    bounds = [15]
  elif script =="reg":
    hls=[[100]]
    num_examples = [300]
    max_time = 24*60
    bounds = [15]
    regs = [0, -1, 1, 0.1]
    if short:
      hls = [[30]]
      num_examples = [100]
      regs = [0, -1, 0.1]
  elif script == "fair":
    fair = True
    num_examples = [300]
    bounds = [15]
    if short:
      num_examples = [200]

  if short:
    num_examples = num_examples[0:3]
    seeds = seeds[0:2]
    max_time = 10

  start = time()
  if script in ["precision", "losses", "push", "reg", "fair"]:
    SR = Script_Master(script, losses, data, num_examples, max_time, seeds, hls, bounds, regs, fair=fair, show=show)
    SR.run_all()
    if table:
      SR.make_tables()
    elif script == "reg":
      SR.plot_reg_results()
    elif script == "fair":
      SR.visualize_fairness("EO")
      SR.visualize_fairness("DP")
    else:
      SR.plot_all()
      #SR.subplot_results()
  else:
    raise Exception("Script %s not known" % script)
  end = time()
  print("Time to run script %s:  %.2f" % (script, (end - start)))

