import argparse
from scripts.exp1 import Exp1
from scripts.exp2 import Exp2
from scripts.exp3 import Exp3

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Optional app description')

  parser.add_argument('--exp', type=str)
  parser.add_argument('--show', action='store_true')
  parser.add_argument('--short', action='store_true')
  parser.add_argument('--loss', type=str, default='sat_margin')
  parser.add_argument('--data', type=str, default='adult')
  parser.add_argument('--batch_size', type=int, default=100)

  args = parser.parse_args()
  exp = args.exp
  show = args.show
  short = args.short
  loss = args.loss
  data = args.data
  batch_size = args.batch_size

  if exp == "exp1":
    run = Exp1(short, show)
  elif exp == "exp2":
    run = Exp2(short, show)
  elif exp == "exp3":
    run = Exp3(short, show, loss, data, batch_size)
  else:
    raise Exception("Experiment %s not known" % (exp))
    
  run.run_all()
  run.plot_results()