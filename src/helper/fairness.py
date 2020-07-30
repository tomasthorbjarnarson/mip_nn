import numpy as np
from pdb import set_trace

def equalized_odds(data, pred_labels, true_labels):
  females = data[:,64]
  males = data[:,65]
  female_pred1_true1 = (females*true_labels*pred_labels).sum() / (females*true_labels).sum()
  male_pred1_true1 = (males*true_labels*pred_labels).sum() / (males*true_labels).sum()

  false_labels = 1 - true_labels
  female_pred1_true0 = (females*false_labels*pred_labels).sum() / (females*false_labels).sum()
  male_pred1_true0 = (males*false_labels*pred_labels).sum() / (males*false_labels).sum()

  return female_pred1_true1, male_pred1_true1, female_pred1_true0, male_pred1_true0

def demographic_parity(data, pred_labels, true_labels):
  females = data[:,64]
  males = data[:,65]
  female_pred1 = (females*pred_labels).sum() / females.sum()
  male_pred1 = (males*pred_labels).sum() / males.sum()

  return female_pred1, male_pred1



#p111 = 0
#n11  = 0
#p101 = 0
#n01  = 0
#p110 = 0
#n10  = 0
#p100 = 0
#n00  = 0
#for i, pred in enumerate(labels_test):
#  if nn.data['test_x'][i,64] == 1 and nn.data['test_y'][i] == 1:
#    n11 += 1
#    if pred == 1:
#      p111 += 1
#  elif nn.data['test_x'][i,64] == 0 and nn.data['test_y'][i] == 1:
#    n01 += 1
#    if pred == 1:
#      p101 += 1
#  elif nn.data['test_x'][i,64] == 1 and nn.data['test_y'][i] == 0:
#    n10 += 1
#    if pred == 1:
#      p110 += 1
#  elif nn.data['test_x'][i,64] == 0 and nn.data['test_y'][i] == 0:
#    n00 += 1
#    if pred == 1:
#      p100 += 1


#EO: 𝑃(̂𝑌 = 1|𝐴 = 1,𝑌 = 𝑦) = 𝑃(̂𝑌 = 1|𝐴 = 0,𝑌 = 𝑦)

#DP: 𝑃(̂𝑌 = 1|𝐴 = 0) = 𝑃(̂𝑌 = 1|𝐴 = 1)