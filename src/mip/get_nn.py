from mip.min_w import MIN_W
from mip.max_correct import MAX_CORRECT
from mip.min_hinge import MIN_HINGE
from mip.sat_margin import SAT_MARGIN
from mip.max_m import MAX_M


def get_nn(loss, data, architecture, bound, reg, fair):
  if loss == "min_w":
    return MIN_W(data, architecture, bound, reg, fair)
  elif loss == "max_m":
    return MAX_M(data, architecture, bound, reg, fair)
  elif loss == "max_correct":
    return MAX_CORRECT(data, architecture, bound, reg, fair)
  elif loss == "min_hinge":
    return MIN_HINGE(data, architecture, bound, reg, fair)
  elif loss == "sat_margin":
    return SAT_MARGIN(data, architecture, bound, reg, fair)
  else:
    raise Exception("model %s not known" % loss)
