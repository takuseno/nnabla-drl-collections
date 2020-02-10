import numpy as np
import nnabla as nn
import nnabla.functions as F


def clip_by_value(x, minimum, maximum):
    return F.minimum_scalar(F.maximum_scalar(x, minimum), maximum)


def set_seed(seed):
    np.random.seed(seed)
    nn.random.prng = np.random.RandomState(seed)
