import nnabla.functions as F


def clip_by_value(x, minimum, maximum):
    return F.minimum_scalar(F.maximum_scalar(x, minimum), maximum)
