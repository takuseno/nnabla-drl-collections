import numpy as np
import nnabla.functions as F
import math


LOGPROBC = -0.5 * math.log(2 * math.pi)

class MultivariateNormal:
    def __init__(self, loc, scale):
        assert loc.shape == scale.shape,\
            'For now, loc and scale must have same shape.'
        if isinstance(loc, np.ndarray):
            loc = nn.Variable.from_numpy_array(loc)
            loc.persistent = True
        if isinstance(scale, np.ndarray):
            scale = nn.Variable.from_numpy_array(scale)
            scale.persistent = True
        self.loc = loc
        self.scale = scale

    def mean(self):
        # to avoid no parent error
        return F.identity(self.loc)

    def stddev(self):
        # to avoid no parent error
        return F.identity(self.scale)

    @property
    def d(self):
        return self.scale.shape[-1]

    def log_prob(self, x):
        m = F.batch_matmul(F.batch_inv(self._diag_scale()),
                           F.reshape(x - self.loc, self.loc.shape + (1,)))
        m = F.reshape(F.batch_matmul(m, m, True), (x.shape[0], 1))
        logz = LOGPROBC * self.d - self._logdet_scale()
        return logz - 0.5 * m

    def _logdet_scale(self):
        return F.sum(F.log(self.scale), axis=1, keepdims=True)

    def _diag_scale(self):
        return F.matrix_diag(self.scale)

    def sample(self, shape=None):
        if shape is None:
            shape = self.loc.shape
        eps = F.randn(mu=0.0, sigma=1.0, shape=shape)
        return self.mean() + self.scale * eps


class Normal:
    def __init__(self, loc, scale):
        assert loc.shape == scale.shape,\
            'For now, loc and scale must have same shape.'
        if isinstance(loc, np.ndarray):
            loc = nn.Variable.from_numpy_array(loc)
            loc.persistent = True
        if isinstance(scale, np.ndarray):
            scale = nn.Variable.from_numpy_array(scale)
            scale.persistent = True
        self.loc = loc
        self.scale = scale

    def mean(self):
        # to avoid no parent error
        return F.identity(self.loc)

    def stddev(self):
        # to avoid no parent error
        return F.identity(self.scale)

    def variance(self):
        return self.stddev() ** 2

    def log_prob(self, x):
        var = self.variance()
        return LOGPROBC - F.log(self.scale) - 0.5 * (x - self.loc) ** 2 / var

    def sample(self, shape=None):
        if shape is None:
            shape = self.loc.shape
        eps = F.randn(mu=0.0, sigma=1.0, shape=shape)
        return self.mean() + self.stddev() * eps
