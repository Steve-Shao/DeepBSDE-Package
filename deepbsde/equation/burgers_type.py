import numpy as np
import tensorflow as tf

from .base import Equation


class BurgersType(Equation):
    """
    Multidimensional Burgers-type PDE in Section 4.5 of Comm. Math. Stat. paper
    doi.org/10.1007/s40304-017-0117-6
    """
    def __init__(self, eqn_config):
        super(BurgersType, self).__init__(eqn_config)
        self.x_init = np.zeros(self.dim)
        self.y_init = 1 - 1.0 / (1 + np.exp(0 + np.sum(self.x_init) / self.dim))
        self.sigma = self.dim + 0.0

    def sample(self, num_sample):
        dw_sample = np.random.normal(size=[num_sample, self.dim, self.num_time_interval]) * self.sqrt_delta_t
        x_sample = np.zeros([num_sample, self.dim, self.num_time_interval + 1])
        x_sample[:, :, 0] = np.ones([num_sample, self.dim]) * self.x_init
        for i in range(self.num_time_interval):
            x_sample[:, :, i + 1] = x_sample[:, :, i] + self.sigma * dw_sample[:, :, i]
        return dw_sample, x_sample

    def f_tf(self, t, x, y, z):
        return (y - (2 + self.dim) / 2.0 / self.dim) * tf.reduce_sum(z, 1, keepdims=True)

    def g_tf(self, t, x):
        return 1 - 1.0 / (1 + tf.exp(t + tf.reduce_sum(x, 1, keepdims=True) / self.dim))