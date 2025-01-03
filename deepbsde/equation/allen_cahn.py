import numpy as np
import tensorflow as tf

from .base import Equation


class AllenCahn(Equation):
    """Allen-Cahn equation in PNAS paper doi.org/10.1073/pnas.1718942115"""
    def __init__(self, eqn_config):
        super(AllenCahn, self).__init__(eqn_config)
        self.x_init = np.zeros(self.dim)
        self.sigma = np.sqrt(2.0)

    def sample(self, num_sample):
        dw_sample = np.random.normal(size=[num_sample, self.dim, self.num_time_interval]) * self.sqrt_delta_t
        x_sample = np.zeros([num_sample, self.dim, self.num_time_interval + 1])
        x_sample[:, :, 0] = np.ones([num_sample, self.dim]) * self.x_init
        for i in range(self.num_time_interval):
            x_sample[:, :, i + 1] = x_sample[:, :, i] + self.sigma * dw_sample[:, :, i]
        return dw_sample, x_sample

    def f_tf(self, t, x, y, z):
        return y - tf.pow(y, 3)

    def g_tf(self, t, x):
        return 0.5 / (1 + 0.2 * tf.reduce_sum(tf.square(x), 1, keepdims=True))