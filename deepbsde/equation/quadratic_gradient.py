import numpy as np
import tensorflow as tf

from .base import Equation


class QuadraticGradient(Equation):
    """
    An example PDE with quadratically growing derivatives in Section 4.6 of Comm. Math. Stat. paper
    doi.org/10.1007/s40304-017-0117-6
    """
    def __init__(self, eqn_config):
        super(QuadraticGradient, self).__init__(eqn_config)
        self.alpha = 0.4
        self.x_init = np.zeros(self.dim)
        base = self.total_time + np.sum(np.square(self.x_init) / self.dim)
        self.y_init = np.sin(np.power(base, self.alpha))

    def sample(self, num_sample):
        dw_sample = np.random.normal(size=[num_sample, self.dim, self.num_time_interval]) * self.sqrt_delta_t
        x_sample = np.zeros([num_sample, self.dim, self.num_time_interval + 1])
        x_sample[:, :, 0] = np.ones([num_sample, self.dim]) * self.x_init
        for i in range(self.num_time_interval):
            x_sample[:, :, i + 1] = x_sample[:, :, i] + dw_sample[:, :, i]
        return dw_sample, x_sample

    def f_tf(self, t, x, y, z):
        x_square = tf.reduce_sum(tf.square(x), 1, keepdims=True)
        base = self.total_time - t + x_square / self.dim
        base_alpha = tf.pow(base, self.alpha)
        derivative = self.alpha * tf.pow(base, self.alpha - 1) * tf.cos(base_alpha)
        term1 = tf.reduce_sum(tf.square(z), 1, keepdims=True)
        term2 = -4.0 * (derivative ** 2) * x_square / (self.dim ** 2)
        term3 = derivative
        term4 = -0.5 * (
            2.0 * derivative + 4.0 / (self.dim ** 2) * x_square * self.alpha * (
                (self.alpha - 1) * tf.pow(base, self.alpha - 2) * tf.cos(base_alpha) - (
                    self.alpha * tf.pow(base, 2 * self.alpha - 2) * tf.sin(base_alpha)
                    )
                )
            )
        return term1 + term2 + term3 + term4

    def g_tf(self, t, x):
        return tf.sin(
            tf.pow(tf.reduce_sum(tf.square(x), 1, keepdims=True) / self.dim, self.alpha))