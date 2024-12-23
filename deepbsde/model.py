"""
Implementation of the NonsharedModel class for solving BSDEs using neural networks.

This module contains the main model architecture that combines multiple feed-forward 
subnets to approximate solutions of high-dimensional BSDEs (Backward Stochastic 
Differential Equations).
"""

# Standard library imports
import json
import logging
import numpy as np
import tensorflow as tf

# Local imports
from . import equation as eqn
from .subnet import FeedForwardSubNet


class NonsharedModel(tf.keras.Model):
    """Neural network model that uses separate subnets for each time step.
    
    This model implements a deep BSDE solver where each time step has its own 
    subnet for better approximation capability.
    
    Args:
        config: Configuration object containing model and equation parameters
        bsde: BSDE equation object defining the problem to solve
    """
    def __init__(self, config, bsde):
        super(NonsharedModel, self).__init__()
        self.eqn_config = config['eqn_config']
        self.net_config = config['net_config']
        self.bsde = bsde
        
        # Initialize y and z variables with random values
        self.y_init = tf.Variable(np.random.uniform(low=self.net_config['y_init_range'][0],
                                                    high=self.net_config['y_init_range'][1],
                                                    size=[1])
                                  )
        self.z_init = tf.Variable(np.random.uniform(low=-.1, high=.1,
                                                    size=[1, self.eqn_config['dim']])
                                  )

        # Create subnet for each time step except the last one
        self.subnet = [FeedForwardSubNet(config) for _ in range(self.bsde.num_time_interval-1)]

    def call(self, inputs, training):
        """Forward pass of the model.
        
        Args:
            inputs: Tuple of (dw, x) where:
                dw: Brownian increments tensor
                x: State process tensor
            training: Boolean indicating training vs inference mode
            
        Returns:
            y: Terminal value approximation
        """
        dw, x = inputs
        time_stamp = np.arange(0, self.eqn_config['num_time_interval']) * self.bsde.delta_t
        all_one_vec = tf.ones(shape=tf.stack([tf.shape(dw)[0], 1]), dtype=self.net_config['dtype'])
        y = all_one_vec * self.y_init
        z = tf.matmul(all_one_vec, self.z_init)

        # Forward propagation through time steps
        for t in range(0, self.bsde.num_time_interval-1):
            y = y - self.bsde.delta_t * (
                self.bsde.f_tf(time_stamp[t], x[:, :, t], y, z)
            ) + tf.reduce_sum(z * dw[:, :, t], 1, keepdims=True)
            z = self.subnet[t](x[:, :, t + 1], training) / self.bsde.dim
            
        # Handle terminal time step
        y = y - self.bsde.delta_t * self.bsde.f_tf(time_stamp[-1], x[:, :, -2], y, z) + \
            tf.reduce_sum(z * dw[:, :, -1], 1, keepdims=True)

        return y


if __name__ == '__main__':
    # -------------------------------------------------------
    # Setup and Configuration
    # -------------------------------------------------------
    # Set random seeds for reproducibility
    tf.random.set_seed(42)
    np.random.seed(42)

    # Load and parse configuration file
    config = json.loads('''{
        "eqn_config": {
            "_comment": "HJB equation in PNAS paper doi.org/10.1073/pnas.1718942115",
            "eqn_name": "HJBLQ", 
            "total_time": 1.0,
            "dim": 100,
            "num_time_interval": 20
        },
        "net_config": {
            "y_init_range": [0, 1],
            "num_hiddens": [110, 110],
            "lr_values": [1e-2, 1e-2],
            "lr_boundaries": [1000],
            "num_iterations": 2000,
            "batch_size": 64,
            "valid_size": 256,
            "logging_frequency": 100,
            "dtype": "float64",
            "verbose": true
        }
    }''')
        
    # Initialize BSDE equation based on config
    bsde = getattr(eqn, config['eqn_config']['eqn_name'])(config['eqn_config'])
    tf.keras.backend.set_floatx(config['net_config']['dtype'])

    # Initialize NonsharedModel
    model = NonsharedModel(config, bsde)

    # -------------------------------------------------------
    # Model Structure Analysis
    # -------------------------------------------------------
    print("\nNonsharedModel Structure:")
    print("------------------------")
    # Create dummy input to build the model
    batch_size = 64
    dw = tf.zeros((batch_size, config['eqn_config']['dim'], config['eqn_config']['num_time_interval']))
    x = tf.zeros((batch_size, config['eqn_config']['dim'], config['eqn_config']['num_time_interval']))
    model((dw, x), training=False)  # Build model
    model.summary()

    # Print model parameters
    print("\nModel Parameters:")
    print("----------------")
    total_params = 0
    for var in model.trainable_variables:
        params = np.prod(var.shape)
        total_params += params
        print(f"{var.name}: shape={var.shape}, params={params}")
    print(f"Total trainable parameters: {total_params}")

    # -------------------------------------------------------
    # Subnet Testing
    # -------------------------------------------------------
    print("\nSubnet Details and Tests:")
    print("-----------------------")
    print(f"Number of subnets: {len(model.subnet)}")
    print(f"Y initialization range: {config['net_config']['y_init_range']}")
    print(f"Z initialization range: [-0.1, 0.1]")
    
    # Test each subnet individually
    test_input = tf.random.normal((batch_size, config['eqn_config']['dim']))
    for i, subnet in enumerate(model.subnet):
        subnet_output = subnet(test_input, training=False)
        print(f"\nSubnet {i} test:")
        print(f"Output shape: {subnet_output.shape}")
        print(f"Output mean: {tf.reduce_mean(subnet_output):.6f}")
        print(f"Output std: {tf.math.reduce_std(subnet_output):.6f}")
        print(f"Output min: {tf.reduce_min(subnet_output):.6f}")
        print(f"Output max: {tf.reduce_max(subnet_output):.6f}")

    # -------------------------------------------------------
    # Full Model Forward Pass Tests
    # -------------------------------------------------------
    print("\nFull Model Forward Pass Tests:")
    print("----------------------------")
    
    # Test 1: Zero inputs
    y_zero = model((dw, x), training=False)
    print("\nTest with zero inputs:")
    print(f"Output shape: {y_zero.shape}")
    print(f"Output mean: {tf.reduce_mean(y_zero):.6f}")
    print(f"Output std: {tf.math.reduce_std(y_zero):.6f}")
    print(f"Output min: {tf.reduce_min(y_zero):.6f}")
    print(f"Output max: {tf.reduce_max(y_zero):.6f}")

    # Test 2: Random normal inputs
    dw_random = tf.random.normal((batch_size, config['eqn_config']['dim'], config['eqn_config']['num_time_interval']))
    x_random = tf.random.normal((batch_size, config['eqn_config']['dim'], config['eqn_config']['num_time_interval']))
    y_random = model((dw_random, x_random), training=False)
    print("\nTest with random normal inputs:")
    print(f"Output shape: {y_random.shape}")
    print(f"Output mean: {tf.reduce_mean(y_random):.6f}")
    print(f"Output std: {tf.math.reduce_std(y_random):.6f}")
    print(f"Output min: {tf.reduce_min(y_random):.6f}")
    print(f"Output max: {tf.reduce_max(y_random):.6f}")

    # Test 3: Edge case with large values
    dw_large = tf.random.normal((batch_size, config['eqn_config']['dim'], config['eqn_config']['num_time_interval'])) * 10
    x_large = tf.random.normal((batch_size, config['eqn_config']['dim'], config['eqn_config']['num_time_interval'])) * 10
    y_large = model((dw_large, x_large), training=False)
    print("\nTest with large inputs:")
    print(f"Output shape: {y_large.shape}")
    print(f"Output mean: {tf.reduce_mean(y_large):.6f}")
    print(f"Output std: {tf.math.reduce_std(y_large):.6f}")
    print(f"Output min: {tf.reduce_min(y_large):.6f}")
    print(f"Output max: {tf.reduce_max(y_large):.6f}")
