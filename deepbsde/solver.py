"""
Implementation of the BSDESolver class for training neural networks to solve BSDEs.

This module contains the main solver that handles training, loss computation, and optimization
of the neural network model for solving Backward Stochastic Differential Equations (BSDEs).
"""

import logging
import time
import json

import numpy as np
import tensorflow as tf

from . import equation as eqn
from .model import NonsharedModel

# Constant for clipping the loss delta to avoid numerical instability
DELTA_CLIP = 50.0


class BSDESolver(object):
    """Solver class that trains neural networks to solve BSDEs.
    
    This class handles the training loop, loss computation, and optimization of the neural
    network model. It uses stochastic gradient descent with the Adam optimizer.

    Args:
        config: Configuration object containing model and equation parameters
        bsde: BSDE equation object defining the problem to solve
    """
    def __init__(self, config, bsde):
        self.eqn_config = config['eqn_config']
        self.net_config = config['net_config']
        self.bsde = bsde

        # Initialize model and get reference to y_init parameter
        self.model = NonsharedModel(config, bsde)
        self.y_init = self.model.y_init

        # Setup learning rate schedule and Adam optimizer
        lr_schedule = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
            self.net_config['lr_boundaries'], self.net_config['lr_values'])
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule, epsilon=1e-8)

    def train(self):
        """Trains the model using stochastic gradient descent.
        
        Performs training iterations, periodically evaluating on validation data
        and logging the results.

        Returns:
            numpy.ndarray: Training history containing step, loss, Y0 and elapsed time
        """
        start_time = time.time()
        training_history = []
        valid_data = self.bsde.sample(self.net_config['valid_size'])

        # Main training loop
        for step in range(self.net_config['num_iterations']+1):
            # Log progress at specified frequency
            if step % self.net_config['logging_frequency'] == 0:
                loss = self.loss_fn(valid_data, training=False).numpy()
                y_init = self.y_init.numpy()[0]
                elapsed_time = time.time() - start_time
                training_history.append([step, loss, y_init, elapsed_time])
                if self.net_config['verbose']:
                    logging.info("step: %5u,    loss: %.4e, Y0: %.4e,   elapsed time: %3u" % (
                        step, loss, y_init, elapsed_time))
            self.train_step(self.bsde.sample(self.net_config['batch_size']))
        return np.array(training_history)

    def loss_fn(self, inputs, training):
        """Computes the loss for training the model.
        
        Calculates the mean squared error between predicted and true terminal values,
        with special handling for large deviations using linear approximation.

        Args:
            inputs: Tuple of (dw, x) containing Brownian increments and state process
            training: Boolean indicating if in training mode

        Returns:
            tf.Tensor: Computed loss value
        """
        dw, x = inputs
        y_terminal = self.model(inputs, training)
        delta = y_terminal - self.bsde.g_tf(self.bsde.total_time, x[:, :, -1])
        # Use linear approximation outside the clipped range for numerical stability
        loss = tf.reduce_mean(tf.where(tf.abs(delta) < DELTA_CLIP, tf.square(delta),
                                     2 * DELTA_CLIP * tf.abs(delta) - DELTA_CLIP ** 2))
        return loss

    def grad(self, inputs, training):
        """Computes gradients of the loss with respect to model parameters.
        
        Args:
            inputs: Tuple of (dw, x) containing Brownian increments and state process
            training: Boolean indicating if in training mode

        Returns:
            list: Gradients for each trainable variable
        """
        with tf.GradientTape(persistent=True) as tape:
            loss = self.loss_fn(inputs, training)
        grad = tape.gradient(loss, self.model.trainable_variables)
        del tape
        return grad

    @tf.function
    def train_step(self, train_data):
        """Performs one training step using gradient descent.
        
        Args:
            train_data: Batch of training data
        """
        grad = self.grad(train_data, training=True)
        self.optimizer.apply_gradients(zip(grad, self.model.trainable_variables))



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

    # -------------------------------------------------------
    # Model Initialization
    # -------------------------------------------------------
    # Initialize BSDE equation
    bsde = getattr(eqn, config['eqn_config']['eqn_name'])(config['eqn_config'])
    tf.keras.backend.set_floatx(config['net_config']['dtype'])

    # Initialize and train BSDE solver
    solver = BSDESolver(config, bsde)
    
    # -------------------------------------------------------
    # Testing Model Components
    # -------------------------------------------------------
    # Test model initialization
    print("\nTesting model initialization...")
    print("Model architecture:", solver.model)
    print("Number of trainable variables:", len(solver.model.trainable_variables))
    print("Initial y_init value:", solver.y_init.numpy()[0])
    
    # Test data generation
    print("\nTesting data generation...")
    test_batch = bsde.sample(config['net_config']['batch_size'])
    print("Sample batch shapes - dw:", test_batch[0].shape, "x:", test_batch[1].shape)
    
    # Test forward pass
    print("\nTesting forward pass...")
    y_pred = solver.model(test_batch, training=False)
    print("Model output shape:", y_pred.shape)
    
    # Test loss computation
    print("\nTesting loss computation...")
    loss = solver.loss_fn(test_batch, training=False)
    print("Initial loss value:", loss.numpy())
    
    # Test gradient computation
    print("\nTesting gradient computation...")
    grads = solver.grad(test_batch, training=True)
    print("Number of gradient tensors:", len(grads))
    print("\nGradient shapes:")
    for i, shape in enumerate(g.shape for g in grads):
        if i > 0 and i % 3 == 0:
            print()  # Add line break every 3 shapes for readability
        print(f"  {shape}", end="  ")
    print("\n")

    # -------------------------------------------------------
    # Training and Results
    # -------------------------------------------------------
    # Test full training
    print("\nStarting full training...")
    training_history = solver.train()
    
    # Output detailed training results
    print("\nTraining completed.")
    print("Training history shape:", training_history.shape)
    print("Columns: [step, loss, Y0, elapsed_time]")
    print("\nFinal metrics:")
    print("Final Loss:", training_history[-1, 1])
    print("Initial Y Value:", training_history[-1, 2]) 
    print("Total Elapsed Time:", training_history[-1, 3])
    print("Loss progression:", training_history[:, 1])
    
    # -------------------------------------------------------
    # Model Persistence Analysis
    # -------------------------------------------------------
    # Test model saving/loading if implemented
    print("\nModel persistence capabilities:")
    print("Trainable variables:")
    for var in solver.model.trainable_variables:
        print(f"  {var.name}")
        
    print("\nVariable shapes:")
    for var, shape in zip(solver.model.trainable_variables, 
                         [var.shape for var in solver.model.trainable_variables]):
        print(f"  {var.name}: {shape}")