"""
Implementation of the FeedForwardSubNet class for solving BSDEs using neural networks.
This module contains the subnet architecture that processes input data for each time step.
"""

import munch
import numpy as np
import tensorflow as tf


class FeedForwardSubNet(tf.keras.Model):
    """A feed-forward neural network with batch normalization layers.
    
    Current Implementation:
    -----------------------
    This network:
      - Takes an input of dimension `dim` (from config.eqn_config.dim).
      - Applies an initial Batch Normalization (BN).
      - Then applies a sequence of (Dense -> BN -> ReLU) layers for each hidden layer.
      - Finally, applies a last Dense layer followed by a final BN.
    
    The final output dimension matches `dim`.

    Note on Implementation Details:
    -------------------------------
    While we do not change the current code logic, here are various design considerations 
    and details one could tune or specify if needed:
    
    1. Parameter Initialization:
       - Dense layers currently rely on default initializers, but one could specify:
         * Weights: GlorotUniform, HeNormal, or custom initializers.
         * Biases: Typically zeros initialization is standard, but can be tuned.
       - BatchNormalization layers: 
         * beta_initializer, gamma_initializer, moving_mean_initializer, 
           moving_variance_initializer can be controlled.
    
    2. Batch Normalization:
       - Momentum and epsilon can be fine-tuned.
       - Deciding whether to use BN before or after activation.
       - Switching between training/inference modes (controlled by `training` flag).
    
    3. Activation Functions:
       - Currently using ReLU (tf.nn.relu).
       - Could consider leaky ReLU, ELU, GELU, or other activations.
       - Could also consider adding activation after the final Dense if needed.
    
    4. Network Structure:
       - Number of hidden layers and their widths can be altered.
       - Could add skip connections, residual blocks, or other architectures.
       - Could add dropout layers, layer normalization, or other regularization techniques.
    
    5. Data Types and Computation:
       - The code runs in float32 by default but config could specify float64 or mixed precision.
       - Ensure the chosen dtype is consistent across the network.
    
    6. Other Techniques:
       - Could incorporate batch normalization in different places (before or after activation).
       - Could add learning rate schedules more directly coupled with optimizer creation (not shown here).
       - Could integrate BatchNorm in a more conditional way (e.g., only for training or certain layers).
       - Could add constraints or regularizers to the Dense layers.

    Despite these possible improvements and choices, the current code remains unchanged 
    in logic to maintain the original functionality.
    """

    def __init__(self, config):
        """Initialize the feed-forward sub-network.
        
        Args:
            config (munch.Munch): Configuration object containing:
                - eqn_config.dim: The dimension of the input and final output.
                - net_config.num_hiddens: A list defining the size of each hidden layer.
        
        This sets up:
          - A list of BatchNormalization layers (one for input, one for each hidden layer, one for output).
          - A list of Dense layers corresponding to hidden layers and a final Dense layer.
        """
        super(FeedForwardSubNet, self).__init__()
        dim = config.eqn_config.dim
        num_hiddens = config.net_config.num_hiddens

        # Create batch normalization layers:
        # We need (len(num_hiddens) + 2) BN layers in total:
        # 1 for initial input normalization, 1 for each hidden layer, and 1 final BN.
        self.bn_layers = [
            tf.keras.layers.BatchNormalization(
                momentum=0.99,
                epsilon=1e-6,
                beta_initializer=tf.random_normal_initializer(0.0, stddev=0.1),
                gamma_initializer=tf.random_uniform_initializer(0.1, 0.5)
            )
            for _ in range(len(num_hiddens) + 2)
        ]

        # Create dense layers for hidden layers:
        self.dense_layers = [
            tf.keras.layers.Dense(
                units=num_hiddens[i],
                use_bias=False,   # BN after this handles bias via shifting, so bias can be omitted.
                activation=None   # Activation handled separately via tf.nn.relu.
            )
            for i in range(len(num_hiddens))
        ]
        
        # Add final Dense layer with output dimension = dim (no activation here).
        self.dense_layers.append(tf.keras.layers.Dense(dim, activation=None))

    def call(self, x, training):
        """Forward pass through the network.
        
        Args:
            x (tf.Tensor): Input tensor of shape [batch_size, dim].
            training (bool): If True, run in training mode (updating BN statistics).
                             If False, run in inference mode (using moving averages in BN).
        
        Returns:
            tf.Tensor: The output tensor of shape [batch_size, dim].
        """
        # Initial BN
        x = self.bn_layers[0](x, training=training)

        # Pass through each hidden layer: Dense -> BN -> ReLU
        for i in range(len(self.dense_layers) - 1):
            x = self.dense_layers[i](x)
            x = self.bn_layers[i + 1](x, training=training)
            x = tf.nn.relu(x)

        # Final Dense layer and final BN (no activation)
        x = self.dense_layers[-1](x)
        x = self.bn_layers[-1](x, training=training)
        return x


if __name__ == "__main__":
    # -------------------------------------------------------
    # Prepare Mock Configuration
    # -------------------------------------------------------
    # We do not alter the network structure or parameters, only the test code.
    mock_config = munch.Munch()
    mock_config.eqn_config = munch.Munch()
    mock_config.net_config = munch.Munch()
    mock_config.eqn_config.dim = 10  # dimension of input/output
    mock_config.net_config.num_hiddens = [20, 20]  # hidden layers

    # Initialize the network
    net = FeedForwardSubNet(mock_config)

    # -------------------------------------------------------
    # Build and Summarize the Model
    # -------------------------------------------------------
    # Build the model for a known input shape so summary can be printed
    net.build((None, mock_config.eqn_config.dim))
    print("\nModel Structure:")
    print("---------------")
    net.summary()
    print("---------------\n")

    # -------------------------------------------------------
    # Check Parameter Initialization
    # -------------------------------------------------------
    # Print mean and std of each layer's parameters to verify initialization distributions.
    print("Parameter Initialization Distributions:")
    for idx, layer in enumerate(net.dense_layers):
        weights = layer.get_weights()
        if len(weights) > 0:  # Dense layers should have at least one weight tensor
            w = weights[0]  # Since use_bias=False for hidden layers, only weights
            print(f"Dense layer {idx} weights shape: {w.shape}")
            print(f"  mean: {np.mean(w):.4f}, std: {np.std(w):.4f}")
        else:
            print(f"Dense layer {idx} has no weights (unexpected).")

    # For BatchNormalization layers: 
    # Beta, Gamma, Moving Mean, Moving Variance parameters
    for idx, bn_layer in enumerate(net.bn_layers):
        bn_weights = bn_layer.get_weights()
        # Typically: gamma, beta, moving_mean, moving_variance
        # Let's print distributions if they exist
        # Note: After initialization, moving_mean and moving_variance might be zeros/ones.
        if len(bn_weights) == 4:
            gamma, beta, moving_mean, moving_variance = bn_weights
            print(f"BN layer {idx} gamma shape: {gamma.shape}, mean: {np.mean(gamma):.4f}, std: {np.std(gamma):.4f}")
            print(f"BN layer {idx} beta shape: {beta.shape}, mean: {np.mean(beta):.4f}, std: {np.std(beta):.4f}")
            print(f"BN layer {idx} moving_mean shape: {moving_mean.shape}, mean: {np.mean(moving_mean):.4f}, std: {np.std(moving_mean):.4f}")
            print(f"BN layer {idx} moving_variance shape: {moving_variance.shape}, mean: {np.mean(moving_variance):.4f}, std: {np.std(moving_variance):.4f}")
        else:
            print(f"BN layer {idx} unexpected number of parameters: {len(bn_weights)}")

    # -------------------------------------------------------
    # Test Input/Output Characteristics
    # -------------------------------------------------------
    # Create input data to test forward pass
    batch_size = 32
    input_dim = mock_config.eqn_config.dim
    test_input_data = np.tile(np.linspace(0, 1, input_dim), (batch_size, 1)).astype(np.float32)

    # Print input distribution
    print("\nInput Data Characteristics:")
    print(f"Input shape: {test_input_data.shape}")
    print(f"Input mean: {np.mean(test_input_data):.4f}, Input std: {np.std(test_input_data):.4f}")

    # Convert to tf.Tensor
    test_input = tf.convert_to_tensor(test_input_data)

    # -------------------------------------------------------
    # Forward Pass in Training Mode
    # -------------------------------------------------------
    output_training = net(test_input, training=True)
    output_training_np = output_training.numpy()
    print("\nOutput (Training Mode) Characteristics:")
    print(f"Output shape: {output_training_np.shape}")
    print(f"Output mean: {np.mean(output_training_np):.4f}, std: {np.std(output_training_np):.4f}")

    # -------------------------------------------------------
    # Forward Pass in Inference Mode
    # -------------------------------------------------------
    output_inference = net(test_input, training=False)
    output_inference_np = output_inference.numpy()
    print("\nOutput (Inference Mode) Characteristics:")
    print(f"Output shape: {output_inference_np.shape}")
    print(f"Output mean: {np.mean(output_inference_np):.4f}, std: {np.std(output_inference_np):.4f}")

    # -------------------------------------------------------
    # Consistency Checks over Multiple Training Passes
    # -------------------------------------------------------
    print("\nTesting Batch Normalization Behavior Over Multiple Training Passes:")
    for i in range(3):
        output = net(test_input, training=True).numpy()
        print(f"Training pass {i+1}: mean={np.mean(output):.4f}, std={np.std(output):.4f}")

    # -------------------------------------------------------
    # Parameter Count and Distribution After Forward Pass
    # -------------------------------------------------------
    # Re-check total parameters
    print("\nModel Parameter Summary:")
    total_params = 0
    for idx, layer in enumerate(net.dense_layers):
        params_count = sum([np.prod(v.shape) for v in layer.get_weights()])
        total_params += params_count
        print(f"Dense layer {idx}: parameter count={params_count}")
    # Include BN layers as well
    for idx, bn_layer in enumerate(net.bn_layers):
        params_count = sum([np.prod(v.shape) for v in bn_layer.get_weights()])
        total_params += params_count
        print(f"BN layer {idx}: parameter count={params_count}")
    print(f"Total trainable parameters (including BN): {total_params}")

    # -------------------------------------------------------
    # Test with Different Batch Sizes for Input
    # -------------------------------------------------------
    print("\nTesting Different Batch Sizes:")
    for bsz in [1, 16, 64]:
        test_input_var = np.tile(np.linspace(0, 1, input_dim), (bsz, 1)).astype(np.float32)
        test_input_var = tf.convert_to_tensor(test_input_var)
        out_var = net(test_input_var, training=False).numpy()
        print(f"Batch size {bsz}: Input {test_input_var.shape} -> Output {out_var.shape}")
