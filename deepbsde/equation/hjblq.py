import numpy as np
import tensorflow as tf

from .base import Equation


class HJBLQ(Equation):
    """Hamilton-Jacobi-Bellman equation with Linear-Quadratic control problem.
    
    This class implements the HJB equation described in the PNAS paper:
    "Solving high-dimensional partial differential equations using deep learning"
    doi.org/10.1073/pnas.1718942115
    
    Attributes:
        x_init (ndarray): Initial state vector, initialized as zero vector
        sigma (float): Diffusion coefficient, set to sqrt(2)
        lambd (float): Control cost coefficient, set to 1.0
    """
    def __init__(self, eqn_config):
        """Initialize the HJBLQ equation with given configuration.
        
        Args:
            eqn_config: Configuration object containing PDE parameters
        """
        super(HJBLQ, self).__init__(eqn_config)
        # Initialize model parameters
        self.x_init = np.zeros(self.dim)
        self.sigma = np.sqrt(2.0)
        self.lambd = 1.0

    def sample(self, num_sample):
        """Generate sample paths for the forward SDE.
        
        Simulates the controlled diffusion process:
        dX_t = σ dW_t
        
        Args:
            num_sample (int): Number of Monte Carlo samples to generate
            
        Returns:
            tuple: (dw_sample, x_sample)
                - dw_sample: Brownian increments
                - x_sample: Sample paths of the state process
        """
        # Generate Brownian increments
        dw_sample = np.random.normal(size=[num_sample, self.dim, self.num_time_interval]) * self.sqrt_delta_t
        
        # Initialize state trajectory array
        x_sample = np.zeros([num_sample, self.dim, self.num_time_interval + 1])
        x_sample[:, :, 0] = np.ones([num_sample, self.dim]) * self.x_init
        
        # Generate forward paths using Euler-Maruyama scheme
        for i in range(self.num_time_interval):
            x_sample[:, :, i + 1] = x_sample[:, :, i] + self.sigma * dw_sample[:, :, i]
            
        return dw_sample, x_sample

    def f_tf(self, t, x, y, z):
        """Implements the driver function (drift term) of the BSDE.
        
        Args:
            t (tf.Tensor): Current time
            x (tf.Tensor): Current state
            y (tf.Tensor): Current solution value
            z (tf.Tensor): Current gradient of the solution
            
        Returns:
            tf.Tensor: Value of the driver function
        """
        # Hamiltonian term: -λ|z|²/2
        return -self.lambd * tf.reduce_sum(tf.square(z), 1, keepdims=True) / 2

    def g_tf(self, t, x):
        """Implements the terminal condition of the PDE.
        
        Args:
            t (tf.Tensor): Terminal time
            x (tf.Tensor): Terminal state
            
        Returns:
            tf.Tensor: Value of the terminal condition
        """
        # Terminal cost: log((1 + |x|²)/2)
        return tf.math.log((1 + tf.reduce_sum(tf.square(x), 1, keepdims=True)) / 2)


if __name__ == "__main__":
    # -------------------------------------------------------
    # Test Configuration
    # -------------------------------------------------------
    class Config:
        def __init__(self):
            self.dim = 3
            self.total_time = 1.0
            self.num_time_interval = 100

    # Initialize equation
    config = Config()
    equation = HJBLQ(config)
    print("\nInitialized HJBLQ equation with:")
    print(f"Dimension: {equation.dim}")
    print(f"Total time: {equation.total_time}")
    print(f"Time intervals: {equation.num_time_interval}")
    print(f"Delta t: {equation.delta_t}")
    print(f"Sigma: {equation.sigma}")
    print(f"Lambda: {equation.lambd}")

    # -------------------------------------------------------
    # Test Sample Generation
    # -------------------------------------------------------
    print("\nTesting sample generation:")
    num_test_samples = 5
    dw, x = equation.sample(num_test_samples)
    print(f"Brownian increments shape: {dw.shape}")
    print(f"State trajectories shape: {x.shape}")
    print(f"Initial state: {x[0,:,0]}")
    print(f"Final state: {x[0,:,-1]}")

    # -------------------------------------------------------
    # Test Driver Function
    # -------------------------------------------------------
    print("\nTesting driver function:")
    test_t = tf.constant(0.5)
    test_x = tf.random.normal([num_test_samples, equation.dim])
    test_y = tf.random.normal([num_test_samples, 1])
    test_z = tf.random.normal([num_test_samples, equation.dim])
    
    f_value = equation.f_tf(test_t, test_x, test_y, test_z)
    print(f"Input shapes - t: {test_t.shape}, x: {test_x.shape}, y: {test_y.shape}, z: {test_z.shape}")
    print(f"Driver function output shape: {f_value.shape}")
    print(f"Sample driver value: {f_value[0].numpy()}")

    # -------------------------------------------------------
    # Test Terminal Condition
    # -------------------------------------------------------
    print("\nTesting terminal condition:")
    test_t_final = tf.constant(equation.total_time)
    test_x_final = tf.random.normal([num_test_samples, equation.dim])
    
    g_value = equation.g_tf(test_t_final, test_x_final)
    print(f"Input shapes - t: {test_t_final.shape}, x: {test_x_final.shape}")
    print(f"Terminal condition output shape: {g_value.shape}")
    print(f"Sample terminal value: {g_value[0].numpy()}")

    # -------------------------------------------------------
    # Test Gradient Computation
    # -------------------------------------------------------
    print("\nTesting gradient computation:")
    with tf.GradientTape() as tape:
        tape.watch(test_x)
        g_value = equation.g_tf(test_t_final, test_x)
    grad = tape.gradient(g_value, test_x)
    print(f"Gradient shape: {grad.shape}")
    print(f"Sample gradient: {grad[0].numpy()}")