# iLQR Implementation Code


import numpy as np
import matplotlib.pyplot as plt


# Model Dynamics:
# Define the kinematic bicycle model equations.

def bicycle_model(state, control, params):
    x, y, psi, v = state # x, y, yaw angle, velocity
    delta_f, a = control # Steering angle, Acceleration
    lr, lf = params['lr'], params['lf']  # Distance from rear wheel to center of mass, Distance from front wheel to center of mass
    L = lr + lf # Wheelbase

    
    beta = np.arctan(lr * np.tan(delta_f) / L)   # Sideslip angle
    
    dx = v * np.cos(psi + beta) # x velocity
    dy = v * np.sin(psi + beta) # y velocity
    dpsi = v / lr * np.sin(beta)# yaw rate
    dv = a # acceleration
    
    return np.array([dx, dy, dpsi, dv])

# Discretize the model using Euler's method.
def discretize_dynamics(f, dt):
    def fd(state, control, params):
        return state + dt * f(state, control, params)
    return fd

dt = 0.1  # time step
discrete_bicycle_model = discretize_dynamics(bicycle_model, dt)


# Initialization:
def initialize_trajectory(N, state_dim, control_dim):
    s_ref = np.zeros((N, state_dim))
    u_ref = np.zeros((N - 1, control_dim))
    return s_ref, u_ref

def cost_function(s, u, Q, R):
    return s.T @ Q @ s + u.T @ R @ u


# iLQR Algorithm:
# Linearize the dynamics around the reference trajectory.
# Perform backward pass to update the control inputs.
# Perform forward pass to update the state trajectory.




# Cost Function:
# Define the cost function to minimize the deviation from the desired trajectory.