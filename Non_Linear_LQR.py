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
def initialize_trajectory(N, state_dim, control_dim, start_state, goal_state):
    s_ref = np.zeros((N, state_dim))
    u_ref = np.zeros((N - 1, control_dim))
    
    # Linearly interpolate between start_state and goal_state for s_ref
    for i in range(state_dim):
        s_ref[:, i] = np.linspace(start_state[i], goal_state[i], N)
    
    # Initialize u_ref with zero acceleration and zero steering angle
    u_ref[:, 0] = 0  # zero steering angle
    u_ref[:, 1] = 0  # zero acceleration

    return s_ref, u_ref

# Define the Cost to Minimize
def cost_function(s, u, Q, R):
    return s.T @ Q @ s + u.T @ R @ u


# iLQR Algorithm:
def ilqr(f, s0, s_ref, u_ref, Q, R, QN, params, max_iter=100):
    N = len(s_ref)
    state_dim = len(s0)
    control_dim = len(u_ref[0])

    s_ref = s_ref.copy()
    u_ref = u_ref.copy()


    s = np.zeros((N, state_dim))
    s[0] = s0
    u = u_ref.copy()


    for _ in range(max_iter):
        # Linearize the dynamics around the reference trajectory.
        A = np.zeros((N-1, state_dim, state_dim))
        B = np.zeros((N-1, state_dim, control_dim))
        

        for k in range(N-1):
            A[k] = np.eye(state_dim) + params['dt'] * np.eye(state_dim) # placeholder for actual jacobian
            B[k] = params['dt'] * np.eye(state_dim, control_dim)

        #solve the discrete-time algebraic Riccati equation
        P = np.zeros((N, state_dim, state_dim))
        P[-1] = QN
        K = np.zeros((N-1, control_dim, state_dim))


        for k in range(N-2, -1, -1):
            Q_k = Q + A[k].T @ P[k+1] @ A[k]
            R_k = R + B[k].T @ P[k+1] @ B[k]
            K[k] = np.linalg.solve(R_k, B[k].T @ P[k+1] @ A[k])
            P[k] = Q_k - K[k].T @ R_k @ K[k]

        
        #update control sequence 
        for k in range(N-1):
            u[k] = -K[k] @ (s[k] - s_ref[k])

        
        #update state trajectory
        for k in range(N-1):
            s[k+1] = f(s[k], u[k], params)

    
    return s, u


# def non_linear_lqr(f, s0, s_ref, u_ref, Q, R, QN, params, max_iter=100):
#     N = len(s_ref)
#     state_dim = len(s0)
#     control_dim = len(u_ref[0])

#     s = s_ref.copy()
#     u = u_ref.copy()


#     for _ in range(max_iter):
#         # Linearize the dynamics around the reference trajectory.



                



        # Perform backward pass to update the control inputs.
        # Perform forward pass to update the state trajectory.
    







# Cost Function:
# Define the cost function to minimize the deviation from the desired trajectory.


# Parameters
params = {'lr': 1.0, 'lf': 1.0, 'dt': dt}
Q = np.eye(4)
R = np.eye(2)
QN = np.eye(4)

# Initial state
s0 = np.array([0.0, 0.0, 0.0, 1.0])

# Goal state
goal_state = np.array([10.0, 0.0, 0.0, 1.0])

# Initialize trajectory
N = 50
s_ref, u_ref = initialize_trajectory(N, 4, 2, s0, goal_state)

# Apply iLQR
s, u = ilqr(discrete_bicycle_model, s0, s_ref, u_ref, Q, R, QN, params)

# Plot the trajectory
plt.figure(figsize=(10, 5))
plt.plot(s[:, 0], s[:, 1], label='Generated Trajectory')
plt.plot(s_ref[:, 0], s_ref[:, 1], 'r--', label='Reference Trajectory')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Reference Trajectory from iLQR')
plt.legend()
plt.grid(True)
plt.show()
plt.savefig('trajectory.png')
