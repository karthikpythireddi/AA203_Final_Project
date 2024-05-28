# iLQR Implementation Code


import numpy as np
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
from scipy.integrate import odeint

# Model Dynamics:
# Define the kinematic bicycle model equations.

def bicycle_model(state, control, params):
    x, y, psi, v = state # x, y, yaw angle, velocity
    delta_f, a = control # Steering angle, Acceleration
    lr, lf = params['lr'], params['lf']  # Distance from rear wheel to center of mass, Distance from front wheel to center of mass
    L = lr + lf # Wheelbase

    
    beta = jnp.arctan(lr * jnp.tan(delta_f) / L)   # Sideslip angle
    
    dx = v * jnp.cos(psi + beta) # x velocity
    dy = v * jnp.sin(psi + beta) # y velocity
    dpsi = v / lr * jnp.sin(beta)# yaw rate
    dv = a # acceleration
    
    return jnp.array([dx, dy, dpsi, dv])

def road_model(state, control, params):
    e_y, e_psi, v = state # lane normal displacement, lane heading difference, velocity
    delta_f, a = control # Steering Angle, Acceleration
    lr, lf = params['lr'], params['lf']  # Distance from rear wheel to center of mass, Distance from front wheel to center of mass
    L = lr + lf # Wheelbase
    R = params['R'] # Road curvature

    beta = jnp.arctan(lr * jnp.tan(delta_f) / L)   # Sideslip angle

    de_y = v * jnp.sin(e_psi + beta)
    de_psi = v * (1/lr * jnp.sin(beta) - 1 / R * jnp.cos(beta)) 
    dv = a # acceleration

    return jnp.array([de_y, de_psi, dv])


# Discretize the model using Euler's method.
def discretize_dynamics(f, dt):
    def fd(state, control, params):
        return state + dt * f(state, control, params)
    return fd


def linearize(f, s, u, params):
    A, B = jax.jacobian(f, (0, 1))(s, u, params)
    return A, B

dt = 0.1  # time step
discrete_bicycle_model = jax.jit(discretize_dynamics(bicycle_model, dt))
discrete_road_model = jax.jit(discretize_dynamics(road_model, dt))


"""
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
    """
# Generate road trajectory with constant radius of curvature, initial heading of 0 and specified final heading
def initialize_trajectory(N, params, final_heading, v0, v_end):
    r = params['R'] # Road curvature
    theta = np.linspace(0, final_heading, N)
    road_x = r * np.sin(theta) # road x position
    road_y = r * (1 - np.cos(theta)) # road y position
    road_dx = r * np.cos(theta)
    road_dy = r * np.sin(theta)
    road_psi = np.arctan2(road_dy, road_dx) # road heading

    s_ref = np.zeros((N, 4))
    s_ref[:, 0], s_ref[:, 1], s_ref[:, 2] = road_x, road_y, road_psi
    s_ref[:, 3] = np.linspace(v0, v_end, N)

    return s_ref
   

# iLQR Algorithm:
def ilqr(f, s0, s_ref, Q, R, QN, params, eps=1e-3, max_iter=100):
    if max_iter <= 1:
        raise ValueError("Argument `max_iter` must be at least 1.")
    n = Q.shape[0]  # state dimension
    m = R.shape[0]  # control dimension
    N = s_ref.shape[0]

    # Initialize gains `Y` and offsets `y` for the policy
    Y = np.zeros((N - 1, m, n))
    y = np.zeros((N - 1, m))

    # Initialize the nominal trajectory `(s_bar, u_bar`), and the
    # deviations `(ds, du)`
    u_bar = np.zeros((N - 1, m))
    s_bar = np.zeros((N, n))
    s_bar[0] = s0
    for k in range(N - 1):
        s_bar[k + 1] = f(s_bar[k], u_bar[k], params)

    ds = np.zeros((N, n))
    du = np.zeros((N - 1, m))

    # iLQR loop
    converged = False
    for i in range(max_iter):
        # Linearize the dynamics at each step `k` of `(s_bar, u_bar)`
        A, B = jax.vmap(linearize, in_axes=(None, 0, 0, None))(f, s_bar[:-1], u_bar, params)
        A, B = jnp.array(A), jnp.array(B)
       # Update `Y`, `y`, `ds`, `du`, `s_bar`, and `u_bar`.
        qN = QN @ (s_bar[-1] - s_ref[-1])
        qK = np.array([Q @ (s_bar[k] - s_ref[k]) for k in range(N)])
        rK = np.array([R @ u_bar[k] for k in range(N-1)])

        v_s, v_ss = qN, QN

        # Backward Pass
        for k in range(N-2, -1, -1):
            c_s, c_u, c_ss, c_uu = qK[k], rK[k], Q, R

            q_s = c_s + A[k].T @ v_s 
            q_u = c_u + B[k].T @ v_s
            q_ss = c_ss + A[k].T @ v_ss @ A[k]
            q_uu = c_uu + B[k].T @ v_ss @ B[k]
            q_us = B[k].T @ v_ss @ A[k]

            y[k] = -jnp.linalg.inv(q_uu) @ q_u
            Y[k] = -jnp.linalg.inv(q_uu) @ q_us

            v_s = q_s - Y[k].T @ q_uu @ y[k]
            v_ss = q_ss - Y[k].T @ q_uu @ Y[k]

        # Forward Pass
        for k in range(N - 1):
            du[k] = Y[k] @ ds[k] + y[k]
            s_new = f(s_bar[k], du[k] + u_bar[k], params)
            ds[k+1] = s_new - s_bar[k+1]
            s_bar[k+1] = s_new
        u_bar = u_bar + du

        if np.max(np.abs(du)) < eps:
            converged = True
            print("converged in " + str(i) + " iterations")
            break
    if not converged:
        raise RuntimeError("iLQR did not converge!")
    return s_bar, u_bar, Y, y


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
params = {'lr': 1.0, 'lf': 1.0, 'dt': dt, 'R': 100.0}
Q = 1e2 * np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 0]])
R = np.array([[1, 0], [0, 100]])
QN = 1e2 * np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 0]])
T = 10.0  # time to reach end of road
t = np.arange(0.0, T, dt)
N = t.size - 1

# Initial and final velocities
v0 = 0 # initial velocity
v_end = 2 # end velocity

# Generate road trajectory with constant radius of curvature, initial heading of 0 and final heading of 90 deg
s_ref = initialize_trajectory(N, params, np.pi/2, v0, v_end)
s0_car = s_ref[0]
sgoal_car = s_ref[-1]

# Apply iLQR
s_bar_car, u_bar, Y, y = ilqr(discrete_bicycle_model, s0_car, s_ref, Q, R, QN, params)

# Simulate control to generate position trajectory
s_car = np.zeros((N, s0_car.shape[0]))
s_car[0] = s0_car
u = np.zeros((N - 1, 2))
for k in range(N - 1):
    u[k] = u_bar[k] + Y[k] @ (s_car[k] - s_bar_car[k]) + y[k]
    s_car[k + 1] = odeint(lambda s, t: bicycle_model(s, u[k], params), s_car[k], t[k : k + 2])[1]


# Plot the positional trajectory
plt.figure(figsize=(10, 5))
plt.plot(s_car[:, 0], s_car[:, 1], label='Generated Trajectory')
plt.plot(s_ref[:, 0], s_ref[:, 1], 'r--', label='Reference Road')
plt.xlabel('x')
plt.ylabel('y')
plt.title('State Reference Trajectory from iLQR')
plt.legend()
plt.grid(True)
plt.show()
plt.savefig('trajectory.png')

