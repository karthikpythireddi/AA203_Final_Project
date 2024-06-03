import numpy as np
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
from scipy.integrate import odeint

# Define helper functions and classes
class LinearDynamics:
    def __init__(self, f_x, f_u):
        self.f_x = f_x
        self.f_u = f_u

class QuadraticCost:
    def __init__(self, c, c_x, c_u, c_xx, c_uu, c_ux):
        self.c = c
        self.c_x = c_x
        self.c_u = c_u
        self.c_xx = c_xx
        self.c_uu = c_uu
        self.c_ux = c_ux

class QuadraticStateCost:
    def __init__(self, v, v_x, v_xx):
        self.v = v
        self.v_x = v_x
        self.v_xx = v_xx

    @staticmethod
    def from_pure_quadratic(pure_quadratic):
        return QuadraticStateCost(
            pure_quadratic[..., -1],
            pure_quadratic[..., :-1],
            pure_quadratic[..., :-1, :-1],
        )

def ensure_positive_definite(matrix):
    eigval, eigvec = jnp.linalg.eigh(matrix)
    eigval = jnp.maximum(eigval, 1e-5)
    return eigvec @ jnp.diag(eigval) @ eigvec.T

def rollout_state_feedback_policy(dynamics, policy, x0, step_range):
    def step(x, u):
        return dynamics(x, u)
    xs, us = [x0], []
    for k in step_range:
        u = policy(xs[-1], k)
        us.append(u)
        xs.append(step(xs[-1], u))
    return jnp.array(xs), jnp.array(us)

def riccati_step(current_step_dynamics, current_step_cost, next_state_value):
    f_x, f_u = current_step_dynamics.f_x, current_step_dynamics.f_u
    c, c_x, c_u, c_xx, c_uu, c_ux = current_step_cost.c, current_step_cost.c_x, current_step_cost.c_u, current_step_cost.c_xx, current_step_cost.c_uu, current_step_cost.c_ux
    v, v_x, v_xx = next_state_value.v, next_state_value.v_x, next_state_value.v_xx

    q = c + v
    q_x = c_x + f_x.T @ v_x
    q_u = c_u + f_u.T @ v_x
    q_xx = c_xx + f_x.T @ v_xx @ f_x
    q_uu = c_uu + f_u.T @ v_xx @ f_u
    q_ux = c_ux + f_u.T @ v_xx @ f_x

    q_uu = ensure_positive_definite(q_uu)
    k = -jnp.linalg.inv(q_uu) @ q_u
    K = -jnp.linalg.inv(q_uu) @ q_ux

    v_x = q_x + K.T @ q_uu @ k + K.T @ q_u + q_ux.T @ k
    v_xx = q_xx + K.T @ q_uu @ K + K.T @ q_ux + q_ux.T @ K

    return QuadraticStateCost(v, v_x, v_xx), (k, K)

@jax.jit
def iterative_linear_quadratic_regulator(dynamics, total_cost, x0, u_guess, maxiter=5000, atol=1e-3):
    running_cost, terminal_cost = total_cost
    n, (N, m) = x0.shape[-1], u_guess.shape
    step_range = jnp.arange(N)

    xs_iterates, us_iterates = jnp.zeros((maxiter, N + 1, n)), jnp.zeros((maxiter, N, m))
    xs, us = rollout_state_feedback_policy(dynamics, lambda x, k: u_guess[k], x0, step_range)
    xs_iterates, us_iterates = xs_iterates.at[0].set(xs), us_iterates.at[0].set(us)
    j_curr = total_cost(xs, us)
    value_functions_iterates = QuadraticStateCost.from_pure_quadratic(jnp.zeros((maxiter, N + 1, n, n)))

    def continuation_criterion(loop_vars):
        i, _, _, j_curr, j_prev, _ = loop_vars
        return (j_curr < j_prev - atol) & (i < maxiter)

    def ilqr_iteration(loop_vars):
        i, xs_iterates, us_iterates, j_curr, j_prev, value_functions_iterates = loop_vars
        xs, us = xs_iterates[i], us_iterates[i]

        f_x, f_u = jax.vmap(jax.jacobian(dynamics, (0, 1)))(xs[:-1], us, step_range)
        c = jax.vmap(running_cost)(xs[:-1], us, step_range)
        c_x, c_u = jax.vmap(jax.grad(running_cost, (0, 1)))(xs[:-1], us, step_range)
        (c_xx, c_xu), (c_ux, c_uu) = jax.vmap(jax.hessian(running_cost, (0, 1)))(xs[:-1], us, step_range)
        v, v_x, v_xx = terminal_cost(xs[-1]), jax.grad(terminal_cost)(xs[-1]), jax.hessian(terminal_cost)(xs[-1])

        # Ensure quadratic cost terms are positive definite.
        c_zz = jnp.block([[c_xx, c_xu], [c_ux, c_uu]])
        c_zz = jax.vmap(ensure_positive_definite)(c_zz)
        c_xx, c_uu, c_ux = c_zz[:, :n, :n], c_zz[:, -m:, -m:], c_zz[:, -m:, :n]
        v_xx = ensure_positive_definite(v_xx)

        linearized_dynamics = LinearDynamics(f_x, f_u)
        quadratized_running_cost = QuadraticCost(c, c_x, c_u, c_xx, c_uu, c_ux)
        quadratized_terminal_cost = QuadraticStateCost(v, v_x, v_xx)

        def scan_fn(next_state_value, current_step_dynamics_cost):
            current_step_dynamics, current_step_cost = current_step_dynamics_cost
            current_state_value, current_step_policy = riccati_step(
                current_step_dynamics,
                current_step_cost,
                next_state_value,
            )
            return current_state_value, (current_state_value, current_step_policy)

        value_functions, policy = jax.lax.scan(scan_fn,
                                               quadratized_terminal_cost,
                                               (linearized_dynamics, quadratized_running_cost),
                                               reverse=True)[1]
        value_functions_iterates = jax.tree.map(lambda x, xi, xiN: x.at[i].set(jnp.concatenate([xi, xiN[None]])),
                                                value_functions_iterates, value_functions, quadratized_terminal_cost)

        def rollout_linesearch_policy(alpha):
            # Note that we roll out the true `dynamics`, not the `linearized_dynamics`!
            l, l_x = policy
            return rollout_state_feedback_policy(dynamics, AffinePolicy(alpha * l, l_x), x0, step_range, xs, us)

        # Backtracking line search (step sizes evaluated in parallel).
        all_xs, all_us = jax.vmap(rollout_linesearch_policy)(0.5**jnp.arange(16))
        js = jax.vmap(total_cost)(all_xs, all_us)
        a = jnp.argmin(js)
        j = js[a]
        xs_iterates = xs_iterates.at[i + 1].set(jnp.where(j < j_curr, all_xs[a], xs))
        us_iterates = us_iterates.at[i + 1].set(jnp.where(j < j_curr, all_us[a], us))
        return i + 1, xs_iterates, us_iterates, jnp.minimum(j, j_curr), j_curr, value_functions_iterates

    i, xs_iterates, us_iterates, j_curr, j_prev, value_functions_iterates = jax.lax.while_loop(
        continuation_criterion, ilqr_iteration,
        (0, xs_iterates, us_iterates, j_curr, jnp.inf, value_functions_iterates))

    return {
        "optimal_trajectory": (xs_iterates[i], us_iterates[i]),
        "optimal_cost": j_curr,
        "num_iterations": i,
        "trajectory_iterates": (xs_iterates, us_iterates),
        "value_functions_iterates": value_functions_iterates
    }

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

dt = 0.1  # time step
discrete_bicycle_model = jax.jit(discretize_dynamics(bicycle_model, dt))
discrete_road_model = jax.jit(discretize_dynamics(road_model, dt))

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

# Define cost functions
def running_cost(xs, us, step_range):
    Q = jnp.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    R = jnp.array([[1, 0], [0, 1]])
    return jnp.sum(Q * xs**2) + jnp.sum(R * us**2)

def terminal_cost(x):
    QN = jnp.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    return jnp.sum(QN * x**2)

# iLQR Algorithm using the provided implementation
def ilqr(dynamics, total_cost, x0, u_guess, params):
    result = iterative_linear_quadratic_regulator(dynamics, total_cost, x0, u_guess)
    return result['optimal_trajectory']

# Parameters
params = {'lr': 1.0, 'lf': 1.0, 'dt': dt, 'R': 100.0}
Q = 1.0e1 * np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
R = 1.0e4 * np.array([[1, 0], [0, 1]])
QN = 1.0e1 * np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
T = 30.0  # time to reach end of road
t = np.arange(0.0, T, dt)
N = t.size - 1

# Initial and final velocities
v0 = 5  # initial velocity
v_end = 5  # end velocity

# Generate road trajectory with constant radius of curvature, initial heading of 0 and final heading of 90 deg
s_ref = initialize_trajectory(N, params, np.pi, v0, v_end)
s0_car = s_ref[0]
sgoal_car = s_ref[-1]

# Initial guess for control inputs
u_guess = np.zeros((N, 2))

# Apply iLQR
total_cost = (running_cost, terminal_cost)
optimal_trajectory = ilqr(discrete_bicycle_model, total_cost, s0_car, u_guess, params)
s_bar_car, u_bar = optimal_trajectory

# Simulate control to generate position trajectory
s_car = np.zeros((N, s0_car.shape[0]))
s_car[0] = s0_car
u = np.zeros((N - 1, 2))
for k in range(N - 1):
    u[k] = u_bar[k] + Y[k] @ (s_car[k] - s_bar_car[k]) + y[k]
    s_car[k + 1] = odeint(lambda s, t: bicycle_model(s, u[k], params), s_car[k], t[k : k + 2])[1]

def nonlinear_lqr(f, s_ref, u_ref, Q, R, QN, params):
    N = s_ref.shape[0]
    n = Q.shape[0]  # state dimension
    m = R.shape[0]  # control dimension

    # Initialize gains and value function
    K = np.zeros((N - 1, m, n))
    v_s = np.zeros((N, n))
    v_ss = np.zeros((N, n, n))

    # Initialize value function at terminal state
    v_s[-1] = QN @ (s_ref[-1] - s_ref[-1])
    v_ss[-1] = QN

    # Backward pass to solve Riccati equations
    for k in range(N - 2, -1, -1):
        A, B = linearize(f, s_ref[k], u_ref[k], params)

        q_s = Q @ (s_ref[k] - s_ref[k])
        q_u = R @ u_ref[k]
        q_ss = Q
        q_uu = R
        q_us = np.zeros((m, n))

        # Riccati recursion
        Q_s = q_s + A.T @ v_s[k + 1]
        Q_u = q_u + B.T @ v_s[k + 1]
        Q_ss = q_ss + A.T @ v_ss[k + 1] @ A
        Q_uu = q_uu + B.T @ v_ss[k + 1] @ B
        Q_us = q_us + B.T @ v_ss[k + 1] @ A

        K[k] = -np.linalg.inv(Q_uu) @ Q_us
        v_s[k] = Q_s - K[k].T @ Q_uu @ K[k] @ Q_s
        v_ss[k] = Q_ss - K[k].T @ Q_uu @ K[k]

    return K

# Apply Nonlinear LQR to track the iLQR reference trajectory
K = nonlinear_lqr(discrete_bicycle_model, s_bar_car, u_bar, Q, R, QN, params)

# Simulate the closed-loop system with NLQR controller
s_nlqr = np.zeros_like(s_car)
s_nlqr[0] = s0_car
u_nlqr = np.zeros_like(u)

for k in range(N - 1):
    u_nlqr[k] = u_bar[k] + K[k] @ (s_nlqr[k] - s_bar_car[k])
    s_nlqr[k + 1] = odeint(lambda s, t: bicycle_model(s, u_nlqr[k], params), s_nlqr[k], t[k : k + 2])[1]

# Plot the trajectory for NLQR
plt.figure(figsize=(10, 5))
plt.plot(s_car[:, 0], s_car[:, 1], label='iLQR Trajectory')
plt.plot(s_nlqr[:, 0], s_nlqr[:, 1], 'g', label='NLQR Trajectory')
plt.plot(s_ref[:, 0], s_ref[:, 1], 'r--', label='Reference Road')
plt.xlabel('x')
plt.ylabel('y')
plt.title('State Trajectory Comparison: iLQR vs NLQR')
plt.legend()
plt.grid(True)
plt.show()
plt.savefig('trajectory_nlqr.png')
