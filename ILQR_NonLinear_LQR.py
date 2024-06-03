import numpy as np
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
from scipy.integrate import odeint

# Model Dynamics:
# Define the kinematic bicycle model equations.
def bicycle_model(state, control, params):
    x, y, psi, v = state  # x, y, yaw angle, velocity
    delta_f, a = control  # Steering angle, Acceleration
    lr, lf = params['lr'], params['lf']  # Distance from rear wheel to center of mass, Distance from front wheel to center of mass
    L = lr + lf  # Wheelbase
    
    beta = jnp.arctan(lr * jnp.tan(delta_f) / L)  # Sideslip angle
    
    dx = v * jnp.cos(psi + beta)  # x velocity
    dy = v * jnp.sin(psi + beta)  # y velocity
    dpsi = v / lr * jnp.sin(beta)  # yaw rate
    dv = a  # acceleration
    
    return jnp.array([dx, dy, dpsi, dv])

def road_model(state, control, params):
    e_y, e_psi, v = state  # lane normal displacement, lane heading difference, velocity
    delta_f, a = control  # Steering Angle, Acceleration
    lr, lf = params['lr'], params['lf']  # Distance from rear wheel to center of mass, Distance from front wheel to center of mass
    L = lr + lf  # Wheelbase
    R = params['R']  # Road curvature

    beta = jnp.arctan(lr * jnp.tan(delta_f) / L)  # Sideslip angle

    de_y = v * jnp.sin(e_psi + beta)
    de_psi = v * (1 / lr * jnp.sin(beta) - 1 / R * jnp.cos(beta)) 
    dv = a  # acceleration

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

# Initialization:
def initialize_trajectory(N, params, final_heading, v0, v_end):
    r = params['R']  # Road curvature
    theta = np.linspace(0, final_heading, N)
    road_x = r * np.sin(theta)  # road x position
    road_y = r * (1 - np.cos(theta))  # road y position
    road_dx = r * np.cos(theta)
    road_dy = r * np.sin(theta)
    road_psi = np.arctan2(road_dy, road_dx)  # road heading

    s_ref = np.zeros((N, 4))
    s_ref[:, 0], s_ref[:, 1], s_ref[:, 2] = road_x, road_y, road_psi
    s_ref[:, 3] = np.linspace(v0, v_end, N)

    return s_ref

# Define the running cost and terminal cost
def running_cost(state, control, step):
    return jnp.dot(state.T, Q @ state) + jnp.dot(control.T, R @ control)

def terminal_cost(state):
    return jnp.dot(state.T, QN @ state)

def total_cost(states, controls):
    running_costs = jax.vmap(running_cost)(states[:-1], controls, jnp.arange(states.shape[0] - 1))
    return jnp.sum(running_costs) + terminal_cost(states[-1])

# Define the iLQR function
@jax.jit
def iterative_linear_quadratic_regulator(dynamics, total_cost, x0, u_guess, maxiter=100, atol=1e-3):
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
            l, l_x = policy
            return rollout_state_feedback_policy(dynamics, AffinePolicy(alpha * l, l_x), x0, step_range, xs, us)

        all_xs, all_us = jax.vmap(rollout_linesearch_policy)(0.5 ** jnp.arange(16))
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

# Generate road trajectory with constant radius of curvature, initial heading of 0 and final heading of 90 degrees
s_ref = initialize_trajectory(N, params, np.pi, v0, v_end)
s0_car = s_ref[0]
sgoal_car = s_ref[-1]

# Initial guess for control inputs (steering angle and acceleration)
u_guess = np.zeros((N, 2))

# Apply iLQR
total_cost = (running_cost, terminal_cost)
result = iterative_linear_quadratic_regulator(discrete_bicycle_model, total_cost, s0_car, u_guess)

# Extract optimal trajectories and controls
s_bar_car, u_bar = result["optimal_trajectory"]

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
