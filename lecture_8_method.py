import jax
import jax.numpy as jnp
import numpy as np

import matplotlib.pyplot as plt; plt.rcParams.update({'font.size': 20})
from ipywidgets import interact, interactive

from typing import Callable, NamedTuple

class LinearDynamics(NamedTuple):
    f_x: jnp.array  # A
    f_u: jnp.array  # B

    def __call__(self, x, u, k=None):
        f_x, f_u = self
        return f_x @ x + f_u @ u if k is None else self[k](x, u)

    def __getitem__(self, key):
        return jax.tree.map(lambda x: x[key], self)


class AffinePolicy(NamedTuple):
    l: jnp.array  # l
    l_x: jnp.array  # L

    def __call__(self, x, k=None):
        l, l_x = self
        return l + l_x @ x if k is None else self[k](x)

    def __getitem__(self, key):
        return jax.tree.map(lambda x: x[key], self)


class QuadraticCost(NamedTuple):
    c: jnp.array  # c
    c_x: jnp.array  # q
    c_u: jnp.array  # r
    c_xx: jnp.array  # Q
    c_uu: jnp.array  # R
    c_ux: jnp.array  # H.T

    @classmethod
    def from_pure_quadratic(cls, c_xx, c_uu, c_ux):
        return cls(
            jnp.zeros((c_xx.shape[:-2])),
            jnp.zeros(c_xx.shape[:-1]),
            jnp.zeros(c_uu.shape[:-1]),
            c_xx,
            c_uu,
            c_ux,
        )

    def __call__(self, x, u, k=None):
        c, c_x, c_u, c_xx, c_uu, c_ux = self
        return c + c_x @ x + c_u @ u + x @ c_xx @ x / 2 + u @ c_uu @ u / 2 + u @ c_ux @ x if k is None else self[k](x)

    def __getitem__(self, key):
        return jax.tree.map(lambda x: x[key], self)


class QuadraticStateCost(NamedTuple):
    v: jnp.array  # p (scalar)
    v_x: jnp.array  # p (vector)
    v_xx: jnp.array  # P

    @classmethod
    def from_pure_quadratic(cls, v_xx):
        return cls(
            jnp.zeros(v_xx.shape[:-2]),
            jnp.zeros(v_xx.shape[:-1]),
            v_xx,
        )

    def __call__(self, x, k=None):
        v, v_x, v_xx = self
        return v + v_x @ x + x @ v_xx @ x / 2 if k is None else self[k](x)

    def __getitem__(self, key):
        return jax.tree.map(lambda x: x[key], self)


def rollout_state_feedback_policy(dynamics, policy, x0, step_range, x_nom=None, u_nom=None):

    def scan_fn(x, k):
        u = policy(x, k) if x_nom is None else u_nom[k] + policy(x - x_nom[k], k)
        x1 = dynamics(x, u, k)
        return (x1, (x1, u))

    xs, us = jax.lax.scan(scan_fn, x0, step_range)[1]
    return jnp.concatenate([x0[None], xs]), us

def riccati_step(
    current_step_dynamics: LinearDynamics,
    current_step_cost: QuadraticCost,
    next_state_value: QuadraticStateCost,
):
    f_x, f_u = current_step_dynamics
    c, c_x, c_u, c_xx, c_uu, c_ux = current_step_cost
    v, v_x, v_xx = next_state_value

    q = c + v
    q_x = c_x + f_x.T @ v_x
    q_u = c_u + f_u.T @ v_x
    q_xx = c_xx + f_x.T @ v_xx @ f_x
    q_uu = c_uu + f_u.T @ v_xx @ f_u
    q_ux = c_ux + f_u.T @ v_xx @ f_x

    l = -jnp.linalg.solve(q_uu, q_u)
    l_x = -jnp.linalg.solve(q_uu, q_ux)

    current_state_value = QuadraticStateCost(
        q - l.T @ q_uu @ l / 2,
        q_x - l_x.T @ q_uu @ l,
        q_xx - l_x.T @ q_uu @ l_x,
    )
    current_step_optimal_policy = AffinePolicy(l, l_x)
    return current_state_value, current_step_optimal_policy


def ensure_positive_definite(a, eps=1e-3):
    w, v = jnp.linalg.eigh(a)
    return (v * jnp.maximum(w, eps)) @ v.T


class TotalCost(NamedTuple):
    running_cost: Callable
    terminal_cost: Callable

    def __call__(self, xs, us):
        step_range = jnp.arange(us.shape[0])
        return jnp.sum(jax.vmap(self.running_cost)(xs[:-1], us, step_range)) + self.terminal_cost(xs[-1])
    

@jax.jit
def iterative_linear_quadratic_regulator(dynamics, total_cost, x0, u_guess, maxiter=1000, atol=1e-3):
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



'''

##### PROBLEM SPECIFIC INFO BEGINS HERE #####
#############################################
#############################################
#############################################
#############################################

'''

class BicycleDynamics(NamedTuple):
    dt: float = 0.1
    lr: float = 1.0
    lf: float = 1.0

    def __call__(self, s, u, k):
        x, y, psi, v = s # x, y, yaw angle, velocity
        delta_f, a = u # Steering angle, Acceleration
        L = self.lr + self.lf # Wheelbase
        beta = jnp.arctan(self.lr * jnp.tan(delta_f) / L)   # Sideslip angle
    
        dx = v * jnp.cos(psi + beta) # x velocity
        dy = v * jnp.sin(psi + beta) # y velocity
        dpsi = v / self.lr * jnp.sin(beta)# yaw rate
        dv = a # acceleration
    
        return jnp.array([x + self.dt * dx, y + self.dt * dy, psi + self.dt * dpsi, v + self.dt * dv])


class RunningCost(NamedTuple):
    Q: jnp.array
    R: jnp.array
    s_ref: jnp.array
    gain: float = 1.0


    def __call__(self, s, u, k):
        return self.gain * ((s - self.s_ref[k]).T @ self.Q @ (s - self.s_ref[k]) + u.T @ self.R @ u)


class TerminalCost(NamedTuple):
    QN: jnp.array
    s_ref: jnp.array
    gain: float = 1.0

    def __call__(self, s):
        return self.gain * ((s - self.s_ref[-1]).T @ self.QN @ (s - self.s_ref[-1]))
    


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
    

dt = 0.1
params = {'lr': 1.0, 'lf': 1.0, 'dt': dt, 'R': 100.0}
Q = 1.0e1 * np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
R = 1.0e1 * np.array([[1, 0], [0, 1]])
QN = 1.0e1 * np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
T = 15.0  # simulation time (decreasing makes car understeer)
t = np.arange(0.0, T, dt)
N = t.size - 1

# Initial and final velocities
v0 = 5
v_end = 5

s_ref = initialize_trajectory(N, params, np.pi/2, v0, v_end)
s0_car, sgoal_car = s_ref[0], s_ref[-1]

u_guess = jnp.zeros((N - 1, 2))

solution = iterative_linear_quadratic_regulator(
    BicycleDynamics(),
    TotalCost(RunningCost(Q= Q, R= R, s_ref= s_ref), TerminalCost(QN= QN, s_ref= s_ref)),
    s0_car,
    u_guess,
)
i = solution["num_iterations"]
print("Converged after " + str(i) + " iterations.")

all_s = solution["trajectory_iterates"][0][:i]
all_u = solution["trajectory_iterates"][1][:i]

x_car, y_car = all_s[-1][:, 0], all_s[-1][:, 1]
delta_f, a = all_u[-1][:, 0], all_u[-1][:, 1]



# Plot the positional trajectory
plt.figure(figsize=(10, 5))
plt.plot(x_car, y_car, label='Generated Trajectory')
plt.plot(s_ref[:,0], s_ref[:,1], 'r--', label='Reference Road')
plt.xlabel('x')
plt.ylabel('y')
plt.title('State Reference Trajectory from iLQR')
plt.legend()
plt.grid(True)
plt.savefig('trajectory.png')
plt.show()

# Plot the control trajectory
plt.figure(figsize=(10, 5))
plt.plot(delta_f, label='Steering Angle')
plt.plot(a, label='Acceleration')
plt.xlabel('Time Step')
plt.ylabel('Value')
plt.title('Control Reference Trajectory from iLQR')
plt.legend()
plt.grid(True)
plt.savefig('control_trajectory.png')
plt.show()

