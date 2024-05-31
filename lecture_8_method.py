import jax
import jax.numpy as jnp
import numpy as np

import matplotlib.pyplot as plt; plt.rcParams.update({'font.size': 20})
from ipywidgets import interact, interactive

from typing import Callable, NamedTuple
import cvxpy as cvx
from functools import partial
from time import time
from tqdm.auto import tqdm

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
def generate_constant_curvature(N, params, final_heading, v0, v_end):
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
R = 1.0e4 * np.array([[1, 0], [0, 1]])
QN = 1.0e1 * np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
T = 30.0  # simulation time (decreasing makes car understeer)
t = np.arange(0.0, T, dt)
N = t.size - 1

# Initial and final velocities
v0 = 10
v_end = 10

s_ref = generate_constant_curvature(N, params, np.pi, v0, v_end)
s0_car, sgoal_car = s_ref[0], s_ref[-1]

u_guess = jnp.zeros((N - 1, 2))

solution = iterative_linear_quadratic_regulator(
    BicycleDynamics(dt=dt),
    TotalCost(RunningCost(Q= Q, R= R, s_ref= s_ref), TerminalCost(QN= QN, s_ref= s_ref)),
    s0_car,
    u_guess,
)
i = solution["num_iterations"]
print("iLQR Converged after " + str(i) + " iterations.")

all_s = solution["trajectory_iterates"][0][:i]
all_u = solution["trajectory_iterates"][1][:i]

x_car, y_car = all_s[-1][:, 0], all_s[-1][:, 1]
delta_f, a = all_u[-1][:, 0], all_u[-1][:, 1]

s_iLQR = all_s[-1]
u_iLQR = all_u[-1]


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
#plt.show()

# Plot the control trajectories
plt.figure(figsize=(10, 5))
plt.plot(delta_f * 180 / np.pi, label='Steering Angle')
plt.xlabel('Time Step')
plt.ylabel('delta_f (degrees)')
plt.title('Steering Angle Reference Trajectory from iLQR')
plt.legend()
plt.grid(True)
plt.savefig('steering_angle_trajectory.png')
#plt.show()

plt.figure(figsize=(10, 5))
plt.plot(a, label='Acceleration')
plt.xlabel('Time Step')
plt.ylabel('Acceleration (m/s^2)')
plt.title('Acceleration Reference Trajectory from iLQR')
plt.legend()
plt.grid(True)
plt.savefig('acceleration_trajectory.png')
#plt.show()



## MPC Using s_ref as trajectory and u_ref[0] as first control input

def bicycle_model(state, control, lr= params['lr'], lf= params['lf'], dt= params['dt']):
    x, y, psi, v = state # x, y, yaw angle, velocity
    delta_f, a = control # Steering angle, Acceleration
    lr, lf = params['lr'], params['lf']  # Distance from rear wheel to center of mass, Distance from front wheel to center of mass
    L = lr + lf # Wheelbase

    
    beta = jnp.arctan(lr * jnp.tan(delta_f) / L)   # Sideslip angle
    
    dx = v * jnp.cos(psi + beta) # x velocity
    dy = v * jnp.sin(psi + beta) # y velocity
    dpsi = v / lr * jnp.sin(beta)# yaw rate
    dv = a # acceleration
    
    return jnp.array([x + dt * dx, y + dt * dy, psi + dt * dpsi, v + dt * dv])

@partial(jax.jit, static_argnums=(0,))
@partial(jax.vmap, in_axes=(None, 0, 0))
def affinize(f, s, u):
    """Affinize the function `f(s, u)` around `(s, u)`."""
    A, B = jax.jacfwd(lambda s_k, u_k: f(s_k, u_k), argnums=(0,1))(s, u)
    c = f(s, u) - A @ s - B @ u
    return A, B, c

def scp_iteration(f, s0, u0, s_road, s_prev, u_prev, P, Q, R, C):
    """Solve a single SCP sub-problem for the obstacle avoidance problem."""
    n = s_prev.shape[-1]  # state dimension
    m = u_prev.shape[-1]  # control dimension
    N = u_prev.shape[0]  # number of steps

    Af, Bf, cf = affinize(f, s_prev[:-1], u_prev)
    Af, Bf, cf = np.array(Af), np.array(Bf), np.array(cf)

    s_cvx = cvx.Variable((N + 1, n))
    u_cvx = cvx.Variable((N, m))
    du_cvx = cvx.Variable((N, m))

    objective = sum(cvx.quad_form(C @ (s_cvx[k] - s_road[k]), Q) + cvx.quad_form(du_cvx[k], R) for k in range(N))
    constraints = [s_cvx[0] == s0]
    constraints += [u_cvx[0] == u0 + du_cvx[0]]
    constraints += [u_cvx[k] == u_cvx[k-1] + du_cvx[k] for k in range(1, N)]
    constraints += [s_cvx[k+1] == Af[k] @ s_cvx[k] + Bf[k] @ u_cvx[k] + cf[k] for k in range(N)]

    prob = cvx.Problem(cvx.Minimize(objective), constraints)
    prob.solve(solver=cvx.OSQP)
    if prob.status != "optimal":
        raise RuntimeError("SCP solve failed. Problem status: " + prob.status)
    
    s = s_cvx.value
    du = du_cvx.value
    u = u_cvx.value
    J = prob.objective.value
    return s, u, J


def solve_obstacle_avoidance_scp(
    f,
    s0,
    u0,
    s_road,
    N,
    P,
    Q,
    R,
    C,
    eps,
    max_iters,
    s_init=None,
    u_init=None,
    convergence_error=False,
):
    """Solve the obstacle avoidance problem via SCP."""
    n = Q.shape[0]  # state dimension
    m = R.shape[0]  # control dimension

    # Initialize trajectory
    if s_init is None or u_init is None:
        s = np.zeros((N + 1, n))
        u = np.zeros((N, m))
        s[0] = s0
        for k in range(N):
            s[k + 1] = f(s[k], u[k])
    else:
        s = np.copy(s_init)
        u = np.copy(u_init)

    # Do SCP until convergence or maximum number of iterations is reached
    converged = False
    J = np.zeros(max_iters + 1)
    J[0] = np.inf
    for i in range(max_iters):
        s, u, J[i + 1] = scp_iteration(f, s0, u0, s_road, s, u, P, Q, R, C)
        dJ = np.abs(J[i + 1] - J[i])
        if dJ < eps:
            converged = True
            break
    if not converged and convergence_error:
        raise RuntimeError("SCP did not converge!")
    return s, u


f = bicycle_model
C = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 0]])

N_mpc = 5 # MPC horizon
N_scp = 5  # maximum number of SCP iterations


# Define constants
n = 4  # state dimension
m = 2  # control dimension
s_mpc = np.zeros((N, N_mpc + 1, n))
u_mpc = np.zeros((N, N_mpc, m))
s = np.copy(s0_car)
total_time = time()
total_control_cost = 0.0
s_init = s_iLQR[0:N_mpc+1]
u_init = u_iLQR[0:N_mpc]
u = u_init[0]
P = 1e2 * np.eye(n)  # terminal state cost matrix
Q = 1e1 * np.eye(n)  # state cost matrix
R = 1e-2 * np.eye(m)  # control cost matrix
eps = 1e-3  # SCP convergence tolerance

for t in tqdm(range(N)):
    # Solve the MPC problem at time `t`
    s_road = np.zeros((N_mpc + 1, n))
    if (t + N_mpc + 1 > N):
         num_greater = (t + N_mpc + 1) - N
         s_road[:N_mpc + 1 - num_greater] = s_ref[t: t + N_mpc + 1 - num_greater]
         s_road[N_mpc + 1 - num_greater: N_mpc + 1] = s_ref[-1]
    else: s_road = s_ref[t:t+N_mpc+1]

    s_mpc[t], u_mpc[t] = solve_obstacle_avoidance_scp(f, s, u, s_road, N_mpc, P, Q, R, C, eps, N_scp, s_init, u_init)

    u = u_mpc[t, 0, :]
    s = f(s, u)


    # Accumulate the actual control cost
    total_control_cost += u_mpc[t, 0].T @ R @ u_mpc[t, 0]

    # Use this solution to warm-start the next iteration
    u_init = np.concatenate([u_mpc[t, 1:], u_mpc[t, -1:]])
    s_init = np.concatenate(
        [s_mpc[t, 1:], f(s_mpc[t, -1], u_mpc[t, -1]).reshape([1, -1])]
    )

total_time = time() - total_time
print("Total elapsed time:", total_time, "seconds")
print("Total control cost:", total_control_cost)

#fig, ax = plt.subplots(1, 2, dpi=150, figsize=(15, 5))
#fig.suptitle("$N = {}$, ".format(N_mpc) + r"$N_\mathrm{SCP} = " + "{}$".format(N_scp))

plt.figure(figsize=(10, 5))
for t in range(N):
    plt.plot(s_mpc[t, :, 0], s_mpc[t, :, 1], "--*", color="k")
plt.plot(s_mpc[:, 0, 0], s_mpc[:, 0, 1], "-o", label='Generated Trajectory')
plt.plot(s_ref[:,0], s_ref[:,1], 'r--', label='Reference Road')
plt.xlabel(r"$x(t)$")
plt.ylabel(r"$y(t)$")
plt.axis("equal")
plt.grid(True)
plt.legend()
plt.savefig("mpc_trajectory.png")

plt.figure(figsize=(10, 5))
plt.plot(u_mpc[:, 0, 0] * 180 / np.pi, "-o", label=r"$delta_f(t)$")
plt.xlabel(r"$t$")
plt.ylabel(r"Steering Angle (deg)")
plt.grid(True)
plt.savefig("mpc_steering_angle_trajectory.png")


plt.figure(figsize=(10, 5))
plt.plot(u_mpc[:, 0, 1], "-o", label=r"$a(t)$")
plt.xlabel(r"$t$")
plt.ylabel(r"$Acceleration (m/s^2)$")
plt.grid(True)
plt.savefig("mpc_acceleration_trajectory.png")


