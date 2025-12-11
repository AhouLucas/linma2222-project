

from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

THETA = 0.5
W_A = 0.1
W_U = 0.2
SIGMA_A = 0.018
SIGMA_P = 0.02
BETA_U = -0.048
GAMMA_U = 0.06

rng = np.random.default_rng(42)  # For reproducibility


K_cl = np.array([-0.5, 0.5, 0.5])
pol_cl = lambda x: float(K_cl @ x)


def model_step(x, u, xi_a=None):
    """Compute the next state given current state x and action u."""
    if xi_a is None: xi_a = rng.normal(0, 1)
    q_next = x[0] + u
    za_next = (1 - W_A) * x[1] + W_A * SIGMA_A * xi_a
    zu_next = (1 - W_U) * x[2] + W_U * BETA_U * u
    return np.array([q_next, za_next, zu_next])


def generate_trajectories(policy, x0=(0, 0, 0), T=1000, N=1,
                          model_step_fn=model_step,xi_a=None,xi_p=None,
                          show_progress=False):
    """Generate the state variables x_t and actions u_t for t=0,...,T
       using the given policy and one–step model.
    """

    x = np.zeros((T+1, 3, N))  # State variables: q_t, za_t, zu_t
    u = np.zeros((T, N))       # Actions

    # initial condition(s)
    if isinstance(x0, str) and x0 == "random":
        x[0] = rng.normal(0, 1, size=(3, N)) * np.array([[1], [SIGMA_A], [BETA_U]])
    elif callable(x0):
        for i in range(N):
            x[0, :, i] = np.asarray(x0(), dtype=float).reshape(3)
    else:
        x[0] = np.array(x0, dtype=float).reshape(3, 1)

    # noises
    if xi_a is None: xi_a = rng.normal(0, 1, size=(T, N))
    if xi_p is None: xi_p = rng.normal(0, 1, size=(T, N)) 

    time_iter = range(T)
    if show_progress:
        time_iter = tqdm(time_iter, desc="Generating trajectories")

    for t in time_iter:
        for i in range(N):
            x_t_i = x[t, :, i]
            u_t_i = policy(x_t_i)          # scalar action
            u[t, i] = u_t_i

            # one-step transition with per-trajectory noise
            x_next_i = model_step_fn(x_t_i, u_t_i, xi_a=xi_a[t, i])
            x[t+1, :, i] = np.asarray(x_next_i, dtype=float).reshape(3)

    return x, u, xi_p



def g(q, za, zu , u, xi_p=None):
    """Compute the gross stage reward at each time given state x and action u."""
    # q, za, zu = x[:, 0], x[:, 1], x[:, 2]
    if xi_p is None:
        if q.ndim > 0:
            xi_p = rng.normal(0, 1, size=q.shape[0])
        else:
            xi_p = rng.normal(0, 1, size=1)

    # Formula found by replacing p_t+1 and p_t by their expressions in the given model (see Question 2.3)
    return 1000 * (q * (za + zu + (GAMMA_U * u) + (SIGMA_P * xi_p)) + THETA * u * (za + zu + (SIGMA_P * xi_p)))


def c_quad(g):
    return g - 0.5 * (g ** 2)

def c(g):   
    """Compute the net stage reward given gross stage reward g_t for each time t"""
    return np.maximum(c_quad(g), 1 - np.exp(-g))

def reward(x, u, xi_p=None, t=None):
    """Compute all the rewards over the trajectory given states x and actions u from time 0 to t."""
    """returns an array of shape (t,N)"""
    g_t = g(x[:t, 0], x[:t, 1], x[:t, 2], u[:t], xi_p=xi_p[:t])
    c_t = c(g_t)
    return c_t

def stage_reward(x_t, u_t, xi_p_t=None):
    """Compute the stage reward at time t given state x_t and action u_t."""
    g_t = g(x_t[0], x_t[1], x_t[2], u_t, xi_p=xi_p_t)
    c_t = c(g_t)
    return c_t

def average_reward(x, u, xi_p, t):
    return np.nanmean(reward(x, u, xi_p, t))


def get_avg_reward(policy, T=1000, N=10):
    x, u, xi_p = generate_trajectories(policy, T=T, N=N)
    avg_r = average_reward(x, u, xi_p, T)
    return avg_r