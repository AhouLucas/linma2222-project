import numpy as np
import matplotlib.pyplot as plt


W_A = 0.1
W_U = 0.2
BETA_U = -0.048
GAMMA_U = 0.06
THETA = 0.5

def generate_trajectories(policy, x0=(0, 0, 0), T=1000):
    """Generate the state variables x_t and actions u_t for t=0,...,T
       using the given policy.

    Args:
        policy: a function that takes in the current state x_t and returns an action u_t
        x0 (ndarray): initial state. Defaults to 0.
        T (int): number of time steps. Defaults to 1000.
    """
    x = np.zeros((T+1, 3))  # State variables: q_t, za_t, zu_t
    u = np.zeros((T))       # Actions
    x[0] = x0

    for t in range(T):
        u[t] = policy(x[t])
        x[t+1, 0] = x[t, 0] + u[t] # q_t
        x[t+1, 1] = (1 - W_A) * x[t, 1]
        x[t+1, 2] = (1 - W_U) * x[t, 2] + W_U * BETA_U * u[t]

    return x, u

def g(q, za, zu, u):
    """Compute the gross stage reward at each time given state x and action u."""

    # Formula found by replacing p_t+1 and p_t by their expressions in the given model (see Question 2.3)
    return 1000 * q * (za + zu + (GAMMA_U * u)) + THETA * u * (za + zu)

def c(g):
    return np.maximum(g - (np.pow(g, 2) / 2), 1 - np.exp(-g))

def reward(x, u):
    """Compute all the rewards over the trajectory given states x and actions u from time 0 to t."""
    """returns an array of shape (t,)"""
    g_t = g(x[:, 0], x[:, 1], x[:, 2], u[:])
    c_t = c(g_t)
    return c_t


def generate_dataset(sigma_exp=0.1, x0=np.array([1, 1, 1])):
    Kcl = np.array([-0.5, 0.5, 0.5])
    policy = lambda x: np.random.normal(Kcl@x, sigma_exp**2)

    return generate_trajectories(policy, x0)


def psi(x, u):
    q, za, zu = x[0], x[1], x[2]
    return np.array([
        1, q, za, zu, u,
        np.power(q, 2), np.power(za, 2), np.power(zu, 2), np.power(u, 2),
        q*za, q*zu, q*u, za*zu, za*u, zu*u
    ])

def lstd(data_x, data_u, W, policy, psi, c, d):
    N = data_x.shape[0] - 1
    gamma = np.zeros(N)
    Gamma = np.zeros((d, N))

    for k in range(N):
        x_k, u_k = data_x[k], data_u[k]
        x_k1 = data_x[k+1]
        gamma[k] = c(g(x_k[0], x_k[1], x_k[2], u_k))
        Gamma[:, k] = psi(x_k, u_k) - psi(x_k1, policy(x_k1))

    R_N = np.zeros((d, d))
    psi_bar = np.zeros(d)

    for k in range(N):
        R_N += Gamma[:, k] @ Gamma[:, k].T
        psi_bar += Gamma[:, k] * gamma[k]

    R_N /= N
    psi_bar /= N

    return np.linalg.solve(W/N + R_N, psi_bar)


def Q_hat(x, u, policy):
    """Compute an approximation of Q by simulating a long trajectory starting at (x, u)

    Args:
        x (ndarray): initial condition
        u (ndarray): first control input
    """
    total_reward = 0

    # Apply first input to the initial condition
    first_input_policy = lambda x: u
    next_state = generate_trajectories(first_input_policy, x, T=1)[0][-1]

    total_reward += c(g(next_state[0], next_state[1], next_state[2], u))

    states, inputs = generate_trajectories(policy, next_state, T=1000)

    total_reward += np.sum(reward(states[:-1], inputs))
    
    return total_reward




d=15
Kcl = np.array([-0.5, 0.5, 0.5])
policy = lambda x: Kcl@x


def plot_lstd_vs_approx(x, u, sigma_exp):
    data_x, data_u = generate_dataset(sigma_exp)
    theta = lstd(data_x, data_u, np.eye(d), policy, psi, c, d)

    Q_lstd = []
    Q_empirical = []

    for i in range(data_x.shape[0]-1):
        x_i = data_x[i]
        u_i = data_u[i]

        Q_lstd.append(theta @ psi(x_i, u_i))
        Q_empirical.append(Q_hat(x_i, u_i, policy))

    plt.scatter(Q_empirical, Q_lstd)
    plt.xlabel(r'$\hat{Q}$')
    plt.ylabel(r"$Q^{\theta}$")
    plt.title(r"$Q^{\theta}$ vs. $\hat{Q}$")
    plt.grid(alpha=0.7)
    plt.savefig('part3/figures/lstd_vs_approx.png', dpi=600)
    plt.show()


def plot_theta_vs_sigma_exp(x0=np.array([1, 1, 1])):
    """Plot the norm of theta vs the exploration noise to generate the dataset
    """

    N = 50
    sigma_exps = np.linspace(0.01, 1, N)
    theta_norms = np.zeros(N)

    for i in range(N):
        sigma_exp = sigma_exps[i]
        data_x, data_u = generate_dataset(sigma_exp, x0)
        theta = lstd(data_x, data_u, np.eye(d), policy, psi, c, d)
        theta_norms[i] = np.linalg.norm(theta)

    plt.plot(sigma_exps, theta_norms)
    plt.xlabel(r'$\sigma_{exp}$')
    plt.ylabel(r'$\|\theta\|$')
    plt.title(r'Norm of $\theta$ vs. Exploration Noise $\sigma_{exp}$')
    plt.yscale("log")
    plt.grid(alpha=0.7)
    plt.savefig('part3/figures/theta_vs_sigma_exp.png', dpi=600)
    plt.show()

def plot_theta_vs_x0():
    """Plot the norm of theta vs the norm of the initial condition to generate the dataset
    """

    sigma_exp = .1
    N = 500
    init_condition_norms = np.zeros(N)
    theta_norms = np.zeros(N)

    for i in range(N):
        x0 = np.random.uniform(np.array([0, -10, -10]), np.array([1, 10, 10]))
        data_x, data_u = generate_dataset(sigma_exp, x0)
        theta = lstd(data_x, data_u, np.eye(d), policy, psi, c, d)

        init_condition_norms[i] = np.linalg.norm(x0)
        theta_norms[i] = np.linalg.norm(theta)

    plt.scatter(init_condition_norms, theta_norms, s=5)
    plt.xlabel(r'$\|x_0\|$')
    plt.ylabel(r'$\|\theta\|$')
    plt.title(r'Norm of $\theta$ vs. Norm of $x_0$')
    plt.yscale("log")
    plt.grid(alpha=0.7)
    plt.savefig('part3/figures/theta_vs_x0.png', dpi=600)
    plt.show()





plot_lstd_vs_approx(np.array([1, 1, 1]), 1, 0.1)
plot_theta_vs_x0()
plot_theta_vs_sigma_exp()