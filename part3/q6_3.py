import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import solve_discrete_are
from scipy.optimize import minimize
import itertools
from model import generate_trajectories

N = 3000   # Increase N for better stability with high-degree polynomials
N1 = 8  # no. of iterations for policy improvement

W_A = 0.1
W_U = 0.2
BETA_U = -0.048
GAMMA_U = 0.06
THETA = 0.5
SIGMA_P = 0.02

A = np.array([[1, 0, 0],
              [0, 1 - W_A, 0], 
              [0, 0, 1 - W_U]])
B = np.array([[1],
              [0],
              [W_U * BETA_U]])

K_cl = -np.array([-0.5, 0.5, 0.5])
K_lqr = np.array([1.11182191, -2.64863317, -2.52792524])

def cost(x, u):
    q, za, zu = x[0], x[1], x[2]
    g = 1000 * (q * (za + zu + GAMMA_U * u) + THETA * u * (za + zu))
    return -max(g - ((g**2)/2), 1 - np.exp(-g))

def psi(x, u):
    # Combine state and input into one vector
    aug = np.append(x, u)
    dim = len(aug)
    degree = 4 # Increase to 4 to capture the quartic nature of the cost
    
    features = [] # Bias term removed for undiscounted task
    
    # Generate all combinations of indices for monomials up to 'degree'
    # This is equivalent to PolynomialFeatures in sklearn
    for d in range(1, degree + 1):
        for indices in itertools.combinations_with_replacement(range(dim), d):
            term = 1.0
            for idx in indices:
                term *= aug[idx]
            features.append(term)
            
    return np.array(features)


d = psi(np.zeros(3), 0).shape[0]

def lspd_implementation(K):
    x0 = np.array([1, 1, 1])
    var = .1
    policy = lambda x: np.random.normal(-K @ x, var**2)

    x = np.zeros((N+1, 3))
    u = np.zeros(N)
    x[0] = x0
    for t in range(N):
        u[t] = policy(x[t])
        x[t+1] = A @ x[t] + B.flatten() * u[t]

    psi0 = np.zeros((d, N))
    psi1 = np.zeros((d, N))
    costs = np.zeros(N)

    for k in range(N):
        psi0[:, k] = psi(x[k], u[k])
        psi1[:, k] = psi(x[k+1], -K @ x[k+1])
        costs[k] = cost(x[k], u[k])

    Upsilon = psi0 - psi1

    phi_bar = np.zeros(d)
    R_N = np.zeros((d, d))
    for i in range(N):
        # LSTD: Project onto the basis functions psi0 (Instrumental Variable)
        phi_bar += costs[i] * psi0[:, i]
        R_N += np.outer(psi0[:, i], Upsilon[:, i])

    phi_bar /= N
    R_N /= N

    # Use solve instead of inv for stability, and add small regularization
    vecH = np.linalg.solve(R_N + 1e-5 * np.eye(d), phi_bar)

    return vecH
    

def lspi():
    K = np.zeros((3, N1+1))
    K[:, 0] = K_cl  # Initial policy
    for it in range(1, N1+1):
        print(f"Iteration {it+1}/{N1}, current policy K: {K[:, it-1]}")

        theta = lspd_implementation(K[:, it-1])

        # Policy update: Optimize Q function obtained from LSTD
        # Sample states to perform policy improvement
        n_samples = 100 # Increased samples for better regression
        X_sample = np.random.uniform(-1, 1, (n_samples, 3))
        U_sample = np.zeros(n_samples)

        for i in range(n_samples):
            x_s = X_sample[i]
            # Find u that minimizes Q(x_s, u)
            # We use the current policy as initial guess
            u0 = -K[:, it-1] @ x_s
            
            # FIX: We want to MAXIMIZE the Q-value (since cost is negative reward)
            # Therefore, we minimize the NEGATIVE Q-value
            # Added bounds to prevent divergence of the polynomial approximation
            res = minimize(lambda u: -theta @ psi(x_s, u), u0, method='L-BFGS-B', bounds=[(-5, 5)])
            U_sample[i] = res.x[0]

        # Fit linear policy u = -K x
        # We solve for K in: -U = X @ K
        XtX = X_sample.T @ X_sample
        XtU = X_sample.T @ (-U_sample)
        
        K[:, it] = np.linalg.solve(XtX + 1e-6 * np.eye(3), XtU)

    return K

def Q_hat(x, u):


    """Compute an approximation of Q by simulating a long trajectory starting at (x, u)

    Args:
        x (ndarray): initial condition
        u (ndarray): first control input
    """
    from model import generate_trajectories

    total_reward = 0

    # Apply first input to the initial condition
    next_state = A @ x + B.flatten() * u
    total_reward += cost(x, u)

    policy = lambda x: -K_cl @ x
    states, inputs, _ = generate_trajectories(policy, next_state, T=1000)
    
    for t in range(len(inputs)):
        x_t = states[t]
        u_t = inputs[t]
        total_reward += cost(x_t, u_t)

    return total_reward

def plot_lspd_vs_Q_hat():
    # Generate test data
    x_0 = np.array([0.8, 1, 0.5])
    var = 0.1
    N_test = 100
    policy = lambda x: np.random.normal(-K_cl @ x, var**2)

    x_test = np.zeros((N_test+1, 3))
    u_test = np.zeros(N_test)
    x_test[0] = x_0
    for t in range(N_test):
        u_test[t] = policy(x_test[t])
        x_test[t+1] = A @ x_test[t] + B.flatten() * u_test[t]

    theta = lspd_implementation(K_cl)
    Q_lspd_eval = np.zeros(N_test)
    Q_hat_eval = np.zeros(N_test)

    for k in range(N_test):
        Q_lspd_eval[k] = theta @ psi(x_test[k], u_test[k])
        Q_hat_eval[k] = Q_hat(x_test[k], u_test[k])

    # Filter evaluation where values are too large
    # mask = (np.abs(Q_hat_eval) < 30) #& (np.abs(Q_lspd_eval) < 100)
    # Q_hat_eval = Q_hat_eval[mask]
    # Q_lspd_eval = Q_lspd_eval[mask]

    plt.figure(figsize=(8, 6))
    plt.scatter(Q_lspd_eval, Q_hat_eval, alpha=0.5)
    # plt.xscale("symlog")
    # plt.yscale("symlog")
    plt.xlabel(r"$Q^{\theta}$")
    plt.ylabel(r"$\hat{Q}$")
    plt.axis('equal')
    plt.title("Comparison of Q estimates")
    plt.plot([min(Q_lspd_eval), max(Q_lspd_eval)], [min(Q_lspd_eval), max(Q_lspd_eval)], 'r--', label="y=x")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.savefig("part3/figures/q63_lspd_vs_qhat.png", dpi=300)
    plt.show()

def plot_rewards():
    """Plot reward for policy Kcl, Klqr and policy learned from LSPI
    """

    policies = {
        "K_cl": K_cl,
        "K_lqr": K_lqr,
        "K_lspi": lspi()[:, -1]
    }

    print("Policies to evaluate:", policies)

    rewards = {}
    x0 = np.array([1, 1, 1])
    for name, K in policies.items():
        total_reward = 0
        policy = lambda x: -K @ x
        states, inputs, _ = generate_trajectories(policy, x0, T=1000)
        rewards[name] = np.zeros(len(inputs))
        for t in range(len(inputs)):
            x_t = states[t]
            u_t = inputs[t]
            # FIX: Accumulate the actual reward (negative value). 
            # A stable policy will converge to a constant negative value.
            total_reward += cost(x_t, u_t)
            rewards[name][t] = total_reward
        print(f"Total reward for policy {name}: {total_reward}")

    plt.figure(figsize=(8, 6))
    for name, reward in rewards.items():
        plt.plot(np.linspace(0, 1000, len(reward)), -reward, label=name)
    plt.legend()
    plt.xlabel("Time step")
    plt.ylabel("Cumulative Reward")
    plt.yscale("symlog")
    plt.title("Comparison of Cumulative Rewards")
    plt.grid(True, alpha=0.3)
    plt.savefig("part3/figures/q63_policy_rewards.png", dpi=300)
    plt.show()

if __name__ == "__main__":
    K_lspi = lspi()
    print("Learned policy K_lspi:", K_lspi[:, -1])
    plot_lspd_vs_Q_hat()
    plot_rewards()