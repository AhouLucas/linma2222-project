from email import policy
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import solve_discrete_are

n = 3
m = 1
d = (m + n) * (m + n + 1) // 2
N = 10   # no. of data points collected for each iteration
N1 = 8  # no. of iterations for policy improvement

W_A = 0.1
W_U = 0.2
BETA_U = -0.048
GAMMA_U = 0.06
THETA = 0.5
SIGMA_P = 0.02

Q = np.array([[-(1000 * SIGMA_P)**2, 1000, 1000], 
                     [1000, 0, 0], 
                     [1000, 0, 0]]) 

R = - (1000 * THETA * SIGMA_P)**2

S = np.array([[1000*GAMMA_U - THETA * ((1000 * SIGMA_P)**2)], 
                     [1000 * THETA], 
                     [1000 * THETA]]) 

A = np.array([[1, 0, 0],
              [0, 1 - W_A, 0], 
              [0, 0, 1 - W_U]])
B = np.array([[1],
              [0],
              [W_U * BETA_U]])

Q = -Q
S = -S
R = -R

#### Compute LQR gain

def compute_lqr_gain(model):
    F, G, Q, R, S = model
    # M = solve_ricatti_infinite_horizon(F, G, Q, R, S)
    # K_opt = optimal_gain(F, G, Q, R, S, M)

    M = solve_discrete_are(F, G, Q, R, e=np.eye(3), s=S)
    K_lqr = -np.linalg.solve(R + G.T @ M @ G, (G.T @ M @ F + (1*S).T))

    residual = F.T @ M @ F - (F.T @ M @ G + S) @ np.linalg.inv(R + G.T @ M @ G) @ (G.T @ M @ F + S.T) + Q - M
    print(f"K_lqr : {K_lqr}  residual norm: {np.linalg.norm(residual, ord='fro')}")

    # Check closed-loop stability
    F_cl = F + G @ K_lqr
    eigenvalues = np.linalg.eigvals(F_cl)
    print("Stable:", all(np.abs(eigenvalues) < 1))

    # K_opt = - np.array([[-0.8864, 2.1253, 1.2096]])

    return K_lqr, M

model = (A, B, Q, R, S)
K_lqr, M = compute_lqr_gain(model)
K_lqr = -K_lqr
K_cl = -np.array([-0.5, 0.5, 0.5])

def psi(x, u):
    x1, x2, x3 = x
    psi = np.array([
        x1**2,
        2*x1*x2,
        2*x1*x3,
        2*x1*u,
        x2**2,
        2*x2*x3,
        2*x2*u,
        x3**2,
        2*x3*u,
        u**2
    ])
    return psi

def lspd_lqr_implementation(K):
    x0 = np.array([1, 1, 1])
    var = 0.1
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
        costs[k] = (x[k].T @ Q @ x[k] + u[k].T * R * u[k] + 2 * x[k].T @ S * u[k])[0]

    Upsilon = psi0 - psi1

    phi_bar = np.zeros(d)
    R_N = np.zeros((d, d))
    for i in range(N):
        phi_bar += costs[i] * Upsilon[:, i]
        R_N += np.outer(Upsilon[:, i], Upsilon[:, i])

    phi_bar /= N
    R_N /= N

    vecH = np.linalg.inv(R_N) @ phi_bar

    return vecH
    

def lspi():
    K = np.zeros((3, N1+1))
    K[:, 0] = K_cl  # Initial policy
    for it in range(1, N1+1):
        print(f"Iteration {it+1}/{N1}, current policy K: {K[:, it-1]}")

        vecH = lspd_lqr_implementation(K[:, it-1])

        # Policy update
        K[:, it] = np.array([vecH[3], vecH[6], vecH[8]]) / vecH[9]  # H_ux / H_uu

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
    total_reward += (x.T @ Q @ x + u.T * R * u + 2 * x.T @ S * u)[0]

    policy = lambda x: -K_cl @ x
    states, inputs, _ = generate_trajectories(policy, next_state, T=1000)
    
    for t in range(len(inputs)):
        x_t = states[t]
        u_t = inputs[t]
        total_reward += (x_t.T @ Q @ x_t + u_t.T * R * u_t + 2 * x_t.T @ S * u_t)[0]

    return total_reward

def plot_lspd_vs_Q_hat():
    # Generate test data
    x_0 = np.array([0.8, 1, 0.5])
    var = 0.01
    N_test = 100
    policy = lambda x: np.random.normal(-K_cl @ x, var**2)

    x_test = np.zeros((N_test+1, 3))
    u_test = np.zeros(N_test)
    x_test[0] = x_0
    for t in range(N_test):
        u_test[t] = policy(x_test[t])
        x_test[t+1] = A @ x_test[t] + B.flatten() * u_test[t]

    theta = lspd_lqr_implementation(K_cl)
    Q_lspd_eval = np.zeros(N_test)
    Q_hat_eval = np.zeros(N_test)

    for k in range(N_test):
        Q_lspd_eval[k] = theta @ psi(x_test[k], u_test[k])
        Q_hat_eval[k] = Q_hat(x_test[k], u_test[k])

    ## Filter evaluation where values are too large
    mask = (np.abs(Q_hat_eval) < 1e3) & (np.abs(Q_lspd_eval) < 1e3)
    Q_hat_eval = Q_hat_eval[mask]
    Q_lspd_eval = Q_lspd_eval[mask]

    plt.figure(figsize=(8, 6))
    plt.scatter(Q_lspd_eval, Q_hat_eval, alpha=0.5)
    # plt.xscale("symlog")
    # plt.yscale("symlog")
    plt.xlabel(r"$Q^{\theta}$")
    plt.ylabel(r"$\hat{Q}$")
    plt.title("Comparison of Q estimates")
    plt.plot([min(Q_lspd_eval), max(Q_lspd_eval)], [min(Q_lspd_eval), max(Q_lspd_eval)], 'r--', label="y=x")
    plt.grid(alpha=0.3)
    plt.legend()
    # plt.savefig("part3/figures/q64_lspd_vs_qhat.png", dpi=300)
    plt.show()

def plot_policy_convergence():
    plt.figure(figsize=(8, 6))
    K = lspi()

    # plot difference in norm between Klqr and K at each iteration
    norms = [np.linalg.norm(K[:, i] - K_lqr.flatten()) for i in range(K.shape[1])]
    plt.plot(range(K.shape[1]), norms, marker='o')
    plt.yscale("log")
    plt.xlabel("Iteration")
    plt.ylabel(r"$\|K_{LQR} - K_i\|$")
    plt.title("Convergence of Learned Policy to LQR Policy")
    plt.grid(alpha=0.3)
    plt.savefig("part3/figures/q64_policy_convergence.png", dpi=300)
    plt.show()


print("LQR Gain K_lqr:", K_lqr)
# if __name__ == "__main__":
#     K_lspi = lspi()
#     print("Learned policy K_lspi:", K_lspi[:, -1])
#     # plot_lspd_vs_Q_hat()
#     plot_policy_convergence()