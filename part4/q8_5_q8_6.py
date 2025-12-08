import numpy as np
import matplotlib.pyplot as plt
# from scipy.linalg import solve_discrete_are
# from scipy.optimize import minimize
# import itertools
from model import generate_trajectories, stage_reward, model_step, K_cl, pol_cl, reward
from lqr import model, stage_reward_approx, model_step_approx
# F, G, H, E, D, Q, R, S = model

def cond_expectation(x, u, policy, psi, model_step, stage_reward, n_mc=20):
    """
    Approximate:
        E[r_t | x_t=x, u_t=u]          -> scalar
        E[psi(x_{t+1}, pi(x_{t+1}))]   -> (d,)
    via Monte Carlo using the model dynamics.
    """
    # ensure d is 1D length of psi
    u = float(np.asarray(u).reshape(-1)[0])
    x = np.asarray(x, dtype=float).reshape(-1)

    d = psi(x, u).shape[0]

    psi_next_mean = np.zeros(d)
    r_sum = 0.0

    for _ in range(n_mc):
        # one-step transition
        x_next = model_step(x, u)

        # stage_reward may return a vector → reduce to scalar
        r = stage_reward(x, u, x_next)
        r = float(np.mean(r))   # or float(np.asarray(r)) if it's already scalar-like

        r_sum += r

        # next action
        u_next = policy(x_next)

        # features at next state-action
        psi_next_mean += psi(x_next, u_next)

    psi_next_mean /= n_mc
    r_bar = r_sum / n_mc       # this is now a scalar float

    return r_bar, psi_next_mean


def generate_dataset(sigma_exp=0.1, x0=np.array([0.1, 0.01, 0.01]), T=3000, N=1, model_step_fn=model_step):
    # Kcl = np.array([-0.5, 0.5, 0.5])
    policy = lambda x: np.random.normal(K_cl@x, sigma_exp**2)

    x, u, xi_p = generate_trajectories(policy, x0, T=T, N=N, model_step_fn=model_step_fn)
    # shape (T+1, 3, N), (T, N)
    print(f"Generated dataset with shape x: {x.shape}, u: {u.shape}")

    # transpose -> (T, N, 3), then reshape to (T*N, 3)
    x = x[:-1].transpose(0, 2, 1).reshape(-1, x.shape[1])

    # u has shape (T, N). Flatten to (T*N,)
    u = u.reshape(-1)


    print(f"Generated dataset with shape x: {x.shape}, u: {u.shape}")

    return x, u, xi_p


def psi(x, u):
    # Make sure x is a flat array of scalars
    x = np.asarray(x).reshape(-1)
    q  = float(x[0])
    za = float(x[1])
    zu = float(x[2])

    u = float(np.asarray(u).reshape(-1)[0])

    return np.array([
        # 1.0, # cannot have a const for poisson equation
        q, za, zu, u,
        q**2, za**2, zu**2, u**2,
        q*za, q*zu, q*u, za*zu, za*u, zu*u
    ], dtype=float)



def lspe(data_x, data_u, W, policy, psi, model_step, stage_reward, d, n_mc=20):
    """
    LSPE: Least-Squares Poisson Error

    data_x : (N_x, nx) array of states (typically length T+1)
    data_u : (N_u, nu) array of actions (typically length T)
    W      : (d+1, d+1) regularization matrix (e.g. λ * I)
    policy : policy π to evaluate (e.g. pol_cl)
    psi    : basis for Q(x,u), returns R^d
    model_step, stage_reward : true model + reward
    n_mc   : Monte Carlo samples for conditional expectations
    """
    # Number of usable samples is limited by the shorter of the two
    N = min(data_x.shape[0], data_u.shape[0])

    # Infer feature dimension from first sample
    d = psi(data_x[0], data_u[0]).shape[0]
    d_theta = d + 1  # θ = [θ_Q; η]

    A = np.zeros((d_theta, d_theta))
    b = np.zeros(d_theta)

    for k in range(N):
        x_k = data_x[k]
        u_k = data_u[k]

        # 1) Conditional expectations at (x_k, u_k)
        r_bar_k, psi_next_mean_k = cond_expectation(
            x_k, u_k, policy, psi, model_step, stage_reward, n_mc=n_mc
        )

        # 2) Current features
        psi_now_k = psi(x_k, u_k)

        # 3) Poisson feature vector φ_k = [ E[psi_next] - psi_now ; -1 ]
        phi_k = np.concatenate([psi_next_mean_k - psi_now_k, np.array([-1.0])])

        # 4) Accumulate A ≈ E[φ φ^T], b ≈ -E[φ r_bar]
        A += np.outer(phi_k, phi_k)
        b -= phi_k * r_bar_k  # r_bar_k is scalar

    A /= N
    b /= N

    # W should be (d+1, d+1)
    A_reg = A + W / N

    theta = np.linalg.solve(A_reg, b)

    theta_Q = theta[:-1]  # Q parameters
    eta_hat = theta[-1]   # average reward

    return theta_Q, eta_hat



# def Q_hat(x, u, policy):
#     """Compute an approximation of Q by simulating a long trajectory starting at (x, u)

#     Args:
#         x (ndarray): initial condition
#         u (ndarray): first control input
#     """
#     total_reward = 0

#     # Apply first input to the initial condition
#     first_input_policy = lambda x: u
#     next_state = generate_trajectories(first_input_policy, x, T=1)[0][-1]

#     # total_reward += c(g(next_state[0], next_state[1], next_state[2], u))
#     total_reward += stage_reward(x, u, next_state)

#     states, inputs = generate_trajectories(policy, next_state, T=1000)

#     # total_reward += np.sum(reward(states[:-1], inputs))
#     total_reward += np.sum(stage_reward(states[:-1], inputs, states[1:]))
    
#     return total_reward

def Q_hat_mc(x, u, policy, model_step, stage_reward,
             eta,                # <-- NEW: average reward (e.g. eta_hat)
             T=1000, n_traj=20):
    """
    Monte-Carlo estimate of the *Poisson* Q(x,u):

        Q(x,u) ≈ E[ sum_{t=0}^{T-1} (r_t - eta) ]

    where:
      - first action is fixed to u,
      - then follow 'policy'.

    eta should approximate the average reward of policy.
    """
    x = np.asarray(x, dtype=float).copy()
    u = float(np.asarray(u).reshape(-1)[0])

    q_vals = []

    for _ in range(n_traj):
        x_curr = x.copy()

        # --- first step: fixed input u ---
        x_next = model_step(x_curr, u)
        r = stage_reward(x_curr, u, x_next)
        r = float(np.mean(np.asarray(r)))
        total = r - eta     # <-- subtract eta here
        x_curr = x_next

        # --- then follow the policy ---
        for _t in range(T - 1):
            u_t = float(policy(x_curr))
            x_next = model_step(x_curr, u_t)
            r = stage_reward(x_curr, u_t, x_next)
            r = float(np.mean(np.asarray(r)))
            total += r - eta   # <-- subtract eta each step
            x_curr = x_next

        q_vals.append(total)

    return float(np.mean(q_vals))



def plot_Q_lspe_vs_Qhat(theta_Q, eta_hat,  # <-- pass eta_hat
                        data_x, data_u,
                        policy,
                        model_step, stage_reward, psi,
                        n_points=100,
                        T_mc=1000, n_traj_mc=20,
                        seed=0):

    rng = np.random.default_rng(seed)

    N = min(data_x.shape[0], data_u.shape[0])
    idxs = rng.choice(N, size=min(n_points, N), replace=False)

    Q_lspe_vals = []
    Q_hat_vals  = []

    for k in idxs:
        x_k = data_x[k]
        u_k = data_u[k]

        # LSPE estimate
        q_lspe = float(theta_Q @ psi(x_k, u_k))

        # Monte Carlo Poisson Q estimate
        q_hat = Q_hat_mc(x_k, u_k, policy,
                         model_step, stage_reward,
                         eta=eta_hat,         # <-- use eta_hat here
                         T=T_mc, n_traj=n_traj_mc)

        Q_lspe_vals.append(q_lspe)
        Q_hat_vals.append(q_hat)

    Q_lspe_vals = np.array(Q_lspe_vals)
    Q_hat_vals  = np.array(Q_hat_vals)


    plt.figure(figsize=(6, 6))
    ### Polyfit
    coeffs = np.polyfit(Q_lspe_vals, Q_hat_vals, deg=1)
    poly_fit = np.poly1d(coeffs)
    x_fit = np.linspace(Q_lspe_vals.min(), Q_lspe_vals.max(), 100)
    y_fit = poly_fit(x_fit)
    plt.plot(x_fit, y_fit, color='red', linestyle='-', label='Polyfit')
    print(f"Polyfit : y = {coeffs[0]:.4f} x + {coeffs[1]:.4f}")

    plt.scatter(Q_lspe_vals, Q_hat_vals, s=15, alpha=0.7)
    lo = min(Q_lspe_vals.min(), Q_hat_vals.min())
    hi = max(Q_lspe_vals.max(), Q_hat_vals.max())
    plt.plot([lo, hi], [lo, hi], linestyle="--")

    plt.xlabel(r"$Q_{\mathrm{LSPE}}(x,u)$")
    plt.ylabel(r"$\hat Q_{\mathrm{MC}}(x,u)$ (Poisson)")
    plt.title("LSPE Poisson Q vs Monte-Carlo Poisson Q")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('part4/figures/Q_lspe_vs_Qhat.pdf', format='pdf')
    plt.show()

if __name__ == "__main__":
    d = 14  # length of psi vector
    W = 1e-4 * np.eye(d + 1)
    data_x, data_u, xi_p = generate_dataset(T=50, N=100)

    # show the dataset repartition
    # p(data_x, data_u)

    theta_Q, eta_hat = lspe(
        data_x, data_u,
        W=W,
        policy=pol_cl,
        psi=psi,
        model_step=model_step,
        stage_reward=stage_reward,
        d=d,
        n_mc=200,
    )

    print("Learned theta_Q:", theta_Q)
    print("Learned average reward eta_hat:", eta_hat)


    plot_Q_lspe_vs_Qhat(
        theta_Q=theta_Q,
        eta_hat=eta_hat,
        data_x=data_x,
        data_u=data_u,
        policy=pol_cl,
        model_step=model_step,
        stage_reward=stage_reward,
        psi=psi,
        n_points=100,       # how many (x,u) pairs to plot
        T_mc=100,          # length of MC rollouts
        n_traj_mc=200,       # MC trajectories per (x,u)
        seed=0,
    )




    ####### on the approximate model
    data_x_approx, data_u_approx, xi_p_approx = generate_dataset(T=50, N=100, model_step_fn=model_step_approx)

    theta_Q_approx, eta_hat_approx = lspe(
        data_x_approx, data_u_approx,
        W=W,
        policy=pol_cl,
        psi=psi,
        model_step=model_step_approx,
        stage_reward=stage_reward_approx,
        d=d,
        n_mc=200,
    )

    print("Learned theta_Q (approx model):", theta_Q_approx)
    print("Learned average reward eta_hat (approx model):", eta_hat_approx)

    plot_Q_lspe_vs_Qhat(
        theta_Q=theta_Q_approx,
        eta_hat=eta_hat_approx,
        data_x=data_x_approx,
        data_u=data_u_approx,
        policy=pol_cl,
        model_step=model_step_approx,
        stage_reward=stage_reward_approx,
        psi=psi,
        n_points=100,       # how many (x,u) pairs to plot
        T_mc=100,          # length of MC rollouts
        n_traj_mc=200,       # MC trajectories per (x,u)
        seed=0,
    )