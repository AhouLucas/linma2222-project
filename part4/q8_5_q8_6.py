import numpy as np
import matplotlib.pyplot as plt 
from model import generate_trajectories, stage_reward, model_step, K_cl, pol_cl, reward
from lqr import model, stage_reward_approx, model_step_approx
from tqdm import tqdm

from scipy.linalg import solve_discrete_lyapunov

# from lqr import get_lqr_policy

from lqr import compute_lqr_gain
from plotting import plot_reward_distribution
from plotting import graph_K_evolution

def cond_expectation(x, u, policy, psi, model_step, stage_reward, n_mc=20):
    """
    Approximate:
        E[r_t | x_t=x, u_t=u]          -> scalar
        E[psi(x_{t+1}, pi(x_{t+1}))]   -> (d,)
    via Monte Carlo using the model dynamics.
    """
    u = float(np.asarray(u).reshape(-1)[0])
    x = np.asarray(x, dtype=float).reshape(-1)

    d = psi(x, u).shape[0]
    psi_next_mean = np.zeros(d)
    r_sum = 0.0

    for _ in range(n_mc):
        # one-step transition
        x_next = model_step(x, u)
        r = stage_reward(x, u, x_next)
        r = float(np.mean(r))

        r_sum += r

        u_next = policy(x_next)
        psi_next_mean += psi(x_next, u_next)

    psi_next_mean /= n_mc
    r_bar = r_sum / n_mc       # this is now a scalar float

    return r_bar, psi_next_mean


def generate_dataset(sigma_exp=0.1, x0=np.array([0.1, 0.01, 0.01]), T=1000, N=1, model_step_fn=model_step, burn_in=10):
    policy = lambda x: np.random.normal(K_cl @ x, sigma_exp)
    x, u, xi_p = generate_trajectories(policy, x0, T=T, N=N, model_step_fn=model_step_fn)
    # transpose -> (T, N, 3), then reshape to (T*N, 3)
    x = x[burn_in:-1].transpose(0, 2, 1).reshape(-1, x.shape[1])
    u = u[burn_in:].reshape(-1)
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

def extract_K_from_theta(theta_Q): #### DEPENDING ON PSI DEFINITION
    a = theta_Q[7]

    b0  = theta_Q[3]
    bq  = theta_Q[10]
    bza = theta_Q[12]
    bzu = theta_Q[13]

    # Effective linear gain (ignore constant b0)
    K = np.array([
        -bq  / (2*a),
        -bza / (2*a),
        -bzu / (2*a)
    ])
    return K




def lspe(data_x, data_u, W, policy, psi, model_step, stage_reward, n_mc=20):
    N = min(data_x.shape[0], data_u.shape[0])

    # Infer feature dimension from first sample
    d = psi(data_x[0], data_u[0]).shape[0]
    d_theta = d + 1  # θ = [θ_Q; η]

    A = np.zeros((d_theta, d_theta))
    b = np.zeros(d_theta)

    psi_xu = np.stack([psi(data_x[i], data_u[i]) for i in range(N)], axis=0)

    # for k in range(N):
    # progress bar
    for k in tqdm(range(N), desc="LSPE"):
        x_k = data_x[k]
        u_k = data_u[k]

        # 1) Conditional expectations at (x_k, u_k)
        r_bar_k, psi_next_mean_k = cond_expectation(
            x_k, u_k, policy, psi, model_step, stage_reward, n_mc=n_mc
        )

        # 2) Current features
        # psi_now_k = psi(x_k, u_k)
        psi_now_k = psi_xu[k]

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


# --- load approximate LQR model matrices ---
F, G, H, E, D, Qc, Rc, Sc = model  # adjust unpacking if your model() returns different
K = K_cl.reshape(1, -1)                # shape (1, nx) for matrix products
# Closed-loop matrix A = F + G K
A = F + G @ K                          # F: (nx,nx), G: (nx,1), K: (1,nx) -> A: (nx,nx)
# Per-step cost under closed-loop policy: ℓ_π(x) = x^T Q_pi x
Q_pi = (
    Qc
    + K.T @ Rc @ K
    + (Sc @ K + K.T @ Sc.T)
)

# Solve discrete-time Lyapunov equation for *cost* differential value:
#   P = A^T P A + Q_pi
P = solve_discrete_lyapunov(A.T, Q_pi)


def Q_exact_lqr(x, u):
    x = np.asarray(x, dtype=float).reshape(-1, 1)   # (nx, 1)
    u = float(np.asarray(u).reshape(-1)[0])
    u = np.array([[u]])                             # (1, 1)

    # -- immediate cost: ℓ(x,u) = x^T Qc x + u^T Rc u + 2 x^T Sc u
    # NOTE: This is the *cost*, not reward.
    cost_immediate = (
        (x.T @ Qc @ x)[0, 0]
        + (u.T @ Rc @ u)[0, 0]
        + 2.0 * (x.T @ Sc @ u)[0, 0]
    )

    # -- next state under approximate LQR dynamics (no noise term, its contribution is a constant)
    x_next = F @ x + G @ u   # (nx, 1)

    # differential value V(x) = x^T P x  (for *cost*)
    V_x      = (x.T      @ P @ x     )[0, 0]
    V_xnext  = (x_next.T @ P @ x_next)[0, 0]

    # Poisson Q for *cost*:
    Q_cost = cost_immediate + V_xnext - V_x

    # Convert to reward-based Q: Q_reward = -Q_cost (up to an additive constant)
    Q_reward = -Q_cost

    return float(Q_reward)

def plot_Q_lspe_vs_Q_fn(theta_Q,
                        data_x, data_u,
                        Q_function,
                        psi,
                        n_points=100,
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

        q_hat = Q_function(x_k, u_k)

        Q_lspe_vals.append(q_lspe)
        Q_hat_vals.append(q_hat)

    Q_lspe_vals = np.array(Q_lspe_vals)
    Q_hat_vals  = np.array(Q_hat_vals)
    
    offset = np.mean(Q_hat_vals - Q_lspe_vals)
    print(f"Poisson Q offset applied: {offset:.4f}")

    Q_lspe_vals += offset


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
    plt.title("LSP" \
    "E Poisson Q vs Monte-Carlo Poisson Q")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('part4/figures/Q_lspe_vs_Qhat.pdf', format='pdf')






############################################
# LSPE + Policy Improvement (Q8.7 / Q8.9) #
############################################

def greedy_policy_from_theta_analytic(theta_Q, _psi, constrained=False):
    assert _psi == psi, "This function assumes a specific psi definition."
    θ = theta_Q

    def policy(x):
        q, za, zu = np.asarray(x).reshape(-1)[:3]

        # Quadratic coefficients
        a = θ[7]  # coefficient of u^2
        b = θ[3] + θ[10]*q + θ[12]*za + θ[13]*zu  # linear in u

        # If a < 0 → parabola opens downward → unique max
        if a < 0:
            u_star = -b / (2*a)
        else:
            # convex or flat: max achieved at boundaries
            u_star = 0.0  # or choose a boundary, but 0 is fine
            
        if constrained:
            u_min = -q
            u_max = 1.0 - q
            u_star = np.clip(u_star, u_min, u_max)
        return float(u_star)

    return policy

def greedy_policy_from_theta_unconstrained(theta_Q, psi, constrained=False, n_grid=201):
    base_u_grid = np.linspace(-1.0, 1.0, n_grid)
    def policy(x):
        if constrained:
            x = np.asarray(x).reshape(-1)
            q = float(x[0])
            u_grid = np.linspace(-q, 1.0 - q, n_grid)
        else:
            u_grid = base_u_grid
            
        vals = [float(theta_Q @ psi(x, u)) for u in u_grid]
        return float(u_grid[int(np.argmax(vals))])

    return policy



def lspe_pi(initial_policy,
            data_x,
            data_u,
            W,
            psi,
            model_step,
            stage_reward,
            n_mc=1000,
            n_pi_iters=5,
            constrained=False,
            u_min=-1.0,
            u_max=1.0,
            n_grid=201):
    """
    LSPE+PI outer loop.

    - initial_policy: π_0 used as starting point (e.g. pol_cl)
    - data_x, data_u: dataset collected with some exploration policy π_exp
                      (here generated by generate_dataset)
    - constrained: if True, use constrained improvement (Q8.9); else unconstrained (Q8.7)
    - returns: (final_policy, theta_Q_last, eta_hat_last)
    """
    policy = initial_policy
    theta_Q_last = None
    eta_hat_last = None
    K_list = []

    for k in range(n_pi_iters):
        print(f"\n=== LSPE+PI iteration {k} ===")
        theta_Q, eta_hat = lspe(
            data_x, data_u,
            W=W,
            policy=policy,
            psi=psi,
            model_step=model_step,
            stage_reward=stage_reward,
            n_mc=n_mc,
        )

        theta_Q_last = theta_Q
        eta_hat_last = eta_hat

        K_current = extract_K_from_theta(theta_Q)
        K_list.append(K_current)

        print(f"{k} :  eta_hat = {eta_hat} | K = {K_current}")

        policy = greedy_policy_from_theta_analytic(theta_Q, psi, constrained=constrained)

    return policy, theta_Q_last, eta_hat_last, np.array(K_list)


def q8_5():
    data_x, data_u, xi_p = generate_dataset(T=50, N=100)

    theta_Q, eta_hat = lspe(
        data_x, data_u,
        W=W,
        policy=pol_cl,
        psi=psi,
        model_step=model_step,
        stage_reward=stage_reward,
        n_mc=100,
    )

    print("Learned theta_Q:", theta_Q)
    print("Learned average reward eta_hat:", eta_hat)

    Q_hat_fn = lambda x, u: Q_hat_mc(x, u, pol_cl, model_step, stage_reward, eta=eta_hat, T=50, n_traj=400)
    plot_Q_lspe_vs_Q_fn(
        theta_Q=theta_Q,
        data_x=data_x,
        data_u=data_u,
        psi=psi,
        Q_function=Q_hat_fn,
        n_points=100,       # how many (x,u) pairs to plot
        seed=0,
    )
    plt.show()



def q8_6():
    ####### on the approximate model
    data_x_approx, data_u_approx, xi_p_approx = generate_dataset(T=50, N=100, model_step_fn=model_step_approx)
    print("Dataset on approximate model generated.")

    theta_Q_approx, eta_hat_approx = lspe(
        data_x_approx, data_u_approx,
        W=W,
        policy=pol_cl,
        psi=psi,
        model_step=model_step_approx,
        stage_reward=stage_reward_approx,
        n_mc=100,
    )

    print("Learned theta_Q (approx model):", theta_Q_approx)
    print("Learned average reward eta_hat (approx model):", eta_hat_approx)

    plot_Q_lspe_vs_Q_fn(
        theta_Q=theta_Q_approx,
        data_x=data_x_approx,
        data_u=data_u_approx,
        psi=psi,
        Q_function=Q_exact_lqr,
        n_points=100,       # how many (x,u) pairs to plot
        seed=0,
    )

    plt.show()



def q8_7():
    # --- Data from exploration policy π_exp (same setting as Q8.5) ---
    data_x_ls, data_u_ls, xi_p_ls = generate_dataset(T=50, N=50)
    print("Dataset for LSPE+PI on true system generated (Q8.7).")



    # Run LSPE+PI (unconstrained improvement)
    pi_lspepi, theta_Q_last, eta_hat_last, K_array = lspe_pi(
        initial_policy=pol_cl, # Initial policy is π_cl
        data_x=data_x_ls,
        data_u=data_u_ls,
        W=W,
        psi=psi,
        model_step=model_step,
        stage_reward=stage_reward,
        n_mc=50,
        n_pi_iters=5,      # number of PI iterations – you can tune this
        constrained=False, # <-- Q8.7 = unconstrained improvement
        u_min=-1.0,
        u_max=1.0,
        n_grid=201,
    )

    print("Final eta_hat (Q8.7) =", eta_hat_last)
    ##### COMPARE POLICIES ######

    policy_list.append((pi_lspepi, "LSPE+PI Learned Policy"))
    plot_reward_distribution( policy_list, name="Q8_7_reward_distribution", T=1000, n_traj=100)


def q8_8():
    print("=== QUESTION 8.8: LSPE+PI on approximate model, evaluate on true system ===")

    # 1) Generate dataset ON THE APPROXIMATE MODEL
    data_x_ap, data_u_ap, xi_p_ap = generate_dataset(sigma_exp=0.5, T=300, burn_in=100, N=100, model_step_fn=model_step_approx)
    print("Dataset generated on approximate model (Q8.8).")

    # 3) Run LSPE+PI but *model_step and stage_reward come from approximate model*
    pi_lspepi_ap, theta_Q_ap_last, eta_hat_ap_last, K_array_ap = lspe_pi(
        initial_policy=pol_cl,
        data_x=data_x_ap,
        data_u=data_u_ap,
        W=W,
        psi=psi,
        model_step=model_step_approx,      # approximate model here
        stage_reward=stage_reward_approx,  # approximate reward
        n_mc=50,
        n_pi_iters=5,
        constrained=False,    # Q8.8 uses UNCONSTRAINED version
    )
    # K_array = np.array([theta_Q_ap_last[:3]])  # store K at each iteration
    graph_K_evolution({"K_k":K_array_ap}, K_lqr, title_suffix="during LSPE+PI on Approx Model")

    print("Final eta_hat (Q8.8, approx model) =", eta_hat_ap_last)

    policy_list.append((pi_lspepi_ap, "LSPE+PI Learned Policy (Approx Model)"))
    plot_reward_distribution( policy_list, name="Q8_8_reward_distribution", T=1000, n_traj=100)

################
# Question 8.9 #
################

def q8_9():
    # Use the same exploration dataset (or regenerate a new one if you prefer)
    data_x_ls_c, data_u_ls_c, xi_p_ls_c = generate_dataset(T=50, N=100)
    print("Dataset for constrained LSPE+PI on true system generated (Q8.9).")

    # Run LSPE+PI with CONSTRAINED improvement (0 ≤ q + u ≤ 1)
    pi_lspepi_constr, theta_Q_last_c, eta_hat_last_c, K_array_c = lspe_pi(
        initial_policy=pol_cl,
        data_x=data_x_ls_c,
        data_u=data_u_ls_c,
        W=W,
        psi=psi,
        model_step=model_step,
        stage_reward=stage_reward,
        n_mc=50,
        n_pi_iters=5,
        constrained=True,   # <-- Q8.9 = constrained improvement
        n_grid=201,
    )
    print("Final eta_hat (Q8.9, constrained) =", eta_hat_last_c)

    policy_list.append((pi_lspepi_constr, "LSPE+PI Learned Policy"))
    plot_reward_distribution( policy_list, name="Q8_9_reward_distribution", T=1000, n_traj=1000)


if __name__ == "__main__":
    d = 14  # length of psi vector
    W = 1e-4 * np.eye(d + 1)

    K_lqr = compute_lqr_gain(model)
    pol_lqr = lambda x: float(K_lqr @ x)
    
    policy_list = [(pol_lqr, "LQR Optimal Policy"), 
                   (pol_cl, "Closed-Loop Policy")]
    
    # q8_5()
    # q8_6()
    # q8_7()
    q8_8()
    # q8_9()

    
    plt.show()
