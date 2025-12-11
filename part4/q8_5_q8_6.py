import numpy as np
import matplotlib.pyplot as plt 
from tqdm import tqdm

from scipy.linalg import solve_discrete_are, solve_discrete_lyapunov
from scipy.optimize import minimize
import itertools

from model import generate_trajectories, stage_reward, model_step, K_cl, pol_cl, reward
from lqr import model, stage_reward_approx, model_step_approx, compute_lqr_gain
from plotting import plot_reward_distribution, plot_trajectories, plot_Xfn_vs_Yfn, graph_K_evolution
# from lqr import get_lqr_policy

from model import get_avg_reward


def cond_expectation(x, u, policy, psi, model_step, stage_reward=stage_reward, n_mc=20):
    """ approximate via Monte-Carlo:
        r_bar = E[ r(x,u) ]
        psi_next_mean = E[ psi(x', policy(x')) ]
    """
    u = float(np.asarray(u).reshape(-1)[0])
    x = np.asarray(x, dtype=float).reshape(-1)

    d = psi(x, u).shape[0]
    psi_next_mean = np.zeros(d)
    r_sum = 0.0

    for _ in range(n_mc):
        # one-step transition
        x_next = model_step(x, u)
        r = stage_reward(x, u)
        r = float(np.mean(r))

        r_sum += r

        u_next = policy(x_next)
        psi_next_mean += psi(x_next, u_next)

    psi_next_mean /= n_mc
    r_bar = r_sum / n_mc 

    return r_bar, psi_next_mean


def generate_dataset(sigma_exp=0.1, x0=np.array([0.1, 0.01, 0.01]), T=1000, N=1, model_step_fn=model_step, burn_in=10, 
                     base_policy=pol_cl):
    policy = lambda x: np.random.normal(base_policy(x), sigma_exp)
    x, u, xi_p = generate_trajectories(policy, x0, T=T, N=N, model_step_fn=model_step_fn)
    # transpose -> (T, N, 3), then reshape to (T*N, 3)
    x = x[burn_in:-1].transpose(0, 2, 1).reshape(-1, 3)
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


def psi_d4(x, u):
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

def extract_K_from_theta(theta_Q): #### DEPENDING ON PSI DEFINITION
    a = theta_Q[7]

    b0  = theta_Q[3]
    bq  = theta_Q[10]
    bza = theta_Q[12]
    bzu = theta_Q[13]

    K = np.array([ -bq  / (2*a), -bza / (2*a),-bzu / (2*a)])
    return K




def lspe(data_x, data_u, policy, psi, model_step, stage_reward, n_mc=20):
    N = min(data_x.shape[0], data_u.shape[0])

    # Infer feature dimension from first sample
    d = psi(data_x[0], data_u[0]).shape[0]
    d_theta = d + 1  # θ = [θ_Q; η]

    A = np.zeros((d_theta, d_theta))
    b = np.zeros(d_theta)

    psi_xu = np.stack([psi(data_x[i], data_u[i]) for i in range(N)], axis=0)

    for k in tqdm(range(N), desc="LSPE"):
        x_k = data_x[k]
        u_k = data_u[k]


        r_bar_k, psi_next_mean_k = cond_expectation(x_k, u_k, policy, psi, model_step, stage_reward, n_mc=n_mc)
        psi_now_k = psi_xu[k] # precomputed psi(x_k, u_k)

        # 3) phi_k = [ E[psi_next] - psi_now ; -1 ]
        phi_k = np.concatenate([psi_next_mean_k - psi_now_k, np.array([-1.0])])

        A += np.outer(phi_k, phi_k) # A +-= E[φ φ^T]
        b -= phi_k * r_bar_k        # b +-= -E[φ r_bar]

    A /= N
    b /= N

    A_reg = A + (1e-3 * np.eye(d + 1)) / N # regularization for stability

    theta = np.linalg.solve(A_reg, b)

    theta_Q = theta[:-1]  # Q parameters
    eta_hat = theta[-1]   # average reward
    return theta_Q, eta_hat


def Q_hat_mc(x, u, policy, model_step, stage_reward, eta=0.0, T=1000, n_traj=20):
    """
    Monte-Carlo approx: Q(x,u) +-= E[ sum_{t=0}^{T-1} (r_t - eta) ]
    """
    x = np.asarray(x, dtype=float).copy()
    u = float(np.asarray(u).reshape(-1)[0])

    q_vals = []

    for _ in range(n_traj):
        x_curr = x.copy()
        total = 0.0

        for _t in range(T):
            u_t = u if _t <= 0 else float(policy(x_curr))
            r = stage_reward(x_curr, u_t)
            r = float(np.mean(np.asarray(r)))
            total += r - eta   # <-- subtract eta each step
            x_curr = model_step(x_curr, u_t)

        q_vals.append(total)

    return float(np.mean(q_vals))


# --- load approximate LQR model matrices ---
F, G, H, E, D, Qc, Rc, Sc = model  # adjust unpacking if your model() returns different
print(f"{Qc=}, {Rc=}, {Sc=}")
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
    cost_immediate = - stage_reward_approx(x, u) # minimize cost 

    # -- next state under approximate LQR dynamics (no noise term, its contribution is a constant)
    x_next = F @ x + G @ u   # (nx, 1)
    # x_next = model_step_approx(x, u, xi_a=np.zeros((1,)))  # (nx, 1)

    # differential value V(x) = x^T P x  (for *cost*)
    V_x      = (x.T      @ P @ x     )[0, 0]
    V_xnext  = (x_next.T @ P @ x_next)[0, 0]

    # Poisson Q for *cost*:
    Q_cost = cost_immediate + V_xnext - V_x

    # Convert to reward-based Q: Q_reward = -Q_cost (up to an additive constant)
    Q_reward = -Q_cost

    return float(Q_reward) * 2






############################################
# LSPE + Policy Improvement (Q8.7 / Q8.9) #
############################################

def greedy_policy_from_theta_d2(theta_Q, _psi, constrained=False):
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



def greedy_policy_from_theta(theta_Q, psi, constrained=False, n_grid=201):
    # use minimize

    def policy(x):
        x = np.asarray(x).reshape(-1)
        
        if constrained:
            q = float(x[0])
            u_min = -q
            u_max = 1.0 - q
            bounds = [(u_min, u_max)]
        else:
            bounds = [(-1.0, 1.0)]
        
        res = minimize(lambda u: -float(theta_Q @ psi(x, u)), 
                       x0=np.array([0.0]), bounds=bounds, method='L-BFGS-B')
        return float(res.x[0])
    return policy


def lspe_pi(initial_policy, psi,model_step, stage_reward,
            n_mc=1000,
            n_pi_iters=5,
            constrained=False,
            get_pol_fn=greedy_policy_from_theta,
            sigma_exp=0.1,
            T_data=200,
            burn_in=50,
            N_traj=100,):
    
    policy = initial_policy
    theta_Q_last = None
    eta_hat_last = None
    K_list = []

    for k in range(n_pi_iters):
        print(f"\n=== LSPE+PI iteration {k} ===")
        data_x, data_u, xi_p = generate_dataset(sigma_exp=sigma_exp,T=T_data,burn_in=burn_in,N=N_traj,model_step_fn=model_step,base_policy=policy)

        theta_Q, eta_hat = lspe(
            data_x, data_u,
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

        policy = get_pol_fn(theta_Q, psi, constrained=constrained)

        # test the policy and print average reward
        avg_reward = get_avg_reward(policy, T=100, N=N_traj)
        print(f"  -> avg reward of new policy: {avg_reward}")

    return policy, theta_Q_last, eta_hat_last, np.array(K_list)


def q8_5():
    data_x, data_u, xi_p = generate_dataset(T=50, N=200)
    plot_trajectories(data_x, data_u, xi_p, filename=f"dataset_trajectories_Q85")

    theta_Q, eta_hat = lspe(data_x, data_u,
        policy=pol_cl,
        psi=psi,
        model_step=model_step,
        stage_reward=stage_reward,
        n_mc=100,
    )

    print("Learned theta_Q:", theta_Q)
    print("Learned average reward eta_hat:", eta_hat)

    Q_hat_fn = lambda x, u: Q_hat_mc(x, u, pol_cl, model_step, stage_reward, eta=eta_hat, T=50, n_traj=400)
    plot_Xfn_vs_Yfn(data_x=data_x,data_u=data_u,
        X_function=lambda x, u: float(theta_Q @ psi(x, u)),
        Y_function=Q_hat_fn,
        n_points=100,       # how many (x,u) pairs to plot
        seed=0,
        x_label=r"$Q_{\mathrm{LSPE}}$",
        y_label=r"$\hat Q_{\mathrm{MC}}$ (Poisson)"
    )



def q8_6():
    ####### on the approximate model
    print("=== QUESTION 8.6: LSPE on approximate model ===")
    # data_x_approx, data_u_approx, xi_p_approx = generate_dataset(T=100, burn_in=50, N=200, model_step_fn=model_step_approx)
    data_x_approx, data_u_approx, xi_p_approx = generate_dataset(sigma_exp=0.5, T=200, burn_in=100, N=200, model_step_fn=model_step_approx)
    # plot_trajectories(data_x_approx, data_u_approx, xi_p_approx, filename=f"dataset_trajectories_Q86_approx_model")


    theta_Q_approx, eta_hat_approx = lspe(data_x_approx, data_u_approx, policy=pol_cl,
        psi=psi,
        # model_step=model_step_approx,
        model_step=model_step_approx_no_noise,
        stage_reward=stage_reward_approx,
        n_mc=1,
    )

    print("Learned theta_Q (approx model):", theta_Q_approx)
    print("Learned average reward eta_hat (approx model):", eta_hat_approx)



    plot_Xfn_vs_Yfn(data_x=data_x_approx, data_u=data_u_approx,
        X_function=lambda x, u: float(theta_Q_approx @ psi(x, u)),
        Y_function=Q_exact_lqr,
        n_points=1000,
        seed=0,
        x_label=r"$Q_{\mathrm{LSPE}}$ (Approx Model)",
        y_label=r"$Q_{\mathrm{Exact}}$ (LQR Model)"
    )

    plot_Xfn_vs_Yfn(
        X_function=lambda x, u: float(theta_Q_approx @ psi(x, u)),
        Y_function=Q_exact_lqr,
        n_points=1000,
        seed=0,
        x_label=r"$Q_{\mathrm{LSPE}}$ (Approx Model)",
        y_label=r"$Q_{\mathrm{Exact}}$ (LQR Model) "
    )



def q8_7_d4():
    # --- Data from exploration policy π_exp (same setting as Q8.5) ---
    print("Dataset for LSPE+PI on true system generated (Q8.7).")


    # Run LSPE+PI (unconstrained improvement)
    # data_x_ls, data_u_ls, xi_p_ls = generate_dataset(T=100, burn_in=50, N=50, sigma_exp=0.1, model_step_fn=model_step)
    pi_lspepi, theta_Q_last, eta_hat_last, K_array = lspe_pi(initial_policy=pol_cl,
        T_data=100, burn_in=50, N_traj=50, sigma_exp=0.1, 
        psi=psi_d4, get_pol_fn=greedy_policy_from_theta,
        model_step=model_step,stage_reward=stage_reward,
        n_mc=50,
        n_pi_iters=10,
        constrained=False, # <-- Q8.7 = unconstrained improvement
    )

    print("Final eta_hat (Q8.7) =", eta_hat_last)
    ##### COMPARE POLICIES ######

    policy_list.append((pi_lspepi, "LSPE+PI (Unconstrained, True System, psi d4)"))
    plot_reward_distribution( policy_list, name="Q8_7_reward_distribution_d4", T=1000, n_traj=100)


def q8_7():
    # --- Data from exploration policy π_exp (same setting as Q8.5) ---
    # data_x_ls, data_u_ls, xi_p_ls = generate_dataset(T=50, N=200)
    # plot_trajectories(data_x_ls, data_u_ls, xi_p_ls, filename=f"dataset_trajectories_Q87_true_system")
    # plt.show()
    print("Dataset for LSPE+PI on true system generated (Q8.7).")
    # Run LSPE+PI (unconstrained improvement)
    pi_lspepi, theta_Q_last, eta_hat_last, K_array = lspe_pi(initial_policy=pol_cl,
        T_data=500, burn_in=50, N_traj=100, sigma_exp=0.1, 
        psi=psi, get_pol_fn=greedy_policy_from_theta_d2,
        model_step=model_step,stage_reward=stage_reward,
        n_mc=50,
        n_pi_iters=10,
        constrained=False, # <-- Q8.7 = unconstrained improvement
    )


    # training on c_quad => converges to LQR
    # pi_lspepi_cq, theta_Q_last_cq, eta_hat_last_cq, K_array_cq = lspe_pi(initial_policy=pol_cl,
    #     T_data=500, burn_in=50, N_traj=50, sigma_exp=0.1,
    #     psi=psi, get_pol_fn=greedy_policy_from_theta_d2,
    #     model_step=model_step,stage_reward=stage_reward_c_quad,
    #     n_mc=50,
    #     n_pi_iters=5,
    #     constrained=False, # <-- Q8.7 = unconstrained improvement
    # )

    print("Final eta_hat (Q8.7) =", eta_hat_last)
    ##### COMPARE POLICIES ######

    policy_list.append((pi_lspepi, "LSPE+PI (Unconstrained, True System)"))
    plot_reward_distribution( policy_list, name="Q8_7_reward_distribution", T=1000, n_traj=1000)


def q8_8(): # approx model, 
    print("=== QUESTION 8.8: LSPE+PI on approximate model, evaluate on true system ===")
    data_x_ap, data_u_ap, xi_p_ap = generate_dataset(sigma_exp=0.5, T=300, burn_in=100, N=100, model_step_fn=model_step_approx)
    print("Dataset generated on approximate model (Q8.8).")

    # 3) Run LSPE+PI but *model_step and stage_reward come from approximate model*
    pi_lspepi_ap, theta_Q_ap_last, eta_hat_ap_last, K_array_ap = lspe_pi( initial_policy=pol_cl,
        data_x=data_x_ap,data_u=data_u_ap,
        psi=psi, get_pol_fn=greedy_policy_from_theta_d2,
        model_step=model_step_approx,stage_reward=stage_reward_approx,  # approximate model here
        n_mc=50,
        n_pi_iters=5,
        constrained=False,    # Q8.8 uses UNCONSTRAINED version
    )

    print("Final eta_hat (Q8.8, approx model) =", eta_hat_ap_last)
    graph_K_evolution({"K_k":K_array_ap}, K_lqr, title_suffix="during LSPE+PI on Approx Model")


    policy_list.append((pi_lspepi_ap, "LSPE+PI (Approx Model)"))
    plot_reward_distribution( policy_list, name="Q8_8_reward_distribution", T=1000, n_traj=100)

################
# Question 8.9 #
################

def q8_9(): # constrained improvement on true system
    data_x_ls_c, data_u_ls_c, xi_p_ls_c = generate_dataset(T=50, N=100)
    print("Dataset for constrained LSPE+PI on true system generated (Q8.9).")

    
    pi_lspepi_constr, theta_Q_last_c, eta_hat_last_c, K_array_c = lspe_pi( initial_policy=pol_cl,
        data_x=data_x_ls_c, data_u=data_u_ls_c, 
        psi=psi, get_pol_fn=greedy_policy_from_theta_analytic,
        model_step=model_step, stage_reward=stage_reward,
        n_mc=50,
        n_pi_iters=5,
        constrained=True,
    )
    print("Final eta_hat (Q8.9, constrained) =", eta_hat_last_c)

    policy_list.append((pi_lspepi_constr, "LSPE+PI (Constrained)"))
    plot_reward_distribution( policy_list, name="Q8_9_reward_distribution", T=1000, n_traj=1000)

def check_Q_exact_vs_Q_hat():    
    ### plot Q_exact vs Q_hat_mc
    print("=== Checking Q_exact_lqr vs Q_hat_mc ===")
    plot_Xfn_vs_Yfn(
        X_function=Q_exact_lqr,
        Y_function=lambda x, u: Q_hat_mc(x, u, pol_cl, model_step_approx_no_noise, stage_reward_approx, T=1000, n_traj=1),
        n_points=200, 
        seed=0,
        title="Comparison",
        x_label=r"$Q_{\mathrm{LQR}}$",
        y_label=r"$\hat{Q}_{\mathrm{MC}}$ (LQR, No Noise, $T=1000$)"
    )


if __name__ == "__main__":
    d = 14  # length of psi vector
    

    K_lqr = compute_lqr_gain(model)
    pol_lqr = lambda x: float(K_lqr @ x)
    
    policy_list = [(pol_lqr, "LQR Optimal Policy"), 
                   (pol_cl, "Closed-Loop Policy")]
    

    model_step_approx_no_noise = lambda x, u: model_step_approx(x, u, xi_a=np.zeros(1))
    model_step_no_noise = lambda x, u: model_step(x, u, xi_a=0.0)
    stage_reward_no_noise = lambda x, u: stage_reward(x, u, xi_p_t=0.0)
    from model import c_quad, g
    stage_reward_c_quad = lambda x, u: c_quad(g(x[0], x[1], x[2], u))
    stage_reward_c_quad_no_noise = lambda x, u: c_quad(g(x[0], x[1], x[2], u, xi_p=0.0))


    



    # plt.show()

    
    # q8_5()
    # check_Q_exact_vs_Q_hat()
    # q8_6()
    
    q8_7()
    # q8_7_d4()


    # q8_8()
    # q8_9()



    
    plt.show()
