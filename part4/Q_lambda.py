import numpy as np
from model import model_step, reward


def q_lambda_learning_LQR(model, N, T, f_x0, alpha=1e-4, alpha_mul=[], lambda_=0.9, seed=0, true_system=False, u_noise_std=0.1):
    print(f"Starting Q-λ learning with λ={lambda_}, α={alpha}, true_system={true_system}")
    rng = np.random.default_rng(seed)
    np.random.seed(seed)
    F, G, H, E, D, Q, R, S = model
    nx = F.shape[0]
    nu = G.shape[1]

    # --- Feature function ψ(x,u): quadratic monomials of [x;u]
    def psi(x, u):
        xu = np.concatenate([x, u]) # = [x1, x2, x3, u1]
        return np.outer(xu, xu).flatten() # = [x1^2, x1*x2, x1*x3, x1*u1, x2^2, ..., u1^2]
    n_feat = (nx + nu) ** 2

    # --- Initialization
    theta = np.zeros(n_feat)     # parameter vector theta_0
    zeta = np.zeros(n_feat)      # eligibility trace zeta_0

    # --- Helper: compute greedy control from θ
    # phi(x,u)_ = argmin_u Q(x,u;θ) = - (H_uu)^(-1) H_ux x
    def compute_K(theta):
        H = theta.reshape((nx + nu, nx + nu))
        H = 0.5 * (H + H.T)  # symmetrize H
        H_ux = H[nx:, :nx]
        H_uu = H[nx:, nx:]
        H_uu = 0.5 * (H_uu + H_uu.T) + 1e-3 * np.eye(nu)
        return -np.linalg.solve(H_uu, H_ux)

    def greedy_policy(theta, x): ####### minimise Q(x,u;θ)
        return compute_K(theta) @ x
    # dict to save data
    n_saves = 2000
    save_index = 0
    data = {"TD_errors": np.zeros(n_saves), "K": np.zeros((n_saves, nu, nx)), "zeta": np.zeros((n_saves, n_feat)), "states": np.zeros((n_saves, nx)), "actions": np.zeros((n_saves, nu)), "i": np.zeros(n_saves)}

    # --- Main loop
    n_iter = N * T
    x = f_x0()

    for n in range(N):
        x = f_x0()
        # if n == 1000_000:
        #     alpha *= 10
        # if n == (N * 9  // 10): alpha *= 0.1
        if alpha_mul and n == int(alpha_mul[0][0] * N):
            alpha *= alpha_mul[0][1]
            alpha_mul.pop(0)
            print(f"Step size alpha changed to {alpha} at iteration {n}")

        for t in range(T):
            x = np.clip(x, -1, 1)  # clip state to avoid explosion

            u = greedy_policy(theta, x) + rng.normal(0, u_noise_std, size=(nu,))  # exploration noise

            # compute cost
            

            # next state
            if true_system:
                # x_next = F @ x + G @ u 
                c = - reward(x.reshape(1, -1), u.reshape(1, -1), np.zeros(1), 1)[0, 0]
                x_next = model_step(x, u[0], xi_a=0)
            else:
                c = float(x.T @ Q @ x + 2 * x.T @ S @ u + u.T @ R @ u) # minimize this cost
                x_next = F @ x + G @ u # skipped : + D @ (0.1 * np.random.randn(D.shape[1]))
            

            # if x_next is exploding
            if np.linalg.norm(x_next) > 1e2:
                print("State exploded, continue")
                continue

            u_next = greedy_policy(theta, x_next)
            # print(f"{u=} {theta=}")
            # print(f"{x_next=} {u_next=}")

            # featuress
            psi_now = psi(x, u)
            psi_next = psi(x_next, u_next)

            # current Q estimates
            Q_now = float(theta @ psi_now)
            Q_next = float(theta @ psi_next)

            # TD error D_{n+1}
            # Q_now = c + Q_next

            D = -Q_now + c + Q_next
            # D = np.clip(D, -10.0, 10.0)

            zeta = lambda_ * zeta + psi_now

            # update θ
            # theta += alpha * D * zeta
            # ## higher steps size for x2 and x3
            alpha_vec = np.array([1, 5, 5, 1])
            alpha_vec = np.outer(alpha_vec, alpha_vec).flatten()
            ## element-wise multiply
            theta += np.multiply(alpha_vec, alpha * D * zeta)
            

            # zeta = lambda_ * zeta + psi_next

            x = x_next

            steps = n * T + t

            # track current K from θ
            # if (n + epoch * len(u_array)) % (n_iter // (n_saves-1)) == 0:
            if (steps) % (n_iter // (n_saves)) == 0:
                K_est = compute_K(theta)
                data["K"][save_index] = K_est.copy()
                data["zeta"][save_index] = zeta.copy()
                data["TD_errors"][save_index] = D
                data["states"][save_index] = x.copy()
                data["actions"][save_index] = u.copy()
                data["i"][save_index] = steps
                save_index += 1

            if steps % (n_iter // 20) == 0:
                print(f"Iter {steps:5d}, TD Error: {D:.4f}, Current K: {K_est.flatten()} x: {np.array2string(x, precision=2)} u: {np.array2string(u, precision=2)} zeta norm: {np.linalg.norm(zeta):.2f}  cost: {c:.4f}")
            # if error is small enough, stop
            # if abs(D) < 1e-8:
            #     print(f"Converged at iteration {n}, TD Error: {D:.6f}")
            #     break

            if np.isnan(x).any():
                print("NaN detected, stopping training.")
                break
        else:
            continue
        break


    # final H and policy
    H = theta.reshape((nx + nu, nx + nu))
    K_learned = compute_K(theta)

    return K_learned, H, data


def get_q_lambda_policy(model=None, clipped=False, use_precomputed=True):
    """Get the Q-λ learning policy.
    
    Args:
        model: System model tuple. If None, uses lqr.model.
        clipped: If True, returns clipped policy (q ∈ [0, 1]).
        use_precomputed: If True, use precomputed K values (fast). Otherwise train (slow).
    
    Returns:
        A policy function that takes state x and returns control u.
    """
    import numpy as np
    
    if use_precomputed:
        # Precomputed from experiments on true system
        K_Q_lambda = np.array([-0.0865145, 0.00021523, 0.01188014])
    else:
        if model is None:
            from lqr import model as model_matrices
            model = model_matrices
        from model import SIGMA_A, BETA_U
        
        rng = np.random.default_rng(42)
        f_x0 = lambda: np.multiply(rng.normal(0, 1, size=(3)), np.array([1, SIGMA_A, BETA_U])) * 0.1
        N, T = 500_000, 2
        K_Q_lambda, _, _ = q_lambda_learning_LQR(
            model, N, T, f_x0, 
            lambda_=0.8, alpha=4e-1, 
            alpha_mul=[(0.07, 10), (0.9, 0.1)]
        )
    
    def q_lambda_policy(x):
        u = float(K_Q_lambda @ x)
        if clipped:
            u = float(np.clip(u, -x[0], 1 - x[0]))
        return u
    
    return q_lambda_policy