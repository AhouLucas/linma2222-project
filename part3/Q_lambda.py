import numpy as np

def q_lambda_learning_LQR(model, alpha=1e-4, lambda_=0.9,
                          n_iter=50000, x0=None,
                          K_init=None, seed=0):
    
    F, G, H, E, D, Q, R, S = model
    np.random.seed(seed)
    nx = F.shape[0]
    nu = G.shape[1]

    # --- Feature function ψ(x,u): quadratic monomials of [x;u]
    def psi(x, u):
        xu = np.concatenate([x, u])
        return np.outer(xu, xu).flatten()  # vector of all quadratic terms

    n_feat = (nx + nu) ** 2

    # --- Initialization
    theta = np.zeros(n_feat)     # parameter vector θ₀
    zeta = np.zeros(n_feat)      # eligibility trace ζ₀
    hist_K = []

    # initial state
    if x0 is None:
        x = np.random.randn(nx)
    else:
        x = np.array(x0, dtype=float)

    # small exploration gain
    if K_init is None:
        K = np.zeros((nu, nx))
    else:
        K = K_init.copy()

    # --- Helper: compute greedy control from θ
    def greedy_policy(theta, x):
        H = theta.reshape((nx + nu, nx + nu))
        H_xx = H[:nx, :nx]
        H_ux = H[nx:, :nx]
        H_uu = H[nx:, nx:]
        # ensure symmetric & invertible
        H_uu = 0.5 * (H_uu + H_uu.T) + 1e-6 * np.eye(nu)
        return -np.linalg.solve(H_uu, H_ux) @ x

    # --- Main loop
    for n in range(n_iter):
        # ε-greedy exploration around current policy
        u = greedy_policy(theta, x) + 0.1 * np.random.randn(nu)

        # compute cost
        c = float(x.T @ Q @ x + 2 * x.T @ S @ u + u.T @ R @ u)

        # next state
        x_next = F @ x + G @ u

        # greedy next control (for bootstrapping)
        u_next = greedy_policy(theta, x_next)

        # features
        psi_now = psi(x, u)
        psi_next = psi(x_next, u_next)

        # current Q estimates
        Q_now = float(theta @ psi_now)
        Q_next = float(theta @ psi_next)

        # TD error D_{n+1}
        D = -Q_now + c + Q_next

        # update θ
        theta += alpha * D * zeta

        # update eligibility trace
        zeta = lambda_ * zeta + psi_next

        # prepare next step
        x = x_next

        # track current K from θ
        if n % 1000 == 0 or n == n_iter - 1:
            H = theta.reshape((nx + nu, nx + nu))
            H_ux = H[nx:, :nx]
            H_uu = H[nx:, nx:]
            H_uu = 0.5 * (H_uu + H_uu.T) + 1e-6 * np.eye(nu)
            K_est = -np.linalg.solve(H_uu, H_ux)
            hist_K.append(K_est)

    # final H and policy
    H = theta.reshape((nx + nu, nx + nu))
    H = 0.5 * (H + H.T)
    H_ux = H[nx:, :nx]
    H_uu = H[nx:, nx:]
    H_uu = 0.5 * (H_uu + H_uu.T) + 1e-6 * np.eye(nu)
    K_learned = -np.linalg.solve(H_uu, H_ux)

    return K_learned, H, hist_K
