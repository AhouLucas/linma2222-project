
import numpy as np
from scipy.linalg import block_diag
from scipy.optimize import minimize, LinearConstraint, Bounds
# z = FF x(t) + GG q
# w = HH z + EE q
def precompute_mpc_condensed_matrices(N, model, y_min, y_max, u_min, u_max):
    F, G, H, E, D, Q, R, S = model
    R_0 = Q

    nx = F.shape[0]; nu = G.shape[1]; ny = H.shape[0]

    # --- Lifted dynamics for stages 0..N-1 (cost) and terminal N ---
    # Zstages = FF x0 + GG q, where Zstages = [z0; z1; ...; z_{N-1}]
    FF = np.vstack([np.linalg.matrix_power(F, i) for i in range(N+1)])  # (nx*(N+1)) x nx
    GG = np.zeros((nx*(N+1), nu*N))
    for i in range(1,N+1):
        for j in range(i):
            GG[i*nx:(i+1)*nx, j*nu:(j+1)*nu] = np.linalg.matrix_power(F, i-1-j) @ G

    # --- Quadratic cost 0.5 q^T Hq q + f^T q (+ const) ---
    # Qblk = np.kron(np.eye(N), Q)                 # diag(Q,...,Q)
    Qblk = block_diag(*([Q] * N + [R_0]))  # (nx*N) x (nx*N)
    Rblk = np.kron(np.eye(N), R)                 # diag(R,...,R)
    Sblk = np.zeros((nx*(N+1), nu*N))
    for i in range(N):
        Sblk[i*nx:(i+1)*nx, i*nu:(i+1)*nu] = S
    H_hat = 2*(GG.T @ Qblk @ GG + Rblk) + 2* (GG.T @ Sblk + Sblk.T @ GG)      # Hessian
    F_hat = 2*((GG.T @ Qblk @ FF) + (Sblk.T @ FF))     # linear term

    # --- Output constraints for k=0..N: y_k = H z_k + E q_k ---
    HH = np.kron(np.eye(N+1), H)  # (ny*(N+1)) x (nx*(N+1))
    EE = np.kron(np.eye(N+1), E)  # (ny*(N+1)) x (nu*(N+1))
    Yq = HH @ GG + EE[:, :nu*N]  # (ny*(N+1)) x (nu*N)
    yx = (HH @ FF)               # (ny*(N+1),)

    A_lin = np.vstack([np.eye(nu*N), Yq]) # output bounds and input bounds


    # Bounds helpers
    def arr_or_inf(v, d, lo):
        if v is None: return np.full(d, -np.inf if lo else np.inf)
        v = np.array(v, dtype=object).reshape(-1)
        out = np.where(v==None, -np.inf if lo else np.inf, v.astype(float))
        return out if out.size==d else np.tile(out, d//out.size)

    # input bounds (repeat over horizon)
    lb_u = np.tile(arr_or_inf(u_min, nu, True),  N)
    ub_u = np.tile(arr_or_inf(u_max, nu, False), N)

    lb_y = np.tile(arr_or_inf(y_min, ny, True),  N+1) 
    ub_y = np.tile(arr_or_inf(y_max, ny, False), N+1)

    return H_hat, F_hat, yx, A_lin, lb_u, ub_u, lb_y, ub_y

def mpc_condensed(x0, N, precomputed_condensed=None):
    ### using x0
    x0 = np.asarray(x0, float).reshape(-1)
    # global precomputed_condensed
    H_hat, F_hat, yx, A_lin, lb_u, ub_u, lb_y, ub_y = precomputed_condensed



    # Combine linear constraints:
    #   lb_u <= I q <= ub_u
    #   lb_y <= Yq q <= ub_y
    # output bounds (for k=0..N), then shift by -yx @ x0
    lb    = np.concatenate([lb_u, lb_y - yx @ x0])
    ub    = np.concatenate([ub_u, ub_y - yx @ x0])
    lin_con = LinearConstraint(A_lin, lb, ub)

    # Variable bounds
    bounds = Bounds(lb_u, ub_u)

    F_hat_x = F_hat @ x0
    def obj(q): return 0.5*q@H_hat@q + F_hat_x@q
    def jac(q): return H_hat@q + F_hat_x
    def hess(q): return H_hat

    # q0 = np.zeros(nu*N)
    q0 = np.zeros(H_hat.shape[0])
    res = minimize(obj, q0, method="trust-constr", jac=jac, hess=hess,
                   constraints=[lin_con], bounds=bounds,
                   options=dict(maxiter=10000, gtol=1e-8, xtol=1e-10, verbose=0))
    if not res.success:
        raise RuntimeError(f"MPC condensed QP failed: {res.message}")
    return res.x  # length nu*N

def _mpc_condensed(x0, N, precomputed_condensed=None):
    # if x0 is batched, loop over
    q = np.array([mpc_condensed(x0[:, i], N, precomputed_condensed) for i in range(x0.shape[1])]).T
    return q[0]

def get_mpc_condensed_policy(N, model, y_min, y_max, u_min=[-1], u_max=[1]):
    precomputed_condensed = precompute_mpc_condensed_matrices(N, model, y_min, y_max, u_min, u_max)
    return lambda x: _mpc_condensed(x, N, precomputed_condensed)