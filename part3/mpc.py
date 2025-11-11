

import numpy as np

from scipy.optimize import minimize, LinearConstraint, Bounds

def precompute_mpc_matrices(model, N, y_min, y_max, u_min, u_max):
    F, G, H, E, D, Q, R, S = model
    R_0 = Q

    nx = F.shape[0]; nu = G.shape[1]; ny = H.shape[0]
    nZ, n = (N+1)*nx, (N+1)*nx + N*nu
    z = lambda k: slice(k*nx, (k+1)*nx)
    q = lambda k: slice(nZ + k*nu, nZ + (k+1)*nu) # slices in w

    # --- Quadratic cost matrix : min w.T [Q, S; S.T, R] w  with w = [z; q] ---
    Hobj = np.zeros((n, n))
    for k in range(N):
        Hobj[z(k), z(k)] += Q
        Hobj[q(k), q(k)] += R
        Hobj[z(k), q(k)] += S
        Hobj[q(k), z(k)] += S.T
    Hobj[z(N), z(N)] += R_0
    Hobj = 0.5 * (Hobj + Hobj.T)

    # --- Equality constraints: x(t+1) = F x(t) + G u(t) ---
    Aeq = np.zeros((nx*(N+1), n))
    Aeq[:nx, z(0)] = np.eye(nx)
    for k in range(N):
        Aeq[z(k+1), z(k+1)] = np.eye(nx)
        Aeq[z(k+1), z(k)]   = -F
        Aeq[z(k+1), q(k)]   = -G

    # --- Output constraints : y_min <= H z + E u <= y_max ---
    ymin = np.array([float(v) if v is not None else -np.inf for v in y_min])
    ymax = np.array([float(v) if v is not None else  np.inf for v in y_max])
    lb, ub = np.tile(ymin, N+1), np.tile(ymax, N+1)
    Aineq = np.zeros(((N+1)*ny, n)) # H z + E u <= y_max
    for k in range(N+1):
        Aineq[k*ny:(k+1)*ny, z(k)] = H
    for k in range(N):
        Aineq[k*ny:(k+1)*ny, q(k)] = E

    # --- Variable bounds (only q) ---
    umin = np.array([float(v) if v is not None else -np.inf for v in u_min])
    umax = np.array([float(v) if v is not None else  np.inf for v in u_max])
    lb_q, ub_q = np.full(n, -np.inf), np.full(n, np.inf)
    lb_q[nZ:], ub_q[nZ:] = np.tile(umin, N), np.tile(umax, N)

    return Hobj, Aeq, Aineq, lb, ub, lb_q, ub_q, n, nx, nu, z, q

def solve_mpc(x0, N, precomputed_matrices):
    Hobj, Aeq, Aineq, lb, ub, lb_q, ub_q, n, nx, nu, z, q = precomputed_matrices
    x0 = np.asarray(x0, dtype=float).reshape(-1)

    beq = np.zeros(nx*(N+1))
    beq[:nx] = x0
    lin_eq = LinearConstraint(Aeq, beq, beq)    # Aeq w = beq
    lin_ineq = LinearConstraint(Aineq, lb, ub)  # lb <= Aineq w <= ub
    bounds = Bounds(lb_q, ub_q)                 # lb_q <= q <= ub_q

    # --- Initial guess ---
    w0 = np.zeros(n)
    w0[z(0)] = x0

    # --- Solve ---
    # jac and hess are provided for efficiency
    res = minimize(lambda q: 0.5 * q @ (Hobj @ q), w0, method="trust-constr", jac= lambda q: Hobj @ q, hess=lambda q: Hobj,
                   constraints=[lin_eq, lin_ineq], bounds=bounds,
                   options=dict(maxiter=10000, gtol=1e-8, verbose=0))
    if not res.success:
        raise RuntimeError(f"MPC solve failed: {res.message}")
    return res.x[q(0)].reshape(nu)


def _solve_mpc(x0, N, precomputed_matrices=None):
    # q = solve_mpc(x0, N)
    q = np.array([solve_mpc(x0[:, i], N, precomputed_matrices) for i in range(x0.shape[1])]).T
    return q[0]


def get_mpc_policy(N, model, y_min, y_max, u_min, u_max):
    precomputed_matrices = precompute_mpc_matrices(model, N, y_min, y_max, u_min, u_max)
    return lambda x: _solve_mpc(x, N, precomputed_matrices)














##############################

# # %pip install cvxpy
# import cvxpy as cp
#### MPC via CVXPY not working because problem is not DCP
# def solve_mpc(x0, N, nu, nx, ny, F, G, H, E, Q, R, S, R_0,
#               u_min, u_max, y_min, y_max):
#     ###### solve
#     # min z.T @ R_0 @ z + sum (z.T @ Q @ z + q.T @ R @ q + z.T @ S @ u)
#     # z_0 = x(t)
#     # z_k+1 = F z_k + G q_k
#     # w_k = H z_k + E q_k
#     # u_min <= q_k <= u_max
#     # y_min <= w_k <= y_max

#     P = np.block([[Q,  S],
#                   [S.T, R]])

#     # Symmetrize weights (numerical safety)
#     Q  = 0.5 * (Q + Q.T)
#     R  = 0.5 * (R + R.T)
#     R_0 = 0.5 * (R_0 + R_0.T)
#     P  = 0.5 * (P + P.T)

#         # Require R ≻ 0 (needed for the completion)
#     # If R is scalar, promote; otherwise check PD
#     try:
#         R_inv = np.linalg.inv(R)
#     except np.linalg.LinAlgError:
#         raise ValueError("R must be positive definite (invertible) for the cross-term rewrite.")

#     # ---- CVXPY variables ----
#     z = [cp.Variable(nx) for _ in range(N + 1)]
#     q = [cp.Variable(nu) for _ in range(N)]

#     constraints = [z[0] == np.asarray(x0, dtype=float).reshape(nx)]

#     # objective = cp.quad_form(z[N], R_0)
#     objective = 0

#     for k in range(N):
#         # cost = z[k].T @ Q @ z[k] + q[k].T @ R @ q[k] + 2 * z[k].T @ S @ q[k]
#         # objective += cp.quad_form(z[k], Q) + cp.quad_form(q[k], R) + 2 * cp.quad_form(z[k], S @ q[k])
#         v_k = cp.hstack([z[k], q[k]])            # length nx + nu
#         objective += cp.quad_form(v_k, P)

#         # dynamics
#         constraints += [z[k + 1] == F @ z[k] + G @ q[k]]
        
#         # input bounds
#         for i in range(nu):
#             if u_min[i] is not None: constraints += [q[k][i] >= float(u_min[i])]
#             if u_max[i] is not None: constraints += [q[k][i] <= float(u_max[i])]
#         # output bounds
#         w_k = H @ z[k] + E @ q[k]
#         for i in range(ny):
#             if y_min[i] is not None: constraints += [w_k[i] >= float(y_min[i])]
#             if y_max[i] is not None: constraints += [w_k[i] <= float(y_max[i])]

#     prob = cp.Problem(cp.Minimize(objective), constraints)

#     # Prefer OSQP for QPs; fall back to default if unavailable.
#     # try:
#     #     prob.solve(solver=cp.OSQP, warm_start=True, eps_abs=1e-6, eps_rel=1e-6, max_iter=20000)
#     #     # Problem does not follow DCP rules. Use other solver.
#     # except Exception:
#     #     prob.solve(warm_start=True)


#     # Try nonconvex-compatible solvers
#     solved = False
#     for solver in [cp.GUROBI, cp.CPLEX, cp.MOSEK, cp.SCIPY]:
#         try:
#             prob.solve(solver=solver, warm_start=True, verbose=False, 
#             solved = prob.status in ("optimal", "optimal_inaccurate")
#             if solved:
#                 print(f"✅ Solved with {solver}")
#                 break
#         except Exception as e:
#             print(f"⚠️ Solver {solver} failed: {e}")
#             continue


#     if not solved:
#         raise RuntimeError(f"MPC QP did not solve: status={prob.status}")

#     if prob.status not in ("optimal", "optimal_inaccurate"):
#         raise RuntimeError(f"MPC QP did not solve: status={prob.status}")

#     q0 = q[0].value
#     return np.asarray(q0, dtype=float).reshape(nu)