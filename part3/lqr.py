

import numpy as np
from scipy.linalg import solve_discrete_are

def solve_ricatti_infinite_horizon(F, G, Q, R, S, max_iterations=10_000, tolerance=1e-10):
    """Solve the discrete-time algebraic Riccati equation for infinite horizon LQR.
    """
    # Initialize with Q (not zero) for better convergence
    P = Q.copy()
    for i in range(max_iterations):
        # CORRECT Riccati iteration for MINIMIZATION problem
        P_next = Q + F.T @ P @ F - (F.T @ P @ G + S) @ np.linalg.inv(R + G.T @ P @ G) @ (G.T @ P @ F + S.T)

        if np.abs(np.max(P - P_next)) < tolerance:
            print(f"Converged after {i+1} iterations")
            break
        P = P_next
    else:
        print("Warning: Riccati iteration did not converge")
    
    return P

def optimal_gain(A, B, Q, R, S, M):
    K = np.linalg.inv(R + B.T @ M @ B) @ (B.T @ M @ A + S.T)
    return K


def compute_lqr_gain(model):
    F, G, H, E, D, Q, R, S = model
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


    # Compute average approximate reward
    J = - np.linalg.trace(M @ D @ D.T)
    print("Average approximate reward J:", J)

    return K_lqr