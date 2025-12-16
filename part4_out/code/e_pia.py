import numpy as np
from scipy.linalg import solve_discrete_lyapunov

def compute_E_PIA(model, max_iterations=1000, tolerance=1e-10, K_init=[-0.5, 0.5, 0.5]):
    F, G, H, E, D, Q, R, S = model
    K = np.asarray(K_init, dtype=float).reshape(1, -1)
    K_list = [K.copy()]

    for _ in range(max_iterations):
        # Policy evaluation
        A_k = F + G @ K           
        Q_k = (Q + S @ K + (S @ K).T + K.T @ R @ K)  

        # eigvals = np.linalg.eigvals(A_k)
        # if np.max(np.abs(eigvals)) >= 1.0:
        #     print("Unstable policy encountered during PIA.")
        #     break

        # Solve P_k = Q_k + A_k' P_k A_k
        P_k = solve_discrete_lyapunov(A_k.T, Q_k)

        # Policy improvement: K_k+1 = - (R + G' P_k G)^(-1) (S' + G' P_k F)
        K_new = -np.linalg.solve(R + G.T @ P_k @ G, S.T + G.T @ P_k @ F)

        # Check convergence
        if np.linalg.norm(K_new - K, ord='fro') < tolerance:
            K = K_new
            K_list.append(K.copy())
            break

        K = K_new
        K_list.append(K.copy())

    return K_list

